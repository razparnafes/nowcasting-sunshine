import torch

import numpy as np
import datetime
from torchvision import transforms
from PIL import Image
import xlrd
import os
import sys
import numpy.ma as ma
import pyproj
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
import time



## Base Generic Loader Object

class TimeSeriesDataLoader(object):
    """
    All of our data conforms with taking a start timestamp and end timestamp and returning everything in between. 
    So this piece of generic code sits in here. Each dataloader needs to implement the `_fetch_data` function, 
    which takes a timestamp as input and returns data specific to its loader.
    """
    def __init__(self, db_path, time_step):
        """
        :param db_path: Path to root dir of the DB
        :param time_step: Sample time of time series in seconds 
        """
        self._db_path = db_path
        self._time_step = datetime.timedelta(seconds=time_step)
    
    def _fetch_data(self, timestamp):
        """
        get timestamp and return the specific data in this time 
        supposed to be implemented in the inheritor class
        """
        raise NotImplementedError
    
    def __getitem__(self, timestamps):
        try:
          if type(timestamps) == tuple:
              start_date, end_date = timestamps

              # Number of timesteps in our time series
              timestep_count = (end_date - start_date).seconds / self._time_step.seconds
              timestep_count = int(np.floor(timestep_count))

              time_series = [self._fetch_data(t) for t in (start_date + self._time_step * n for n in range(timestep_count))]

              return torch.stack(time_series)
          else:
              # For cases where we'd like to sample a single value from our time series
              return self._fetch_data(timestamps)
        except:
              return None

    def preload_and_store(self, start_date, end_date, output_dir, window_size=1):
        """
        loads each item in the range as a tensor, and saved it to a file in output_dir.
        this allows us to preprocess data ahead of time (like cropping the images for the cloudmasks)
        """
        #todo: add support for saving time windows and not single timestamps
        if window_size != 1:
            raise NotImplementedError

        # Number of timesteps in our range
        timestep_count = (end_date - start_date).total_seconds() / self._time_step.seconds
        timestep_count = int(np.floor(timestep_count))

        print(timestep_count)
        
        for n in range(timestep_count):
            try:
              timestamp = start_date + self._time_step * n
              tensor = self._fetch_data(timestamp)
              file_name = f'{timestamp.strftime("%Y%m%d%H%M")}.pt'
              full_path = os.path.join(output_dir, file_name)
              print("hehe")
              torch.save(tensor.clone(), full_path)
            except:
              pass


class SatelliteImageLoader(TimeSeriesDataLoader):
    def _fetch_data(self, timestamp):
        file_name = f'{timestamp.strftime("%Y%m%d%H%M")}.png'
        full_path = os.path.join(self._db_path, file_name)

        pil_image = Image.open(full_path)
        tensor_transform = transforms.ToTensor()

        return tensor_transform(pil_image)


class RadiationPowerLoader(TimeSeriesDataLoader):
    ROW_WHERE_DATA_STARTS = 5
    RADIATION_MEASUREMENT_COLOUMN = 2

    def __init__(self, db_path, time_step):
        super().__init__(db_path, time_step)

        # Cache per day workbook, no need to reload it on every timestamp
        self._workbook_cache = {}

    def __get_workbook(self, timestamp):
        file_name = f'Global_radiation_{timestamp.strftime("%Y%m%d")}.xls'
        if file_name in self._workbook_cache:
            return self._workbook_cache[file_name]

        full_path = os.path.join(self._db_path, file_name)
        workbook = xlrd.open_workbook(full_path)

        self._workbook_cache[file_name] = workbook
        return workbook

    def _fetch_data(self, timestamp):
      try:
        workbook = self.__get_workbook(timestamp)
        sheet = workbook.sheet_by_index(0)
        row_index = timestamp.hour * 60 + timestamp.minute

        cell = sheet.cell(self.ROW_WHERE_DATA_STARTS + row_index, self.RADIATION_MEASUREMENT_COLOUMN)
        return torch.tensor([[cell.value]])
      except:
        return None


class CloudMaskLoader(TimeSeriesDataLoader):
    # Israel coordinates
    X1 = 34.
    Y1 = 29.5

    X2 = 36.
    Y2 = 33.5

    @staticmethod
    def lc2yxgdal(lines=None, columns=None, GT=None):
        '''
        :param lines: position of the raster lines
        :param columns: position of the raster columns
        :param GT: Geotransform GDAL table
        :return: Xgdal, Ygdal corresponding to the intermediate coordinates
        lines & columns have to be matrices with the raster size
        '''
        Ygdal = GT[3] + lines * GT[5] + columns * GT[4]
        Xgdal = GT[0] + columns * GT[1] + lines * GT[2]
        return Ygdal, Xgdal

    @staticmethod
    def crop_area_irregular_grid(PSI,lon,lat,x1,x2,y1,y2):
        mask_lon = np.logical_and(x1<=lon,lon<=x2)
        mask_lat = np.logical_and(y1<=lat,lat<=y2)
        
        mask = np.logical_and(mask_lon==1,mask_lat==1)  

        idx_y,idx_x = np.where(mask==True)

        idx_y_min = min(idx_y)
        idx_y_max = max(idx_y)
        
        idx_x_min = min(idx_x)
        idx_x_max = max(idx_x)
        
        PSI_cut = PSI[idx_y_min:idx_y_max,idx_x_min:idx_x_max]
        lon_cut = lon[idx_y_min:idx_y_max,idx_x_min:idx_x_max]
        lat_cut = lat[idx_y_min:idx_y_max,idx_x_min:idx_x_max]

        return PSI_cut,lon_cut,lat_cut
    
    def obtain_pixel_center(self, nc_fid, col_correction=0.5, lin_correction=0.5):
        '''
        please see reference at: www.gdal.org/gdal_datamodel.html
        this module uses corrected GT avoiding nx and ny
        the correction is done through coff and loff following the formulae:
        GT[0] = -1 * (coff  - 0.5) * GT[1]
        GT[3] = -1 * (loff  - 0.5) * GT[5]
        i.e the pixel size is asumed as correct in GT but the coordinate of the left top
        corner of the raster is recalculed. This calculation has to be suppressed when the bug will be fixed
        nc_fid: netcdf dataset
        col_correction=0 and lin_correction=0 correspond to the upper left corner of the pixel, to calculate the center
        of the pixel set to 0.5
        '''

        number_of_lines = nc_fid.dimensions['ny'].size
        number_of_columns = nc_fid.dimensions['nx'].size
        
        lines_matrix = np.zeros((number_of_lines, number_of_columns))
        for i in range(number_of_lines):
            lines_matrix[i, :] = i + lin_correction

        columns_matrix = np.zeros((number_of_lines, number_of_columns))
        for i in range(number_of_columns):
            columns_matrix[:, i] = i + col_correction
        # reading the gdal projection
        geodef = nc_fid.getncattr('gdal_projection')
        #
        gdal_proj = pyproj.Proj(geodef, errcheck=True)
        GT = nc_fid.getncattr("gdal_geotransform_table")
        # reading cgms, begining of the correction TBD: suppress
        gsmdef = nc_fid.getncattr('cgms_projection')

        proj_gsm_data = gsmdef.split(' +')
        proj_gsm_data_value = {}
        for element in proj_gsm_data:
            key = element.split('=')[0]
            try:
                val = float(element.split('=')[1])
            except:
                val = element.split('=')[1]
            proj_gsm_data_value[key] = val
        coff = proj_gsm_data_value['coff']
        loff = proj_gsm_data_value['loff']

        # correcting GT table in acordance with pixel size and offset

        GT[0] = -1 * (coff - 0.5) * GT[1]
        GT[3] = -1 * (loff - 0.5) * GT[5]
        # end of the GT correction
        # calculating the Y and X of the intermediate coordinates
        Ygdal_matrix, Xgdal_matrix = self.lc2yxgdal(lines=lines_matrix, columns=columns_matrix, GT=GT)


        lon, lat = gdal_proj(Xgdal_matrix, Ygdal_matrix, inverse=True)
        lons = ma.array(lon)
        lons = ma.masked_equal(lons, 1e30)

        lats = ma.array(lat)
        lats = ma.masked_equal(lats, 1e30)
        # returning masked lons lats
        return lons, lats

    def _fetch_data(self, timestamp):
        file_name = f'S_NWC_CMA_MSG4_MSG-N-VISIR_{timestamp.strftime("%Y%m%d")}T{timestamp.strftime("%H%M%S")}Z.nc'
        full_path = os.path.join(self._db_path, file_name)

        nc_cma = netCDF4.Dataset(full_path)
        CMA = nc_cma.variables['cma']

        # For any questions about this, ask Ori
        sat_lon, sat_lat = self.obtain_pixel_center(nc_cma)
        CMA_cut,sat_lon_cut,sat_lat_cut = self.crop_area_irregular_grid(CMA,sat_lon,sat_lat, self.X1, self.X2, self.Y1, self.Y2)

        CMA_cut = np.round(CMA_cut,1)
        CMA_vector = CMA_cut.flatten()

        return torch.tensor(CMA_vector)


class TensorDataLoader(TimeSeriesDataLoader):
    def _fetch_data(self, timestamp):
      try:
        file_name = f'{timestamp.strftime("%Y%m%d%H%M")}.pt'
        full_path = os.path.join(self._db_path, file_name)

        ret = torch.load(full_path)
        return ret
      except:
        return None

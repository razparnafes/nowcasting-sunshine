import os
import datetime
import time

import torch
from torch.utils.data import Dataset
import utils

import functools

import numpy as np

import tqdm


class CloudMaskRadiationLoader(Dataset):
    CLOUD_MASK_SAMPLE_TIME = 60 * 15
    RADIATION_SAMPLE_TIME = 60 * 1

    def __init__(self, cms_data_loader, radiation_data_loader, start_time, end_time, cms_window_size, cms_window_step,
                 cms_transform=None, radiation_transform=None, window_count=None):
        """
        :param root_dir: Dataset base path
        :param start_time: Time where dataset starts from. Base reference time
        :param cms_window_size: Cloud Mask window size
        :param cms_window_step: Cloud Mask window step
        :param cms_transform: Cloud Mask transform function
        :param radiation_transform: Radiation transform function
        """
        super().__init__()

        # Initialize dataloaders
        self._cms_data_loader = cms_data_loader
        self._radiation_data_loader = radiation_data_loader
        
        self._start_time = start_time
        self._end_time = end_time
        
        self._cms_window_size = cms_window_size
        self._cms_window_step = cms_window_step

        self._cms_transform = cms_transform
        self._radiation_transform = radiation_transform

        # Prefetch all the windows into RAM
        self._windows = []
        if window_count is None:
            self._window_count = int((self._end_time - self._start_time).total_seconds() / self.CLOUD_MASK_SAMPLE_TIME)
        else:
            self._window_count = window_count
        self._load_all_windows()
    
    def __len__(self):
        return self._window_count
    
    def _load_window(self, idx):
        # TODO: Implement window step
        # Add 2 hours because UTC
        cms_window_start = self._start_time + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * idx) + datetime.timedelta(hours=2)
        cms_window_end = cms_window_start + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * self._cms_window_size)

        cms_window = self._cms_data_loader[cms_window_start, cms_window_end]

        # The radiation we try to predict is the one at the end of the CMS window
        radiation = self._radiation_data_loader[cms_window_end]

        return cms_window, radiation
    
    def _load_all_windows(self):
        original_window_count = self._window_count
        for idx in tqdm.trange(len(self)):
            cms_window, radiation = self._load_window(idx)

            # Skip this sample if we couldn't load the window
            # Some of the data is missing so this'll happen from time to time
            if cms_window is None:
                self._window_count = self._window_count - 1
                continue
            self._windows.append((cms_window, radiation))
        print(f"[*] Pruned {original_window_count - self._window_count} windows")

    @functools.lru_cache(maxsize=300)
    def __getitem__(self, idx):
        
        # Just load everything to RAM instead
        # cms_window, radiation = self._load_window(idx)

        cms_window, radiation = self._windows[idx]

        # Pass data through preprocessors
        if self._cms_transform:
            cms_window = self._cms_transform(cms_window)
        if self._radiation_transform:
            radiation = self._radiation_transform(radiation)

        return cms_window, radiation

class CloudMaskGrayScaleTimeStampRadiationLoader(Dataset):
    CLOUD_MASK_SAMPLE_TIME = 60 * 15
    GRAYSCALE_SAMPLE_TIME = 60 * 5
    RADIATION_SAMPLE_TIME = 60 * 1

    def __init__(self, cms_data_loader, grayscale_data_loader, radiation_data_loader, start_time, end_time, window_size, window_step,
                 prediction_step, cms_transform=None, grayscale_transform=None, timestamp_transform=None, radiation_transform=None,
                 window_count=None):
        """
        :param root_dir: Dataset base path
        :param start_time: Time where dataset starts from. Base reference time
        :param window_size: Cloud Mask window size
        :param window_step: Cloud Mask window step
        :param transform: Cloud Mask transform function
        :param radiation_transform: Radiation transform function
        """
        super().__init__()

        # Initialize dataloaders
        self._cms_data_loader = cms_data_loader
        self._grayscale_data_loader = grayscale_data_loader
        self._radiation_data_loader = radiation_data_loader
        
        self._start_time = start_time
        self._end_time = end_time
        
        self._window_size = window_size
        self._window_step = window_step

        self._cms_transform = cms_transform
        self._grayscale_transform = grayscale_transform
        self._radiation_transform = radiation_transform
        self._timestamp_transform = timestamp_transform

        self._prediction_step = prediction_step

        # Prefetch all the windows into RAM
        self._windows = []
        if window_count is None:
            self._window_count = int((self._end_time - self._start_time).total_seconds() / self.CLOUD_MASK_SAMPLE_TIME)
        else:
            self._window_count = window_count
        self._load_all_windows()
    
    def __len__(self):
        return self._window_count
    
    def _load_window(self, idx):
        # TODO: Implement window step
        # Add 2 hours cuz UTC+2
        window_start = self._start_time + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * idx) + datetime.timedelta(hours=2)
        window_end = window_start + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * self._window_size)

        cms_window = self._cms_data_loader[window_start, window_end]
        grayscale_window = self._grayscale_data_loader[window_start, window_end]

        # Calculate timestamps. Use CMS sampling frequency because this is how far apart our objects are in time
        # pretty much the same calcuation done in the timeseries data loader
        timestep_count = (window_end - window_start).total_seconds() / self.CLOUD_MASK_SAMPLE_TIME
        timestep_count = int(np.floor(timestep_count))

        # Conver to UNIX timestamp
        timestamps = [time.mktime(t.timetuple()) for t in (window_start + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME) * n for n in range(timestep_count))]
        timestamps = torch.tensor(timestamps)

        # The radiation we try to predict is the one at the end of the CMS window
        # Also, the radiation data is in UTC+2, while the CMS is in UTC, so gotta make up
        # for that..
        radiation_prediction = self._radiation_data_loader[window_end + self._prediction_step]

        # Radiation at end of window
        radiation = self._radiation_data_loader[window_end]

        # Calculated prediction on clear day for prediction time
        location = {"lat": 31.2716, "lon": 34.38941} # Ashalim
        cos_alpha_window_end = utils.cossza(window_end, location)
        cos_alpha_prediction = utils.cossza(window_end + self._prediction_step, location)
        dummy_radiation = (cos_alpha_prediction / cos_alpha_window_end) * radiation

        return cms_window, grayscale_window, timestamps, radiation, dummy_radiation, radiation_prediction
    
    def _load_all_windows(self):
        original_window_count = self._window_count
        for idx in tqdm.trange(len(self)):
            cms_window, grayscale_window, timestamps, radiation, dummy_radiation, radiation_prediction = self._load_window(idx)

            # Skip this sample if we couldn't load the window
            # Some of the data is missing so this'll happen from time to time
            if cms_window is None or grayscale_window is None:
                self._window_count = self._window_count - 1
                continue
            self._windows.append((cms_window, grayscale_window, timestamps, radiation, dummy_radiation, radiation_prediction))
        print(f"[*] Pruned {original_window_count - self._window_count} windows")

    @functools.lru_cache(maxsize=300)
    def __getitem__(self, idx):
        
        # Just load everything to RAM instead
        # cms_window, radiation = self._load_window(idx)

        cms_window, grayscale_window, timestamps, radiation, dummy_radiation, radiation_prediction = self._windows[idx]

        # Pass data through preprocessors
        if self._cms_transform:
            cms_window = self._cms_transform(cms_window)
        if self._grayscale_transform:
            grayscale_window = self._grayscale_transform(grayscale_window)
        if self._timestamp_transform:
            timestamps = self._timestamp_transform(timestamps)
        if self._radiation_transform:
            radiation = self._radiation_transform(radiation)
            radiation_prediction = self._radiation_transform(radiation_prediction)
            dummy_radiation = self._radiation_transform(dummy_radiation)

        return {
            "cms_window": cms_window,
            "grayscale_window": grayscale_window,
            "timestamps": timestamps,
            "radiation_prediction": radiation_prediction,
            "dummy_radiation": dummy_radiation,
            "radiation": radiation
        }

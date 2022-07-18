
# We have all kinds of models we'd like to try, and each one treats the data quite differently 
# (what's considered a label and what's considered an input). 
# So we're gonna build the different data loaders over here (that use the dataloaders defined above ofc).


import os
from datetime import datetime, timedelta

import torch
from torch.utils.data import Dataset

import functools


class CloudMaskRadiationLoader(Dataset):
    """
    Input is cloudmask, label is radiation.
    Since we take as input a series of cloudmask images, we need to specify a window size and window step.
    """
    CLOUD_MASK_SAMPLE_TIME = 60 * 15
    RADIATION_SAMPLE_TIME = 60 * 1

    def __init__(self, radiation_data_dir, cloadmask_tenzor_dir, start_time, end_time, cms_window_size, cms_window_step,
                 cms_transform=None, radiation_transform=None):
        """
        :param root_dir: Dataset base path
        :param start_time: Time where dataset starts from. Base reference time
        :param cms_window_size: Cloud Mask window size
        :param cms_window_step: Cloud Mask window step
        :param cms_transform: Cloud Mask transform function
        :param radiation_transform: Radiation transform function
        """
        # Initialize dataloaders
        self._cms_data_loader = TensorDataLoader(
            cloadmask_tenzor_dir, self.CLOUD_MASK_SAMPLE_TIME)
        self._radiation_data_loader = RadiationPowerLoader(
            radiation_data_dir, self.RADIATION_SAMPLE_TIME)
        
        self._start_time = start_time
        self._end_time = end_time
        
        self._cms_window_size = cms_window_size
        self._cms_window_step = cms_window_step

        self._cms_transform = cms_transform
        self._radiation_transform = radiation_transform
    
    def __len__(self):
        return int((self._end_time - self._start_time).total_seconds() / self.CLOUD_MASK_SAMPLE_TIME)

    @functools.lru_cache(maxsize=300)
    def __getitem__(self, idx):
        # TODO: Implement window step
        cms_window_start = self._start_time + timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * idx)
        cms_window_end = cms_window_start + timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * self._cms_window_size)

        cms_window = self._cms_data_loader[cms_window_start, cms_window_end]

        # The radiation we try to predict is the one at the end of the CMS window
        # Also, the radiation data is in UTC+2, while the CMS is in UTC, so gotta make up
        # for that..
        radiation = self._radiation_data_loader[cms_window_end + timedelta(hours=2)]

        # Pass data through preprocessors
        if self._cms_transform:
            self._cms_transform(cms_window)
        if self._radiation_transform:
            self._radiation_transform(radiation)

        return cms_window, radiation

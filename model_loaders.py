
# We have all kinds of models we'd like to try, and each one treats the data quite differently 
# (what's considered a label and what's considered an input). 
# So we're gonna build the different data loaders over here (that use the dataloaders defined above ofc).


import os
import datetime

import torch
from torch.utils.data import Dataset

import functools

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
        cms_window_start = self._start_time + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * idx)
        cms_window_end = cms_window_start + datetime.timedelta(seconds=self.CLOUD_MASK_SAMPLE_TIME * self._cms_window_size)

        cms_window = self._cms_data_loader[cms_window_start, cms_window_end]

        # The radiation we try to predict is the one at the end of the CMS window
        # Also, the radiation data is in UTC+2, while the CMS is in UTC, so gotta make up
        # for that..
        radiation = self._radiation_data_loader[cms_window_end + datetime.timedelta(hours=2)]

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

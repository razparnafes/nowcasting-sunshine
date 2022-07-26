import datetime
import calendar

import torch


class RadiationNormalizer(object):
    def __init__(self, radiation_data_loader, start_time, end_time):
        self._radiation_data_loader = radiation_data_loader
        
        self._start_time = start_time
        self._end_time = end_time

        self._mean, self._std = self._calculate_mean_std()
        print(f"[*] Radiation mean: {self._mean}, STD: {self._std}")

    def _calculate_mean_std(self):
        # Load all the data to RAM and calcualte the mean and std on it
        radiation_data = self._radiation_data_loader[self._start_time, self._end_time]

        return torch.std_mean(radiation_data)

    def __call__(self, x):
        return (x - self._mean) / self._std


class TimeStampNormalizer(object):

    def _normalize_timestamp(self, timestamps):
        """
        Create three distinct one-hots for the minute, hour, day and month
        """
        normalized_timestamps = []

        for timestamp in timestamps:
            timestamp = datetime.datetime.fromtimestamp(timestamp.int())

            days_in_month = calendar.monthrange(timestamp.year, timestamp.month)[1]
            day = torch.tensor((timestamp.day) / float(days_in_month), dtype=torch.float)
            hour = torch.tensor(timestamp.hour / 24., dtype=torch.float)
            minute = torch.tensor(timestamp.minute / 60., dtype=torch.float)
            
            normalized_timestamps.append(torch.stack([day, hour, minute]))

        return torch.stack(normalized_timestamps)

    def __call__(self, x):
        return self._normalize_timestamp(x)

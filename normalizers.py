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

import math
import torch

def cossza(timestamp, location):
  """
  	how to use:
  	location = {"lat": 31.2716, "lon": 34.38941} # Ashalim
  	radiation / cossza(timestamp, location)
  	"""
  PI= 3.14159265358979323846264338327950288
  time_zone=2  # Israel time zone UTC+2

  base_date = datetime.date(timestamp.year, 1, 1) # month and day are 1-base
  day_delta = (timestamp.date() - base_date).days + 1

  # calculate Solar Zenith Angle (SZA)
  gamma = (2*PI/365)*((day_delta-1)+(timestamp.hour - 12)/24)
  decl_angle = 0.006918-(0.399912*math.cos(gamma))+ 0.070257*math.sin(gamma)-0.006758*math.cos(2*gamma)+0.000907*math.sin(2*gamma)-0.002697*math.cos(3*gamma)+0.00148*math.sin(3*gamma)
  eqtime_min = 229.18*(0.000075+0.001868*math.cos(gamma)-0.032077*math.sin(gamma)-0.014615*math.cos(2*gamma)-0.040849*math.sin(2*gamma))
  time_offset = eqtime_min + 4*location["lon"] - 60*time_zone
  true_solar_time = timestamp.hour*60 + timestamp.minute + (timestamp.second/60) + time_offset
  solar_hr_angle = (true_solar_time/4) - 180
  SZA = math.acos(((math.sin(location["lat"]*PI/180.0)*math.sin(decl_angle))+(math.cos(location["lat"]*PI/180.0)*math.cos(decl_angle)*math.cos(solar_hr_angle*PI/180.0)))) * 180.0 / PI  # in degrees
  COS_SZA = math.cos(math.radians(SZA))
  return COS_SZA


def crop_israel_to_ashalim(israel_tensor):
  """
    get israel cloudtype or cloudmask tensor
    return tensor of ashalim cloudtype or cloudmask in the same time
    """
  start = 6111
  line = 92
  one_line = 13
  ct_size = 143
  subtensors = []
  for i in range(int(ct_size / one_line)):
    index = start + (line * i)
    subtensors.append(israel_tensor[index:index + one_line])
  ashalim_handmande_tensor = torch.cat(subtensors)
  return  ashalim_handmande_tensor
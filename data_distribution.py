import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis, jet, RdYlGn_r
from netCDF4 import Dataset as NetCDFFile
import netCDF4 as nc
import json
import xarray as xr

import os

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def random_window(array, size=32):
    # print(array.shape)
    x = np.random.randint(0, array.shape[0]-size)
    y = np.random.randint(0, array.shape[1]-size)
    
    return array[x:x+size, y:y+size]

def normalize_array(data):
    vmax = np.amax(data)
    vmin = np.min(data)
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    
    return normalized_data

north_america_box = -135,10,-65.0,60
europe_box = -10, 40, 35, 70
africa_box = -20, -35, 50, 40
asia_box = 40, 0, 150, 70
south_america_box = -80, -50, -40, 10
oceania_box = 115, -40, 155, -10

def sample_within_coordinates(array, coordinates, lon, lat):
    lonW, latS, lonE, latN = coordinates
    lonW = find_nearest(lon, lonW)
    lonE = find_nearest(lon, lonE)
    latS = find_nearest(lat, latS)
    latN = find_nearest(lat, latN)

    if latS > latN:
        return random_window(array[latN:latS, lonW:lonE])
    else:
        return random_window(array[latS:latN, lonW:lonE])

def bin_values(array, bins=11):
    binned_array = np.digitize(array, np.linspace(0, 1, bins))
    
    return binned_array

def gridify_than_normalize(data):
    sub_arrays_binned = []
    # data = np.asarray(data)
    for i in range(0, data.shape[0], 32):
        for j in range(0, data.shape[1], 32):
            sub_array = data[i:i+32, j:j+32]
            if sub_array[~sub_array.mask].size > 0:
                norm_sub_array = normalize_array(sub_array[~sub_array.mask])
                bin_samples = bin_values(norm_sub_array)
                bin_count = np.bincount(bin_samples)
                sub_arrays_binned.append(bin_count)
    return sub_arrays_binned

# regions = {
#     'Europe': europe_box,
#     'North America': north_america_box,
#     'South America': south_america_box,
#     'Oceania': oceania_box,
#     'Africa': africa_box,
#     'Asia': asia_box
# }

# region_bins = {region: [] for region in regions}

# for filename in os.listdir('../Point_source_detection/data/OCO-2/'):
#     file = xr.open_dataset('../Point_source_detection/data/OCO-2/'+filename)
#     xco2 = file['XCO2'][0,:,:]
#     lon = file['lon']
#     lat = file['lat']

#     for region_name, region_box in regions.items():
#         region_win = sample_within_coordinates(xco2, region_box, lon, lat)
#         norm_region_win = normalize_array(region_win)
#         bins_region = bin_values(norm_region_win)
#         region_card = np.bincount(bins_region.flatten())
#         region_bins[region_name].append(region_card)

# # Save as numpy array
# try:
#     np.save('region_bins.npy', region_bins)
# except:
#     print('Error saving oco2 numpy array')

# # Save as json file
# try:
#     with open('region_bins.json', 'w') as f:
#         json.dump({k: [arr.tolist() for arr in v] for k, v in region_bins.items()}, f)
# except:
#     print('Error saving oco2 json file')

# Print current location
print(os.getcwd())
modis_bins = []
for filename in os.listdir('../RDS/earthdata/MOD11C1_061-20241106_171429/'):
    file = nc.Dataset('../RDS/earthdata/MOD11C1_061-20241106_171429/'+filename, 'r')
    temp_data = file['LST_Day_CMG'][:]
    gridified_data = gridify_than_normalize(temp_data)
    filtered_gridified_data = [item for item in gridified_data if len(item) == 12]
    filtered_gridified_data = np.asarray(filtered_gridified_data)[:, 1:11]
    filtered_gridified_data_avg = np.mean(filtered_gridified_data, axis=0)
    modis_bins.append(filtered_gridified_data_avg)

try:    
    np.save('modis_bins.npy', modis_bins)
except:
    print('Error saving modis numpy array')
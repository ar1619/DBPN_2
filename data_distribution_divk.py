import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis, jet, RdYlGn_r
from netCDF4 import Dataset as NetCDFFile
import netCDF4 as nc
import json
import xarray as xr
import time
import os

import skimage as ski
from skimage.color import rgb2gray

def normalize_array(data):
    vmax = np.amax(data)
    vmin = np.min(data)
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    
    return normalized_data

def bin_values(array, bins=100):
    binned_array = np.digitize(array, np.linspace(0, 1, bins))
    
    return binned_array

def gridify_than_normalize(data):
    norm_arrays = [0 for i in range(100)]
    # data = np.asarray(data)
    for i in range(0, data.shape[0], 512):
        for j in range(0, data.shape[1], 512):
            sub_array = data[i:i+512, j:j+512]
            try:
                normed_sub_array = normalize_array(sub_array)
                bin_samples = bin_values(normed_sub_array).flatten()
                bin_count = np.bincount(bin_samples)
                norm_arrays = [x + y for x, y in zip(norm_arrays, bin_count)]
            except:
                print("Error dividing by zero")
    return norm_arrays

dota_dist = [0 for i in range(101)]
for filename in os.listdir('../RDS/DOTA/'):
    try:
        file = ski.io.imread('../RDS/DOTA/'+filename)
        file_gray = np.asarray(rgb2gray(file))
        normed_data = gridify_than_normalize(file_gray)
        dota_dist = [x + y for x, y in zip(dota_dist, normed_data)]
    except:
        print("Error reading file", filename)

np.save('dota_dist.npy', dota_dist)
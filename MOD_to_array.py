from netCDF4 import Dataset as NetCDFFile
import numpy as np
import os

ignore_map = [[2, 0], [2, 2], [3, 0], [4, 0], [4, 1],
                       [5, 0], [5, 1], [5, 3], [5, 4], [5, 5]]

def decompose_images(original_image, x_h, c_h, x_w, c_w):
    shape = (x_h, x_w)
    return np.lib.stride_tricks.sliding_window_view(original_image, shape)[::c_h, ::c_w]

def normalize_array(data):
    vmax = np.amax(data[np.nonzero(data)])
    vmin = np.amin(data[np.nonzero(data)])
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    threshold = - 1./range_data
    data = np.maximum(normalized_data,threshold)
    data[data == threshold] = -1
    
    return data

for filename in os.listdir('../MOD_Data/'):
    file = filename.replace('.hdf','').replace('.', '_')
    print(f"Preparing picture for {file}")
    filename = f"../MOD_Data/{filename}"
    file_destination = '../MOD_tensor/'+file
    
    nc = NetCDFFile(filename)
    LST = np.asarray(nc.variables['LST_Day_CMG'][:])

    decomposed_image = decompose_images(LST, 592, 500, 1200, 1000)
    print(f"Saving each sub-picture for {file}")
    for i in range(7):
        for j in range(7):
            if [i, j] not in ignore_map:
                sub_image = normalize_array(decomposed_image[i, j, ...])
                file_destination_part = file_destination+'_'+str(i)+'_'+str(j)
                np.save(file_destination_part, sub_image)
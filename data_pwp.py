from os import listdir
from os.path import join, split

import torch.utils.data as data
import torch
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def load_img(filepath):
    img = np.load(filepath)
    img = np.expand_dims(img, axis = 2)
    return img

def normalize_array(data):
    vmax = np.amax(data[np.nonzero(data)])
    vmin = np.amin(data[np.nonzero(data)])
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    
    return normalized_data

def array_centered_on_powerplant(xco2_array, boundaries):
    arrays = np.zeros((len(boundaries), 40, 32, 1))
    for i in range(len(boundaries)):
        if xco2_array[boundaries[i, 0]:boundaries[i, 2], boundaries[i, 1]:boundaries[i, 3]].shape == (40, 32, 1):
            arrays[i] = xco2_array[boundaries[i, 0]:boundaries[i, 2], boundaries[i, 1]:boundaries[i, 3]]
    return np.float32(arrays)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, boundaries, year):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.year = year
        self.boundaries = boundaries

    def __getitem__(self, index):
        _, file = split(self.image_filenames[index])
        if self.year in self.image_filenames[index]:
            input = load_img(self.image_filenames[index])
            input = normalize_array(input)
            input = array_centered_on_powerplant(input, self.boundaries)
            input = torch.from_numpy(input)
                
            return input, file
        else:
            input = load_img(self.image_filenames[index])
            input = torch.from_numpy(input)
            return input, file
      
    def __len__(self):
        return len(self.image_filenames)

def get_eval_set(lr_dir, upscale_factor, boundaries, year):
    return DatasetFromFolderEval(lr_dir, upscale_factor, boundaries, year)


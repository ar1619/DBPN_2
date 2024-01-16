import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from skimage.transform import pyramid_reduce
import random
import time
from random import randrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".npy"])

def normalize_array(data):
    vmax = np.amax(data)
    vmin = np.min(np.where(data >= 0, data, np.inf))
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    
    return normalized_data

def normalize_array_eval(data):
    vmax = np.amax(data)
    vmin = np.min(data)
    range_data = vmax - vmin
    
    normalized_data = (data - vmin)/range_data
    
    return normalized_data

def create_mask(data):
    """
    Returns a mask where the mask has the value 1 for non-negative elements
    and 0 for negative elements of the input array.

    Parameters:
    - data: A NumPy array.

    Returns:
    - A NumPy array of integers (0 and 1), where the shape is the same as the input array.
    """
    mask = (data >= 0).astype(int)
    return mask

def quantize_array(data, decimals):
    data = np.round(data, decimals)
    data = np.float32(data)
    return data

def noise_array(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise
    data = np.float32(data)
    return data

def load_img(filepath):
    img = np.load(filepath)
    img = normalize_array(img)
    img = np.expand_dims(img, axis = 2)
    img = np.float32(img)
    #y, _, _ = img.split()
    return img

def load_img_eval(filepath):
    img = np.load(filepath)
    img = normalize_array_eval(img)
    img = np.expand_dims(img, axis = 2)
    img = np.float32(img)
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.shape[0], img_in.shape[1]
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in[iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = np.flipud(img_in)
        img_tar = np.flipud(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = np.rot90(img_in)
            img_tar = np.rot90(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = np.rot90(img_in, k=2)
            img_tar = np.rot90(img_tar, k=2)
            info_aug['trans'] = True
            
    img_in = np.float32(img_in)
    img_tar = np.float32(img_tar)
    return img_in, img_tar, info_aug

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, upscale_factor, noise_level, noise, data_augmentation, decimals=5, quantize=False, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.noise_level = noise_level
        self.noise = noise
        self.decimals = decimals
        self.quantize = quantize
        if self.upscale_factor == 2:
            self.random_factor = np.random.choice([1, 2, 4, 8], len(self.image_filenames), p=[0.25, 0.25, 0.25, 0.25])
        elif self.upscale_factor == 4:
            self.random_factor = np.random.choice([1, 2, 4], len(self.image_filenames), p=[0.34, 0.33, 0.33])
        else:
            self.random_factor = 1

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        if self.upscale_factor == 2:
            random_factor = self.random_factor[index]
            if random_factor == 1:
                input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
            else:
                input = pyramid_reduce(target, downscale=random_factor*2, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
                target = pyramid_reduce(target, downscale=random_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)

        elif self.upscale_factor == 4:
            random_factor = self.random_factor[index]
            if random_factor == 1:
                input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
            else:
                input = pyramid_reduce(target, downscale=random_factor*4, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
                target = pyramid_reduce(target, downscale=random_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)

        else:
            input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
            if self.noise:
                input = noise_array(input, self.noise_level)
        
        input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)

        if self.quantize:
            input = quantize_array(input, self.decimals)
            target = quantize_array(target, self.decimals)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        
        mask = create_mask(target)
        if self.transform:
            input = self.transform(input.copy())
            target = self.transform(target.copy())
            mask = self.transform(mask.copy())
                
        return input, target, mask

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, out_dir, upscale_factor, decimals, quantize=False, transform=None, shuffle=False):
        super(DatasetFromFolderEval, self).__init__()
        list_original = [filename for filename in listdir(lr_dir)]
        list_done = [filename for filename in listdir(out_dir)]
        missing_list = list(set(list_original) - set(list_done))
        self.image_filenames = [join(lr_dir, x) for x in missing_list if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.quantize = quantize
        self.decimals = decimals

        if shuffle:
            random.shuffle(self.image_filenames)
        else:
            self.image_filenames.sort()
    def __getitem__(self, index):
        input = load_img_eval(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        
        if self.quantize:
            input = quantize_array(input, self.decimals)

        if self.transform:
            try:
                input = self.transform(input)
            except:
                input = torch.tensor(input)
            
        # print(input.shape)
        return input, file
      
    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromFolderValid(data.Dataset):
    def __init__(self, hr_dir, upscale_factor, decimals=5, quantize=False, transform=None):
        super(DatasetFromFolderValid, self).__init__()
        self.image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.quantize = quantize
        self.decimals = decimals

        random.shuffle(self.image_filenames)

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='reflect', cval=0, channel_axis=2)
        _, file = os.path.split(self.image_filenames[index])

        if self.quantize:
            input = quantize_array(input, self.decimals)
        mask = create_mask(target)
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
            mask = self.transform(mask)
        return input, target, mask
      
    def __len__(self):
        return len(self.image_filenames)

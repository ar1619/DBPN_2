import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from skimage.transform import pyramid_reduce
import skimage
import random
from random import randrange
from maskedtensor import masked_tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".npy"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    img = np.array(img)
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
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug

def mask_input(inp_batch):
    nb_mask = inp_batch.size(0)
    
    for i in range(nb_mask):
        img = inp_batch[i]
        mask = torch.rand(img.size()) > 0.9
        #inp_batch[i] = masked_tensor(img, mask)
    return inp_batch
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, upscale_factor, data_augmentation, transform=None, observation_type="training", observation = None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        if self.upscale_factor == 2:
            self.random_factor = np.random.choice([1, 2, 4, 8], len(self.image_filenames), p=[0.25, 0.25, 0.25, 0.25])
        elif self.upscale_factor == 4:
            self.random_factor = np.random.choice([1, 2, 4], len(self.image_filenames), p=[0.34, 0.33, 0.33])
        else:
            self.random_factor = 1
        
        self.observation_type = observation_type
        self.observation = observation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        if self.upscale_factor == 2:
            random_factor = self.random_factor[index]
            if random_factor == 1:
                input = skimage.transform.pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
            else:
                input = skimage.transform.pyramid_reduce(target, downscale=random_factor*2, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
                target = skimage.transform.pyramid_reduce(target, downscale=random_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)

        elif self.upscale_factor == 4:
            random_factor = self.random_factor[index]
            if random_factor == 1:
                input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
            else:
                input = pyramid_reduce(target, downscale=random_factor*4, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
                target = pyramid_reduce(target, downscale=random_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)

        else:
            input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='constant', cval=0, channel_axis=2)
        
        input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        if self.observation_type == "training":
            self.observation = mask_input(target)
                
        return input, target, self.observation

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, obs_dir = "", observations = False, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.lr_dir = lr_dir
        self.observations = observations
        self.obs_dir = obs_dir
        self.image_filenames = [x for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.lr_dir+self.image_filenames[index])
        if self.observations:
            observation = load_img(self.obs_dir+self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        
        if self.transform:
            input = self.transform(input)
            if self.observations:
                observation = self.transform(observation)
                
        return input, observation
      
    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromFolderValid(data.Dataset):
    def __init__(self, hr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderValid, self).__init__()
        self.image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        input = pyramid_reduce(target, downscale=self.upscale_factor, sigma=None, order=3, mode='reflect', cval=0, channel_axis=2)
        _, file = os.path.split(self.image_filenames[index])
        
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        return input, target
      
    def __len__(self):
        return len(self.image_filenames)

from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolderValid, DatasetFromFolderOCO2

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, upscale_factor, noise_level, noise, patch_size, data_augmentation, decimals, quantize):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir,patch_size, upscale_factor, noise_level, noise, data_augmentation, decimals, quantize,
                             transform=transform())

def get_validation_set(data_dir, hr, upscale_factor, decimals, quantize):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolderValid(hr_dir, upscale_factor, decimals, quantize,
                             transform=transform())

def get_eval_set(lr_dir, out_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, out_dir, upscale_factor,
                             transform=transform())

def get_eval_set_uncert(lr_dir, out_dir, upscale_factor, noise):
    return DatasetFromFolderEval(lr_dir, out_dir, upscale_factor, noise, 
                             transform=transform())

def get_data_set(oco2_folder, out_dir):
    return DatasetFromFolderOCO2(oco2_folder, out_dir,
                             transform=transform())

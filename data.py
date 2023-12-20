from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolderValid

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, upscale_factor, noise_level, patch_size, data_augmentation, decimals, quantize):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir,patch_size, upscale_factor, noise_level, data_augmentation, decimals, quantize,
                             transform=transform())

def get_validation_set(data_dir, hr, upscale_factor):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolderValid(hr_dir, upscale_factor,
                             transform=transform())

def get_eval_set(lr_dir, out_dir, upscale_factor, quantize):
    return DatasetFromFolderEval(lr_dir, out_dir, upscale_factor, quantize,
                             transform=transform())


from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net as DBPN
from data import get_data_set
from functools import reduce

import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--quantize', type=bool, default=True)
parser.add_argument('--combination', type=int, default=2)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='XCO2')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--model', default='models/DBPNLL_x8.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
#print(opt)

def load_checkpoint(model):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model'])

    return model

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

#print('===> Loading datasets')
test_set = get_data_set(opt.input_dir, os.path.join(opt.output,opt.test_dataset))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

#print('===> Building model')
if opt.model_type == 'DBPNLL':
    model = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor, combination=opt.combination, tuning=True) ###D-DBPN

model= load_checkpoint(model)
#print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    sr_array = np.zeros((12, 20, 512, 512))
    model.eval()
    for batch in testing_data_loader:
        t0 = time.time()
        with torch.no_grad():
            input, min_max, name = Variable(batch[0]), batch[1], batch[2]

        input = input[0]
        if cuda:
            input = input.cuda(gpus_list[0])
        with torch.no_grad():
            for i in range(input.size(0)):
                for j in range(input.size(1)):
                    output = model(input[i][j])
                    output_array = output.cpu().data
                    sr_array[i][j] = output_array.squeeze().numpy()

        t1 = time.time()

        print("===> Processed: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_file(sr_array, min_max[0].numpy(), name[0])

def unnorm(img, min_max):
    """
    Unnormalize the image after super resolution
    :param img: super resolved CO2 patch
    :param min_max: min and max values of the original patch
    :return: unnormalized patch
    """
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # max value is in min_max[i,j,1] and min_value is in min_max[i,j,0]
            img[i][j] = img[i][j] * (min_max[i,j,1] - min_max[i,j,0]) + min_max[i,j,0]
    return img

def unslide(sr_array, strategie="mean"):
    """
    Reconstruct the image from the patches
    :param sr_array: super resolved patches
    :param strategie: mean or none
    :return: reconstructed image
    """
    mask = np.zeros((5792, 9328))
    reconstructed = np.zeros((5792, 9328))
    if strategie == "mean":
        for i in range(sr_array.shape[0]):
            for j in range(sr_array.shape[1]):
                reconstructed[i*480:i*480+512, j*464:j*464+512] += sr_array[i,j]
                mask[i*480:i*480+512, j*464:j*464+512] += 1
        reconstructed /= mask
        return reconstructed[:5776,64:9280]
    elif strategie == "none":
        for i in range(sr_array.shape[0]):
            for j in range(sr_array.shape[1]):
                reconstructed[i*480:i*480+512, j*464:j*464+512] = sr_array[i,j]
        return reconstructed[:5776,64:9280]

def save_file(img, min_max, img_name):
    unnorm_img = unnorm(img, min_max)
    reconstructed_img = unslide(unnorm_img, strategie="mean")
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    return np.save(save_fn, reconstructed_img)

eval()

# Divide original file into 32*32 patches / Original size 361*576
# Super resolve each patch
# Put each patch back together to form the original file in high resolution

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
#from dbpn_v1 import Net as DBPNLL
#from dbpn_iterative import Net as DBPNITER
from data import get_eval_set_uncert
from functools import reduce

#import scipy.io as sio
import time
#import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--noise', type=float, default=0.0, help='noise level')
parser.add_argument('--combination', type=int, default=2)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='XCO2')
parser.add_argument('--model_type', type=str, default='DBPNLL')
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
test_set = get_eval_set_uncert(os.path.join(opt.input_dir,opt.test_dataset), os.path.join(opt.output,opt.test_dataset), opt.upscale_factor, opt.noise)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

#print('===> Building model')
if opt.model_type == 'DBPNLL':
    model = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor, combination=opt.combination, tuning=True) ###D-DBPN
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) ###D-DBPN
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor, combination=opt.combination, tuning=True) ###D-DBPN
    
# if cuda:
#     model = torch.nn.DataParallel(model, device_ids=gpus_list)

model= load_checkpoint(model)
#print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = Variable(batch[0]), batch[1]
        if cuda:
            input = input.cuda(gpus_list[0])
        #print(name[0])
        t0 = time.time()
        # if opt.chop_forward:
        #     with torch.no_grad():
        #         prediction = chop_forward(input, model, opt.upscale_factor)
        # else:
        #     if opt.self_ensemble:
        #         with torch.no_grad():
        #             prediction = x8_forward(input, model)
        #     else:
        #         with torch.no_grad():
        #if name[0] not in alreadysaved:
        with torch.no_grad():
            prediction = model(input)
                
        # if opt.residual:
        #     prediction = prediction + bicubic

        t1 = time.time()

        print("===> Processed: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        #print(name[0])
        save_img_sr(prediction.cpu().data, name[0])
        save_img_lr(input.cpu().data, name[0])

def save_img_sr(img, img_name):
    save_img = img.squeeze().numpy().transpose(0,1)
    #save_img = img.squeeze().numpy()
    # save img
    folder = os.path.join(opt.test_dataset+'/sr', str(opt.noise).replace('.', ''))
    save_dir=os.path.join(opt.output,folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    return np.save(save_fn, save_img)

def save_img_lr(img, img_name):
    save_img = img.squeeze().numpy().transpose(0,1)
    #save_img = img.squeeze().numpy()
    # save img
    folder = os.path.join(opt.test_dataset+'/lr', str(opt.noise).replace('.', ''))
    save_dir=os.path.join(opt.output,folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    return np.save(save_fn, save_img)

##Eval Start!!!!

eval()

# replace 0.01 with a string 001
n = 0.01
n = str(n).replace('.', '')

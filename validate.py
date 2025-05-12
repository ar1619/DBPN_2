import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net as DBPN
from data import get_validation_set
from functools import reduce

import time

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--quantize', type=bool, default=True)
parser.add_argument('--decimals', type=int, default=2)
parser.add_argument('--combination', type=int, default=2)
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--hr_valid_dataset', type=str, default='../MOD_tensor_validate')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--test_dataset', type=str, default='XCO2')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--model', default='models/DBPNLL_x8.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def load_checkpoint(model):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

    return model, epoch

model = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=16, combination=opt.combination, tuning=True).cuda()
model, epoch = load_checkpoint(model)


criterion = nn.L1Loss(reduction='none').cuda()

validation_set = get_validation_set(opt.data_dir, opt.hr_valid_dataset, 16, opt.decimals, quantize=True)
validation_data_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

epoch_val_loss = 0
model.eval()
with torch.no_grad():
    for iteration, batch in enumerate(validation_data_loader, 1):
        input, target, mask= Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda()
            target = target.cuda()
            mask = mask.cuda()
        prediction = model(input)
        loss_val = criterion(prediction, target)
        loss_val = loss_val * mask
        loss_val = loss_val.sum() / mask.sum()
        
        epoch_val_loss += loss_val.item()
    val_loss = epoch_val_loss / len(validation_data_loader)
    print("===> Validation Loss for epoch {}: {:.6f}".format(epoch, val_loss))
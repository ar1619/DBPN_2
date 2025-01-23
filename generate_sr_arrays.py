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
from data import get_eval_set
from functools import reduce
import xarray as xr
from datetime import datetime, timedelta

import time
parser = argparse.ArgumentParser()
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

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def array_centered(file_name, lat, lon, window_size=16):
    # Open the file
    ds = xr.open_dataset(file_name)
    # Get the lat, lon, xco2 array
    lats = ds['lat'].values
    lons = ds['lon'].values
    xco2 = ds['XCO2'].values
    # Find the index of the nearest lat and lon
    lat_idx = find_nearest(lats, lat)
    lon_idx = find_nearest(lons, lon)
    # Get the window of data
    window = xco2[0, lat_idx-window_size:lat_idx+window_size, lon_idx-window_size:lon_idx+window_size]
    return window

def get_dates(start_date, end_date):
    # Get the start and end dates
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    # Get filename
    current_date = start_date
    filename = '../../OCO-2/oco2_GEOS_L3CO2_day_{}_B10206Ar.nc4'.format(current_date)
    file_list = [[filename, current_date]]
    # Get the list of names
    while current_date != end_date:
        current_date_timestep = datetime.strptime(current_date, '%Y%m%d')
        current_date = (current_date_timestep + timedelta(days=1)).strftime('%Y%m%d')
        filename = '../../OCO-2/oco2_GEOS_L3CO2_day_{}_B10206Ar.nc4'.format(current_date)
        file_list.append([filename, current_date])
    return file_list

def load_checkpoint(model):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model'])

    return model

def save_img(img, img_name):
    save_img = img.squeeze().numpy().transpose(0,1)
    #save_img = img.squeeze().numpy()
    # save img
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    return np.save(save_fn, save_img)


def main():
    lat = -53.000695
    lon = -72.420612
    window_size = 16
    factory = 'Invierno_Mine'
    filename_list = get_dates('20180101', '20211231')

    for item in filename_list:
        try:
            lr_array = np.load('Input/Mines/'+factory+'/'+item[1]+'.npy')
            continue
        except:
            name = item[0]
            date = item[1]
            lr_array = array_centered(name, lat, lon, window_size)
            np.save('Input/Mines/'+factory+'/'+date+'.npy', lr_array)

    print('LR arrays ready for SR')

    # Load the model
    #print('===> Loading datasets')
    test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset), os.path.join(opt.output,opt.test_dataset), opt.upscale_factor, opt.quantize)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    #print('===> Building model')
    if opt.model_type == 'DBPNLL':
        model = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor, combination=opt.combination, tuning=True) ###D-DBPN

    model= load_checkpoint(model)
    #print('Pre-trained SR model is loaded.')

    if cuda:
        model = model.cuda(gpus_list[0])

    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = Variable(batch[0]), batch[1]
        if cuda:
            input = input.cuda(gpus_list[0])
        t0 = time.time()
        with torch.no_grad():
            prediction = model(input)

        t1 = time.time()

        print("===> Processed: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])

if __name__ == '__main__':
    main()
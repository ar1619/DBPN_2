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
from data_pwp import get_eval_set
from functools import reduce

import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--pwp_data', type=str, default='../Paper/pwp_data/pwp_train_2016.npy', help='power plant data')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='XCO2')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--year', type=str, default='2016')
parser.add_argument('--model', default='models/DBPNLL_x8.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

train_test = False

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

#print('===> Building model')
model = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) ###D-DBPN
    
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('SR model is loaded.')

if train_test:
    pwp_coord = np.load(opt.pwp_data)
    #TO BE QUICKER
    pwp_coord = np.concatenate((pwp_coord[:150], pwp_coord[-150:]))
    file_save = opt.year+'_train_200.npy'
else:
    print('Creating test set')
    dataset = opt.pwp_data.replace('train', 'test')
    pwp_coord = np.load(dataset)
    #TO BE QUICKER
    pwp_coord = np.concatenate((pwp_coord[:50], pwp_coord[-50:]))
    file_save = opt.year+'_test_50.npy'


if cuda:
    model = model.cuda(gpus_list[0])

def eval(filename):
    # list_of_days = ['20160114_diff.npy','20161014_diff.npy','20160812_diff.npy','20160418_diff.npy']
    day = 0
    list_of_days = ['20160114_diff.npy', '20160117_diff.npy', '20160214_diff.npy', '20160111_diff.npy',
                    '20160313_diff.npy', '20160310_diff.npy', '20160414_diff.npy', '20160417_diff.npy',
                    '20160512_diff.npy', '20160515_diff.npy', '20160616_diff.npy', '20160619_diff.npy',
                    '20160714_diff.npy', '20160717_diff.npy', '20160811_diff.npy', '20160814_diff.npy',
                    '20160915_diff.npy', '20160918_diff.npy', '20161013_diff.npy', '20161016_diff.npy',
                    '20161113_diff.npy', '20161110_diff.npy', '20161215_diff.npy', '20161218_diff.npy']
    final_array = np.zeros((len(list_of_days), len(pwp_coord)))
    model.eval()
    for batch in testing_data_loader:
        t0 = time.time()
        if batch[1][0] in list_of_days:
            with torch.no_grad():
                input, name = Variable(torch.permute(batch[0], (1,0,4,2,3))), batch[1]
                if cuda:
                    input = input.cuda(gpus_list[0])
            #if name[0] not in alreadysaved:
            with torch.no_grad():
                # print(len(input))
                for i in range(len(input)):
                    #print(input[i].size())
                    prediction = model(input[i])
                    sr_array = prediction.cpu().data.squeeze().numpy().transpose(0,1)
                    sr_pixel_value = get_best_pixel(sr_array, boundaries[i], pwp_coord[i])
                    # print(sr_pixel_value, 'for pwp ', pwp_coord[i])
                    final_array[day, i] = sr_pixel_value
                day = day + 1
                # print(final_array)
                print("===> Processing: %s || Timer: %.4f sec." % (name, (time.time() - t0)))

        else:
            name = batch[1]

    np.save(os.path.join(opt.output,opt.test_dataset,filename), final_array)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# The boundary format is low_lat, left_lon, high_lat and right_lon
# The box is centered on the sensor and 20x20 degrees
def boundaries_box(coordinates, size):
    lat = np.linspace(-90, 90, 361)
    lon = np.linspace(-180, 179.375, 576)
    nb_sensors = len(coordinates)
    boundaries = np.ones((nb_sensors, 4)).astype(int)
    for i in range(nb_sensors):
        lat_sensor = float(coordinates[i][0])
        lon_sensor = float(coordinates[i][1])
        boundaries[i, 0] = find_nearest(lat, lat_sensor - size)
        boundaries[i, 1] = find_nearest(lon, lon_sensor - size)
        boundaries[i, 2] = find_nearest(lat, lat_sensor + size)
        boundaries[i, 3] = find_nearest(lon, lon_sensor + size)
    return boundaries

##SR best pixel
def get_best_pixel(array, boundaries, coords):
    h, w = array.shape
    bound_right = 180*boundaries[3]/576 - 180*(1-boundaries[3]/576)
    bound_left = 180*boundaries[1]/576 - 180*(1-boundaries[1]/576)
    bound_bottom = 90*boundaries[0]/361 - 90*(1-boundaries[0]/361)
    bound_top = 90*boundaries[2]/361 - 90*(1-boundaries[2]/361)
    lat_options = np.linspace(bound_bottom, bound_top, h)
    lon_options = np.linspace(bound_left, bound_right, w)
    # lat_options = np.linspace(boundaries[0], boundaries[2], h)
    # lon_options = np.linspace(boundaries[1], boundaries[3], w)
    best_lat = find_nearest(lat_options, float(coords[2]))
    best_lon = find_nearest(lon_options, float(coords[1]))
    #print(sensor_coord[i], lat_options[best_lat], lon_options[best_lon])
    return array[best_lat, best_lon]

##Eval Start!!!!
boundaries = boundaries_box(pwp_coord, 10)
#print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset), opt.upscale_factor, boundaries, opt.year)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

eval(file_save)

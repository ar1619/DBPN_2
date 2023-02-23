from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net as DBPNLL
from data import get_training_set, get_validation_set
import pdb
import socket
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--validBatchSize', type=int, default=1, help='validation batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--patience', type=int, default=4, help='patience value for early stopping')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--hr_train_dataset', type=str, default='../DIV2K_train_HR')
parser.add_argument('--hr_valid_dataset', type=str, default='../MOD_tensor_validate')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='1channel_16_MOD', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    epoch_val_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        model.train()
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
        
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(validation_data_loader, 1):
            input, target= Variable(batch[0]), Variable(batch[1])
            if cuda:
                input = input.cuda(gpus_list[0])
                target = target.cuda(gpus_list[0])

            t0 = time.time()
            prediction = model(input)

            loss_val = criterion(prediction, target)
            t1 = time.time()
            epoch_val_loss += loss_val.data
            
        val_loss = epoch_val_loss / len(validation_data_loader)

    print("===> Epoch {} Training: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Validation: Avg. Loss: {:.4f}".format(epoch, val_loss))
    return val_loss

def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+".pth"
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

valid_set = get_validation_set(opt.data_dir, opt.hr_valid_dataset, opt.upscale_factor)
validation_data_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.validBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor)
    
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.MSELoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
best_val_loss = 100000
i = 0
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    val_loss = train(epoch)
    if val_loss >= best_val_loss:
        i += 1
    else:
        i = 0
        best_val_loss = val_loss
        checkpoint(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        
    if i == opt.patience:
        print('Loss stopped improving at epoch N: {}'.format(epoch))
        break

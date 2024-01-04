from __future__ import print_function
import argparse
from math import log10

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
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
parser.add_argument('--patience', type=int, default=15, help='patience value for early stopping')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--quantize', type=bool, default=True)
parser.add_argument('--decimals', type=int, default=2)
parser.add_argument('--combination', type=int, default=1, help='combination of kernel, stride and padding')
parser.add_argument('--noise_level', type=float, default=0.01)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='../DIV2K_train_HR')
parser.add_argument('--hr_valid_dataset', type=str, default='../MOD_tensor_validate')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='1channel', help='Location to save checkpoint models')
#parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')

opt = parser.parse_args()
gpus_list = [i for i in range(opt.gpus)]
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
print(gpus_list)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_data(rank, world_size, num_workers=0):
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.noise_level, opt.patch_size, opt.data_augmentation, opt.decimals, opt.quantize)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=opt.batchSize, shuffle=False, sampler=sampler)

    validation_set = get_validation_set(opt.data_dir, opt.hr_valid_dataset, 16, opt.decimals, quantize=True)
    val_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    validation_data_loader = DataLoader(dataset=validation_set, num_workers=num_workers, batch_size=opt.validBatchSize, shuffle=False, sampler=val_sampler)
    return training_data_loader, validation_data_loader

def cleanup():
    dist.destroy_process_group()

def train(epoch, training_data_loader, criterion, optimizer, model):
    epoch_loss = 0
    training_data_loader.sampler.set_epoch(epoch)
    
    for iteration, batch in enumerate(training_data_loader, 1):
        model.train()
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)
        t_inter = time.time()

        if cuda:
            target = target.to(prediction.device)
        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f},{:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t_inter - t0),(t1 - t_inter)))

    cleanup()

    print("===> Epoch {} Training: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
#    print("===> Epoch {} Validation: Avg. Loss: {:.4f}".format(epoch, val_loss))
#    return val_loss

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

def checkpoint(epoch, model, optimizer):
    if torch.distributed.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        model_out_path = opt.save_folder + opt.model_type + "_post_tuning" + ".pth"
        torch.save(checkpoint, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

def load_checkpoint(model, optimizer, rank):
    loc = 'cuda:{}'.format(torch.distributed.get_rank())
    checkpoint = torch.load(opt.pretrained_sr, map_location=loc)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    return model, optimizer, epoch

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

def main(rank, world_size):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    if rank == 0:
        print('===> Loading datasets')
    training_data_loader, validation_data_loader = prepare_data(rank=rank, world_size=world_size, num_workers=0)

    if rank == 0:
        print('===> Building model ', opt.model_type)

    if opt.pretrained:
        model = DBPNLL(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor, residual=opt.residual)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, rank)

    model = DBPNLL(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=16, combination=opt.combination, tuning=True).to(rank)
        
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss().to(rank)

    if rank == 0:
        print('---------- Networks architecture -------------')
        print_network(model)
        print('----------------------------------------------')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    i = 0
    best_val_loss = None
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        if rank == 0:
            t0 = time.time()
        epoch_loss = 0
        training_data_loader.sampler.set_epoch(epoch)
        validation_data_loader.sampler.set_epoch(epoch)
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1])
            if cuda:
                input = input.cuda()

            optimizer.zero_grad()
            prediction = model(input)

            if cuda:
                target = target.to(prediction.device)
            loss = criterion(prediction, target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

        loss_tensor = torch.tensor([loss.item()], device=rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        epoch_val_loss = 0
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validation_data_loader, 1):
                input, target= Variable(batch[0]), Variable(batch[1])
                if cuda:
                    input = input.cuda()
                    target = target.cuda()
                prediction = model(input)
                loss_val = criterion(prediction, target)
                
                epoch_val_loss += loss_val.data
            val_loss_tensor = torch.tensor([loss_val.item()], device=rank)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        if (epoch+1) % (opt.nEpochs/2) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
                print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
                
        if rank == 0:
            average_loss = loss_tensor.item() / world_size
            val_loss = val_loss_tensor.item() / world_size
            t1 = time.time()
            print("===> Epoch {} Training: Avg. Train Loss: {:.4f}, Val Loss: {:.4f} || Timer: {:.4f}".format(epoch, average_loss, val_loss, t1 - t0))

            if best_val_loss is None or val_loss < best_val_loss:
                checkpoint(epoch, model, optimizer)
                best_val_loss = val_loss
            if i == opt.patience:
                print('Loss stopped improving at epoch N: {}'.format(epoch))
                break

    cleanup()

if __name__ == '__main__':
    world_size = opt.gpus
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
#valid_set = get_validation_set(opt.data_dir, opt.hr_valid_dataset, opt.upscale_factor)
#validation_data_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.validBatchSize, shuffle=False)

#best_val_loss = 100000
    #val_loss = train(epoch)
    #if val_loss >= best_val_loss:
    #    i += 1
    #else:
    #    i = 0
    #    checkpoint(epoch)
    #    best_val_loss = val_loss

    # learning rate is decayed by a factor of 10 every half of total epochs
        
#    model.eval()
#    with torch.no_grad():
#        for iteration, batch in enumerate(validation_data_loader, 1):
#            input, target= Variable(batch[0]), Variable(batch[1])
#            if cuda:
#                input = input.cuda(gpus_list[0])
#                target = target.cuda(gpus_list[0])

#            t0 = time.time()
#            prediction = model(input)

#            loss_val = criterion(prediction, target)
#            t1 = time.time()
#            epoch_val_loss += loss_val.data
            
#        val_loss = epoch_val_loss / len(validation_data_loader)
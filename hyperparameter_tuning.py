import argparse
from math import log10

import os
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
from data import get_training_set

import ray
from ray import train, tune
from ray.tune.search.hyperopt import HyperOptSearch

import pdb
import socket
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10, help='patience value for early stopping')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='../../RDS/DIV2K/DIV2K_train_HR_16')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='replicate_16_MOD', help='Location to save checkpoint models')

opt = parser.parse_args()
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_data(rank, world_size, num_workers=0):
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.noise_level, opt.patch_size, opt.data_augmentation, opt.decimals, opt.quantize)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=opt.batchSize, shuffle=False, sampler=sampler)
    return training_data_loader

def cleanup():
    dist.destroy_process_group()

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
        model_out_path = opt.save_folder + opt.model_type + "ITER" + str(epoch) + ".pth"
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

def train_ddp(config, num_epochs=10):
    # Set up the DDP environment.
    
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, 16, config['noise_level'], opt.patch_size, opt.data_augmentation, config['decimals'], config['quantize'])
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=config['batch_size'], shuffle=True)

    model = DBPNLL(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=16).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.L1Loss().cuda()
    
    # Define your training loop.
    for epoch in range(num_epochs):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            model.train()
            input, target = Variable(batch[0]), Variable(batch[1])
            if cuda:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            prediction = model(input)

            if cuda:
                target = target.to(prediction.device)
            loss = criterion(prediction, target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

            epoch_total_loss = epoch_loss / len(training_data_loader)
        tune.report(loss=epoch_total_loss)
    
    # Clean up DDP.
    dist.destroy_process_group()

def tune_ddp(num_samples=10, num_epochs=10, gpus_per_trial=1):
    # Define the hyperparameter search space.
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4]),
        "noise_level": tune.choice([0.05, 0.01, 0.005, 0.001]),
        "decimals": tune.choice([2, 3, 4, 5]),
        "quantize": tune.choice([True, False])
        # Add other hyperparameters here
    }
    
    # Define the scheduler and search algorithm.
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)
    search_alg = HyperOptSearch(metric="loss", mode="min")
    
    # Launch the Ray Tune run.
    analysis = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_ddp), resources={"cpu": 16, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=search_alg
        ),
        param_space=config)
    
    results = analysis.fit()
    best_hyperparameters = results.get_best_result().config
    print("Best hyperparameters found were: ", best_hyperparameters)
# Example usage:
# This should be executed in a script that is run with `ray submit` or similar.
num_samples = 10
num_epochs = 10
gpus_per_trial = 1
tune_ddp(num_samples, num_epochs, gpus_per_trial)
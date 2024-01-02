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
from data import get_training_set, get_validation_set

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
parser.add_argument('--hr_valid_dataset', type=str, default='../../RDS/DIV2K/DIV2K_valid_HR_16')
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
print(os.getcwd())

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

    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, 16, config['noise_level'], opt.patch_size, opt.data_augmentation, config['decimals'], quantize=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=4, shuffle=True)

    validation_set = get_validation_set(opt.data_dir, opt.hr_valid_dataset, 16, config['decimals'], quantize=True)
    validation_data_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    model = DBPNLL(num_channels=1, base_filter=64,  feat = 256, num_stages=10, scale_factor=16, combination=config['combination'], tuning=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.L1Loss().cuda()
    
    # Training loop.
    for epoch in range(num_epochs):
        t0 = time.time()
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
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t_iter - t_inter)))

        t_iter = time.time()
        epoch_total_loss = epoch_loss / len(training_data_loader)
        # print("===> Epoch {} Complete: Training Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, epoch_total_loss, (t_iter - t0)))
        # train.report({"train_loss": epoch_total_loss})

    # Validation loop.
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
                
                epoch_val_loss += loss_val.item()
            val_loss = epoch_val_loss / len(validation_data_loader)
                    
            train.report({"val_loss": val_loss})
        t1 = time.time()
        print("Epoch {} complete. Train Loss: {:.4f}. Val Loss: {:.4f} || Total time:{:.4f}".format(epoch, epoch_total_loss, val_loss, t1-t0))

def tune_ddp(num_samples, num_epochs, gpus_per_trial=1):
    # Hyperparameters search space.
    config = {
        "lr": tune.loguniform(1e-5, 5e-4),
        "noise_level": tune.choice([0.05, 0.01, 0.005]),
        "decimals": tune.choice([2, 3, 4, 5]),
        "combination": tune.choice([1, 2])
    }
    
    # Scheduler to stop bad trials.
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=3,
        reduction_factor=2)
    search_alg = HyperOptSearch(metric="val_loss", mode="min")
    
    # Tuner.
    analysis = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_ddp), resources={"cpu": 16, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=search_alg
        ),
        param_space=config)
    
    results = analysis.fit()
    best_results = results.get_best_result("val_loss", "min")
    print("Best result: ", best_results)
    print("Best hyperparameters found were: ", best_results.config)

num_samples = 10
num_epochs = 20
gpus_per_trial = 1
tune_ddp(num_samples, num_epochs, gpus_per_trial)
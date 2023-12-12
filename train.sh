#!/bin/bash -l

#SBATCH --job-name=sr_w_noise
#SBATCH -e tr_noise_clip-%j.err
#SBATCH -o tr_noise_clip-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python main.py --nEpochs 2000 --start_iter 1 --upscale_factor 16 --patch_size 32 --batchSize 4 --gpus 1 --hr_train_dataset ../MOD_tensor

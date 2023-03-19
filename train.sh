#!/bin/bash -l

#SBATCH --job-name=wd_0_00001_16
#SBATCH -e training-%j.err
#SBATCH -o training-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testgpu

srun python main.py --nEpochs 1000 --upscale_factor 16 --patch_size 32 --batchSize 4 --gpus 1 --prefix 1channel_wd_00001_16_MSE --hr_train_dataset ../MOD_tensor --patience 50 --pretrained_sr MOD_tensorese-hivemindDBPNLL1channel_wd_00001_16_MSE.pth --weight_decay 0.0001

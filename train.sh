#!/bin/bash -l

#SBATCH --job-name=train_post_tuning
#SBATCH -e post_tuning-%j.err
#SBATCH -o post_tuning-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python -u main.py --nEpochs 100 --start_iter 1 --upscale_factor 16 --patch_size 32 --lr 0.00025 --name replica --noise_level 0.01 --decimals 5 --batchSize 4 --combination 2 --gpus 2 --hr_train_dataset ../MOD_tensor --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val

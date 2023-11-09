#!/bin/bash -l

#SBATCH --job-name=pwp_difference_train
#SBATCH -e build_dataset-%j.err
#SBATCH -o build_dataset-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python pwp_difference.py --upscale_factor 16 --input_dir ../../RDS/OCO-2 --output ../../RDS/OCO-2/datasets_diff --test_dataset OCO-2_diff --year 2016 --model weights/MOD_tensorese-hivemindDBPNLL1channel_16_MSE.pth

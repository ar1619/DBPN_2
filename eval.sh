#!/bin/bash -l

#SBATCH --job-name=sr_16
#SBATCH --partition=hive_flash
#SBATCH --qos=hive_flash_low

#SBATCH -e sr_w_noise-%j.err
#SBATCH -o sr_w_noise-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

nvidia-smi
#nvcc --version
source /scratch_hive/ar1619/miniconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python eval.py --upscale_factor 16 --combination 2 --input_dir Input --output Result --test_dataset sentinel --model weights/new_loss_post_tuning.pth

#!/bin/bash -l

#SBATCH --job-name=uncer_lr
#SBATCH --partition=hive_flash
#SBATCH --qos=hive_flash_low

#SBATCH -e uncer_lr-%j.err
#SBATCH -o uncer_lr-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0

nvidia-smi
#nvcc --version
source /scratch_hive/ar1619/miniconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python uncertainty_model.py

#!/bin/bash -l

#SBATCH --job-name=data_dist
#SBATCH --partition=hive_short
#SBATCH --qos=hive_short_high

#SBATCH -e data_dist-%j.err
#SBATCH -o data_dist-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0

nvidia-smi
#nvcc --version
source /scratch_hive/ar1619/miniconda3/etc/profile.d/conda.sh
conda activate maps

srun python data_distribution.py

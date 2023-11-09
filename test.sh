#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH -e test-%j.err
#SBATCH -o test-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:00:10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python test.py 
#!/bin/bash -l

#SBATCH --job-name=debug
#SBATCH -e debug-%j.err
#SBATCH -o debug-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testgpu

srun python debug.py 

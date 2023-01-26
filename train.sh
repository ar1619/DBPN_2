#!/bin/bash -l

#SBATCH --job-name=model_training
#SBATCH -e training-%j.err
#SBATCH -o training-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testgpu

srun python main.py --nEpochs 1000 --upscale_factor 8

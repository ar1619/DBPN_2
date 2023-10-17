#!/bin/bash -l

#SBATCH --job-name=2_inputs_1
#SBATCH -e training-%j.err
#SBATCH -o training-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_env

srun python main.py --nEpochs 5 --start_iter 1 --upscale_factor 16 --patch_size 32 --batchSize 4 --gpus 1 --prefix 2_inputs --hr_train_dataset ../../RDS/DIV2K_train_HR

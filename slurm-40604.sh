#!/bin/bash -l

#SBATCH --job-name=hp_tuning
#SBATCH -e hp_tun-%j.err
#SBATCH -o hp_tun-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python -u hyperparameter_tuning.py --start_iter 1 --patch_size 32 --gpus 2 --hr_train_dataset /home/ar1619/DBPN_2/MOD_tensor --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_validate

#!/bin/bash -l

#SBATCH --job-name=validation
#SBATCH -e val-%j.err
#SBATCH -o val-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_13_1.pth
srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_14_1.pth
srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_15_1.pth
srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_16_1.pth
srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_17_1.pth
srun python -u validate.py --upscale_factor 16 --combination 2 --decimals 5 --hr_valid_dataset /home/ar1619/DBPN_2/MOD_tensor_val --model weights/DBPNLL_post_tuning_18_1.pth

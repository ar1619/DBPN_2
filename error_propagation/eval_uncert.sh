#!/bin/bash -l

#SBATCH --job-name=eval_uncert
#SBATCH --partition=hive_flash
#SBATCH --qos=hive_flash_low

#SBATCH -e eval_uncert-%j.err
#SBATCH -o eval_uncert-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

nvidia-smi
#nvcc --version
source /scratch_hive/ar1619/miniconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python eval_uncert.py --upscale_factor 16 --combination 2 --noise 0.1 --input_dir Input --output ../RDS/OCO-2 --test_dataset uncertainty/sites/ --model weights/new_loss_post_tuning.pth

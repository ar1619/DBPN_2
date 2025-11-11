#!/bin/bash -l

#SBATCH --job-name=sr_16
#SBATCH --partition=hive_inter
#SBATCH --qos=hive_inter_high

#SBATCH -e xco2_dataset-%j.err
#SBATCH -o xco2_dataset-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

nvidia-smi
#nvcc --version
source /scratch_hive/ar1619/miniconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python create_dataset.py --upscale_factor 16 --combination 2 --input_dir ../RDS/OCO-2/low_res --output ../RDS/ephemeral/ --test_dataset OCO2 --model weights/new_loss_post_tuning.pth

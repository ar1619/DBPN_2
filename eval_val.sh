#!/bin/bash -l

#SBATCH --job-name=val
#SBATCH -e sr_val-%j.err
#SBATCH -o sr_val-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python eval.py --upscale_factor 16 --input_dir ../../RDS/difference_xco2_val/lr --output ../../RDS/difference_xco2_val/sr --test_dataset diff --model weights/MOD_tensorese-hivemind1channel_50.pth

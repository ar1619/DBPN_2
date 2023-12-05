#!/bin/bash -l

#SBATCH --job-name=cent_arr_sr
#SBATCH -e building-%j.err
#SBATCH -o building-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

srun python eval.py --upscale_factor 16 --input_dir Input --output Result --test_dataset XCO2 --model weights/MOD_tensorese-hivemindDBPNLL1channel_16_MSE.pth

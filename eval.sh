#!/bin/bash -l

#SBATCH --job-name=cent_arr_sr
#SBATCH -e sr_w_noise-%j.err
#SBATCH -o sr_w_noise-%j.out
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

srun python eval.py --upscale_factor 16 --combination 2 --input_dir Input --output Result --test_dataset XCO2 --model ../../RDS/dbpn/DBPN_2/weights/new_loss_post_tuning.pth

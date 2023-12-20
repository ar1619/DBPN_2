#!/bin/bash -l

#SBATCH --job-name=hp_tuning
#SBATCH -e hp_tun-%j.err
#SBATCH -o hp_tun-%j.out
#SBATCH --mem-per-cpu=1500
#SBATCH --time=4-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2

#nvidia-smi
#nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_current2

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

head_node=${nodes_array[0]} # Getting the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address) # Getting the head node IP

if [[$head_node_ip == *" "*]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus="${SLURM_CPUS_PER_TASK}" --block &
sleep 5

worker_num = $((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  echo "Starting WORKER $i at ${nodes_array[$i]}"
  srun --nodes=1 --ntasks=1 -w "${nodes_array[$i]}" ray start --address="$ip_head" --num-cpus="${SLURM_CPUS_PER_TASK}" --block &
  sleep 5
done

python -u hyperparameter_tuning.py --start_iter 1 --patch_size 32 --gpus 2 --hr_train_dataset ../MOD_tensor "$SLURM_CPUS_PER_TASK"

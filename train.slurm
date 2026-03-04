#!/bin/bash
#SBATCH --job-name=longformer
#SBATCH --output=/ceph/project/SW10-CausalLM/LongFormer/logs/slurm_%x_%j.out
#SBATCH --error=/ceph/project/SW10-CausalLM/LongFormer/logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:4

set -e

PROJECT_DIR=/ceph/project/SW10-CausalLM/LongFormer
cd $PROJECT_DIR

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

# Force exactly the allocated GPUs to be visible to avoid NCCL deadlock oversubscription
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1","$2","$3","$4}')

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

uv pip install -e .
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Ensure master_port is randomized slightly to prevent port collision on shared nodes
MASTER_PORT=$((10000 + $RANDOM % 20000))

uv run torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log
#!/bin/bash
#SBATCH --job-name=longformer
#SBATCH --output=/ceph/project/SW10-CausalLM/LongFormer/logs/slurm_%x_%j.out
#SBATCH --error=/ceph/project/SW10-CausalLM/LongFormer/logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l4:1

set -e

PROJECT_DIR=/ceph/project/SW10-CausalLM/LongFormer
cd $PROJECT_DIR

mkdir -p logs

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

# Usage of specific GPU selection removed to rely on Slurm or env vars
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a logs/train_live_${SLURM_JOB_ID}.log
nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

# Ensure uv is in the PATH
export PATH="$HOME/.local/bin:$PATH"

uv run python -u src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log
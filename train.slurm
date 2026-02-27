#!/bin/bash
#SBATCH --job-name=longformer
#SBATCH --output=/ceph/project/SW10-CausalLM/LongFormer_NewArchitecture/logs/slurm_%x_%j.out
#SBATCH --error=/ceph/project/SW10-CausalLM/LongFormer_NewArchitecture/logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32   # Increased to feed data to 4 GPUs faster
#SBATCH --mem=64G            # Increased memory to support 4 processes
#SBATCH --gres=gpu:l4:4      # <--- Requesting 4 GPUs!

set -e

PROJECT_DIR=/ceph/project/SW10-CausalLM/LongFormer_NewArchitecture
cd $PROJECT_DIR

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log
nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

# Use torchrun to launch distributed training on 4 GPUs
uv run torchrun --nproc_per_node=4 src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log
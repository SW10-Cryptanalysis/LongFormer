#!/bin/bash
#SBATCH --job-name=longformer
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:4

# NOTE: set -e intentionally removed from the top level.
# It caused sbatch to abort the entire script if job submission
# failed (e.g. cluster job-limit reached), silently killing the chain.

# ==============================================================================
# DYNAMIC AUTO-CHAINING LOGIC
# ==============================================================================
if [ -d "outputs/final_model" ]; then
    echo "Success: 'outputs/final_model' found. Training is completely finished."
    echo "Breaking the chain. No further jobs will be submitted."
    exit 0
fi

echo "Dynamically queuing the next 12-hour continuation job with arguments: $@"
NEXT_JOB_ID=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID $0 "$@") || {
    echo "WARNING: Could not submit next chained job (cluster limit reached?)." \
         "Training will still run this slot, but auto-chaining has stopped." \
         "Re-submit manually after this job finishes if training is incomplete."
}
[ -n "$NEXT_JOB_ID" ] && echo "Next job queued with ID: $NEXT_JOB_ID"
# ==============================================================================

set -e  # Enable strict mode AFTER the sbatch call

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1","$2","$3","$4}')

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

uv pip install -e .
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

MASTER_PORT=$((10000 + $RANDOM % 20000))

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    -m src.train "$@" 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log
# Custom Causal LM
Custom causal language model with RoPE, FlashAttention, and optional Liger kernels for solving long homophonic ciphers.

## Usage
### Initialization
- `uv sync`
### Training
- `sbatch train.slurm`

#### Train without spaces
- `sbatch train.slurm --without-spaces`

### Monitoring during training
- `tail -f logs/train_live_<JOB_ID>.log`

### Cancellation of training
- `scancel --name=mistral_cipher`
- `scancel -u USERNAME`

## Configuration

All parameters are listed in `src/config.py`.

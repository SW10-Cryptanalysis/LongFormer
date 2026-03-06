from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers based on provided metadata
TEXT_LEN = 10000 
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 10

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Expecting pre-tokenized Arrow directories from preprocess.py
TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VAL_DIR = DATA_DIR / "tokenized_normal" / "Validation"

@dataclass
class Config:
    # ARCHITECTURE
    unique_homophones: int = 2067 
    unique_letters: int = 26
    vocab_size: int = 2176  # 2067 + 26 + buffer, padded to multiple of 64
    max_context: int = TOTAL_SEQ + BUFFER  
    dims: int = 512
    layers: int = 16
    att_heads: int = 8 
    
    # Custom Arch
    window_size: int = 512
    rope_theta: float = 1000000.0 # Increased for ~20k sequence lengths
    use_liger: bool = True
    packing: bool = True
    torch_compile: bool = False 
    bf16: bool = True
    hidden_act: str = "silu"
    
    @property
    def hidden_size(self):
        return self.dims
        
    @property
    def intermediate_size(self):
        return self.dims * 4

    # TRAINING
    batch_size: int = 2 
    grad_accum: int = 16 
    learning_rate: float = 2e-4
    epochs: int = 3
    grad_checkpoint: bool = True
    log_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 1000

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VAL_DIR

cfg = Config()
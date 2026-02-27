from dataclasses import dataclass
from pathlib import Path

# Constants matching roughly 16k context for recurrence distances
# text_len = 8192, total_seq = 16384. 
# We need vocab > 16384 to cover all possible recurrence distances.
TEXT_LEN = 8_192
TOTAL_SEQ = TEXT_LEN * 2

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
TRAINING_DIR = DATA_DIR / "Training_Arrow"
TEST_DIR = DATA_DIR / "Test_Arrow"
EVAL_DIR = DATA_DIR / "Test"
VALIDATION_DIR = DATA_DIR / "Validation"
TOKENIZED_DATA_DIR = DATA_DIR / "Training_Tokenized"
TOKENIZED_TEST_DIR = DATA_DIR / "Test_Tokenized"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

@dataclass
class Config:
    # ARCHITECTURE
    # Vocab size needs to be large enough for max recurrence distance (which is seq_len)
    # plus 30 output chars + special tokens.
    unique_homophones: int = 8192
    max_context: int = TOTAL_SEQ
    vocab_size: int = 16500 
    dims: int = 512
    layers: int = 16
    att_heads: int = 8 
    
    # Custom Arch
    window_size: int = 512
    rope_theta: float = 10000.0
    use_liger: bool = True
    packing: bool = True
    torch_compile: bool = True
    bf16: bool = True
    
    @property
    def hidden_size(self):
        return self.dims
        
    @property
    def intermediate_size(self):
        # SwiGLU usually expands the intermediate dim by 4x or 8/3x
        return self.dims * 4

    # TRAINING
    batch_size: int = 1 # Per device (will be packed)
    grad_accum: int = 8
    learning_rate: float = 2e-4
    epochs: int = 1
    grad_checkpoint: bool = True  # Enable by default for 16k
    log_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 5000

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = TRAINING_DIR
    test_dir: Path = TEST_DIR
    eval_dir: Path = EVAL_DIR
    tokenized_data_dir: Path = TOKENIZED_DATA_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR

# Instantiate to use across other files
cfg = Config()

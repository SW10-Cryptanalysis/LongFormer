from dataclasses import dataclass
from pathlib import Path

TEXT_LEN = 10_000
TOTAL_SEQ = TEXT_LEN * 2

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
TRAINING_DIR = DATA_DIR / "Training_Arrow"
TEST_DIR = DATA_DIR / "Test_Arrow"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
TOKENIZED_DATA_DIR = OUTPUT_DIR / "Training_Tokenized"
TOKENIZED_TEST_DIR = OUTPUT_DIR / "Test_Tokenized"

@dataclass
class Config:
    # ARCHITECTURE
    unique_homophones: int = 2500
    max_context: int = TOTAL_SEQ  
    vocab_size: int = 2560 
    dims: int = 512
    layers: int = 16
    att_heads: int = 8 
    
    # Custom Arch
    window_size: int = 512
    rope_theta: float = 10000.0
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
    data_dir: Path = TRAINING_DIR
    test_dir: Path = TEST_DIR
    tokenized_data_dir: Path = TOKENIZED_DATA_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR

cfg = Config()
from dataclasses import dataclass

TEXT_LEN = 8_192
UNIQUE_HOMOPHONE_COUNT = 8192
UNIQUE_LETTER_COUNT = 30
TOTAL_SEQ = TEXT_LEN * 2
OUTPUT_DIR = "./outputs"

@dataclass
class Config:
    # ARCHITECTURE
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    vocab_size: int = unique_homophones + unique_letters + 5
    max_context: int = TOTAL_SEQ
    dims: int = 384
    layers: int = 16
    att_heads: int = 6

    # TRAINING
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 3e-4
    epochs: int = 1
    grad_checkpoint: bool = True
    log_steps: int = 10
    save_steps: int = 500 # Fixed missing type hint

    # SYSTEM
    output_dir: str = OUTPUT_DIR # Fixed missing type hint

# Instantiate to use across other files
cfg = Config()
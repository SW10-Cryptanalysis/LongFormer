 
from dataclasses import dataclass

TEXT_LEN = 8_192
UNIQUE_HOMOPHONE_COUNT = 500
UNIQUE_LETTER_COUNT = 26
TOTAL_SEQ = TEXT_LEN * 2
OUTPUT_DIR = "./outputs"


@dataclass
class Config:
    # ARCHITECTURE

    # Vocab needs to be larger than unique homophone count + unique letter count + 1 (start/end/padding)
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    vocab_size: int = unique_homophones + unique_letters + 1
    # Input is ciphertext + plaintext
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
    save_steps = 500

    # SYSTEM
    output_dir = OUTPUT_DIR
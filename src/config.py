from dataclasses import dataclass
from pathlib import Path

TEXT_LEN = 8_192
UNIQUE_HOMOPHONE_COUNT = 8192
UNIQUE_LETTER_COUNT = 30
TOTAL_SEQ = TEXT_LEN * 2
DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
TRAINING_DIR = DATA_DIR / "Training" / "Training_Arrow"
TEST_DIR = DATA_DIR / "Test" / "Test_Arrow"
VALIDATION_DIR = DATA_DIR / "Validation"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
TOKENIZED_DATA_DIR = OUTPUT_DIR / "Training_Tokenized"
TOKENIZED_TEST_DIR = OUTPUT_DIR / "Test_Tokenized"

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
    grad_accum: int = 4
    learning_rate: float = 3e-4
    epochs: int = 1
    grad_checkpoint: bool = False
    log_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = TRAINING_DIR
    test_dir: Path = TEST_DIR
    tokenized_data_dir: Path = TOKENIZED_DATA_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR

# Instantiate to use across other files
cfg = Config()
from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers based on provided metadata
TEXT_LEN = 9961
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 78

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Expecting pre-tokenized Arrow directories from preprocess.py
TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VAL_DIR = DATA_DIR / "tokenized_normal" / "Validation"


@dataclass
class Config:
    """Configuration dataclass for model architecture, training, and system paths."""

    # ARCHITECTURE
    unique_homophones: int = 2494
    unique_letters: int = 26
    vocab_size: int = 2560  # Padded to multiple of 64
    max_context: int = TOTAL_SEQ + BUFFER

    # Custom Arch
    dims: int = 512
    layers: int = 16
    att_heads: int = 8
    window_size: int = 512
    rope_theta: float = 1_000_000.0
    use_liger: bool = True
    packing: bool = True
    bf16: bool = True
    hidden_act: str = "silu"

    @property
    def hidden_size(self) -> int:
        """Return the hidden size, equal to dims."""
        return self.dims

    @property
    def intermediate_size(self) -> int:
        """Return the intermediate MLP size, equal to dims * 4."""
        return self.dims * 4

    # TRAINING
    batch_size: int = 2
    grad_accum: int = 16
    learning_rate: float = 2e-4
    epochs: int = 3
    grad_checkpoint: bool = True
    log_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 1000
    save_total_limit: int = 2

    # Token IDs
    pad_token_id: int = 0
    sep_token_id: int = 2495
    space_token_id: int = 2496
    bos_token_id: int = 2497
    eos_token_id: int = 2498
    char_offset: int = 2499

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VAL_DIR


cfg = Config()

import argparse
import logging
import json
import os
from dataclasses import dataclass
from pathlib import Path
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--without-spaces",
    action="store_true",
    default=False,
    help="If enabled the model trains without space tokens in the training dataset",
)
cli_args, _ = parser.parse_known_args()

MAX_PLAIN_SPACES = 13077
MAX_PLAIN_NORMAL = 10063

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
HOMOPHONE_FILE = "metadata.json"

# Expecting pre-tokenized Arrow directories from preprocess.py
TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VAL_DIR = DATA_DIR / "tokenized_normal" / "Validation"

TOKENIZED_SPACED_TRAINING_DIR = DATA_DIR / "tokenized_spaced" / "Training"
TOKENIZED_SPACED_VALIDATION_DIR = DATA_DIR / "tokenized_spaced" / "Validation"
TOKENIZED_SPACED_TEST_DIR = DATA_DIR / "tokenized_spaced" / "Test"


@dataclass
class Config:
    """Configuration dataclass for model architecture, training, and system paths."""

    # ARCHITECTURE
    buffer: int = 10
    unique_letters: int = 26
    unique_homophones: int = 0
    vocab_size: int = 0

    @property
    def max_len(self) -> int:
        """Calculate dynamic variables after the dataclass is initialized."""
        if self.use_spaces:
            return (MAX_PLAIN_SPACES * 2) + self.buffer
        return (MAX_PLAIN_NORMAL * 2) + self.buffer

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
    logging_steps: int = 10
    save_total_limit: int = 2
    use_spaces: bool = not cli_args.without_spaces
    torch_compile: bool = False

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_val_dir: Path = TOKENIZED_VAL_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_spaced_train_dir: Path = TOKENIZED_SPACED_TRAINING_DIR
    tokenized_spaced_val_dir: Path = TOKENIZED_SPACED_VALIDATION_DIR
    tokenized_spaced_test_dir: Path = TOKENIZED_SPACED_TEST_DIR

    # Token IDs
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of sequence token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Offset for character token IDs."""
        return self.eos_token_id + 1

    def load_homophones(self) -> None:
        """Load homophone mappings from the metadata file."""
        homophone_path = os.path.join(DATA_DIR, HOMOPHONE_FILE)
        if not os.path.exists(homophone_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {homophone_path}. "
                "Cannot determine unique_homophones — aborting.",
                1,
            )
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except OSError as e:
            raise OSError(f"Could not read file: {homophone_path}") from e
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid or missing 'max_symbol_id' in {homophone_path}",
            ) from e

        raw = self.unique_homophones + self.unique_letters + self.buffer
        self.vocab_size = (
            (raw + 63) // 64 * 64
        )  # Padded to nearest multiple of 64 for L4 Ada Lovelace Tensor Cores
        logger.info(
            f"Config initialized: unique_homophones={self.unique_homophones}, sep_token_id={self.sep_token_id}, space_token_id={self.space_token_id}, bos_token_id={self.bos_token_id}, eos_token_id={self.eos_token_id}, char_offset={self.char_offset}, vocab_size={self.vocab_size}",
        )
        logger.info(
            f"Max len set to {self.max_len} based on use_spaces={self.use_spaces}"
        )


cfg = Config()
cfg.load_homophones()

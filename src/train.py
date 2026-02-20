import json
import glob
import os
import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.utils.data import Dataset
from config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CipherPlainData(Dataset):
    def __init__(self, directory_path):
        self.file_paths = glob.glob(os.path.join(directory_path, "*.json"))

        if not self.file_paths:
            raise ValueError(f"No .json files found in {os.path.abspath(directory_path)}. Did you upload the data?")
        
        # 1. Dynamically find the max homophone ID if not in Config
        # Or just use Config.unique_homophones if you're sure it's the max
        self.max_homophone = Config.unique_homophones 
        
        # 2. Define special tokens relative to the max homophone
        self.sep_token = self.max_homophone + 1
        self.char_offset = self.sep_token + 1
        
        # 3. Create a stable character mapping (No more ASCII 97)
        # In a real project, you'd load this from a JSON file
        self.chars = "abcdefghijklmnopqrstuvwxyz " # Add whatever chars you expect
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)

        # Ciphertext: [1, 18, 5] -> No change needed since it starts at 1
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        
        # Plaintext: Map char to its index, then add the offset
        # Use .get(c, 0) or similar to handle unknown characters safely
        plain_ids = [self.char_to_id[c] + self.char_offset for c in data["plaintext"]]

        # Combine
        full_seq = cipher_ids + [self.sep_token] + plain_ids
        full_seq = full_seq[:Config.max_context]
        
        # Labels: Mask everything before the plaintext
        labels = ([-100] * (len(cipher_ids) + 1)) + plain_ids
        labels = labels[:Config.max_context]

        # Padding (0 is safe because homophones start at 1)
        padding_length = Config.max_context - len(full_seq)
        input_ids = full_seq + [0] * padding_length
        labels = labels + [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum,
        learning_rate=Config.learning_rate,
        gradient_checkpointing=Config.grad_checkpoint,
        logging_steps=Config.log_steps,
        save_steps=Config.save_steps,
        # OOM without below
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(directory_path="./data/train"),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        trainer.train()

    trainer.save_model(f"{Config.output_dir}/model")


if __name__ == "__main__":
    train()
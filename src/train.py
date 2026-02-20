import json
import glob
import os
import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from config import cfg

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CipherPlainData(Dataset):
    def __init__(self, directory_path):
        self.file_paths = glob.glob(os.path.join(directory_path, "*.json"))
        if not self.file_paths:
            raise ValueError(f"No .json files found in {os.path.abspath(directory_path)}.")
        
        self.max_homophone = cfg.unique_homophones 
        self.sep_token = self.max_homophone + 1
        self.char_offset = self.sep_token + 1
        
        self.chars = "abcdefghijklmnopqrstuvwxyz " 
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)

        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        plain_ids = [self.char_to_id.get(c, 0) + self.char_offset for c in data["plaintext"]]

        # Safety measure to ensure we don't truncate away all the plaintext
        max_cipher_len = cfg.max_context - len(plain_ids) - 1 
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        full_seq = cipher_ids + [self.sep_token] + plain_ids
        
        labels = full_seq.copy()
        padding_length = cfg.max_context - len(full_seq)

        input_ids = full_seq + [0] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = [1] * len(full_seq) + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        gradient_checkpointing=cfg.grad_checkpoint,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(directory_path="./data/train"),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    trainer.train()

    trainer.save_model(f"{cfg.output_dir}/model")

if __name__ == "__main__":
    train()
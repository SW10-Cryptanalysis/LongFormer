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

        # 1. Safely truncate plain_ids first if it's absurdly long
        max_plain_len = cfg.max_context - 2
        if len(plain_ids) > max_plain_len:
            plain_ids = plain_ids[:max_plain_len]

        # 2. Now max_cipher_len is guaranteed to be >= 1
        max_cipher_len = cfg.max_context - len(plain_ids) - 1 
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        full_seq = cipher_ids + [self.sep_token] + plain_ids
        
        # 3. Final safety net
        full_seq = full_seq[:cfg.max_context]
        
        labels = full_seq.copy()
        padding_length = cfg.max_context - len(full_seq)

        input_ids = full_seq + [0] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = [1] * len(full_seq) + [0] * padding_length
        
        assert max(input_ids) < cfg.vocab_size, f"Found token ID {max(input_ids)} but vocab size is {cfg.vocab_size}"

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
        train_dataset=CipherPlainData(directory_path=cfg.data_dir),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    # Check if there is a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by number to get the latest (e.g. checkpoint-500, checkpoint-1000)
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            last_checkpoint = os.path.join(cfg.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.output_dir}/model")

if __name__ == "__main__":
    train()
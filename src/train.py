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
        # Ensure directory_path is a string (handles pathlib.Path from config)
        self.file_paths = glob.glob(os.path.join(str(directory_path), "*.json"))
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
        
        input_ids = full_seq
        labels = full_seq.copy()
        attention_mask = [1] * len(full_seq)
        
        assert max(input_ids) < cfg.vocab_size, f"Found token ID {max(input_ids)} but vocab size is {cfg.vocab_size}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        gradient_checkpointing=cfg.grad_checkpoint,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        bf16=True,
        ddp_find_unused_parameters=True,
        dataloader_num_workers=8,
    )
    
    def custom_collator(features):
        # 1. Find the actual max length in this specific batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # 2. Longformer requires sequence lengths to be a multiple of its attention window (512)
        # So we bump max_len up to the nearest multiple of 512
        remainder = max_len % 512
        if remainder != 0:
            max_len += (512 - remainder)
        
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            # Critical: pad labels with -100 so the model doesn't try to predict padding
            batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            
        return {k: torch.stack(v) for k, v in batch.items()}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(directory_path=str(cfg.data_dir)),
        eval_dataset=CipherPlainData(directory_path=str(cfg.test_dir)),
        data_collator=custom_collator
    )

    print(f"Training on {torch.cuda.device_count()} GPUs...")

    # Check if there is a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            last_checkpoint = os.path.join(cfg.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(cfg.output_dir) + "/model")

if __name__ == "__main__":
    train()
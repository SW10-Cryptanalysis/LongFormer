import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import Dataset
from model import get_model
from transformers import Trainer, TrainingArguments
from config import cfg

# Force expanded segments for fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

## train.py (Partial Update)
class ArrowDatasetWrapper(Dataset):
    def __init__(self, directory_path):
        # Load the Hugging Face Arrow dataset natively
        self.hf_dataset = load_from_disk(str(directory_path))
        if len(self.hf_dataset) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")
            
        # Setup tokenization (matching evaluate.py)
        self.max_homophone = cfg.unique_homophones 
        self.sep_token = self.max_homophone + 1
        char_offset = self.sep_token + 1
        chars = "abcdefghijklmnopqrstuvwxyz "
        
        # Create character to ID mapping for the plaintext
        self.char_to_id = {char: i + char_offset for i, char in enumerate(chars)}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Access the row directly from the Arrow dataset
        data = self.hf_dataset[idx]
        
        # 1. Parse Ciphertext integers from the string
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        
        # 2. Parse Plaintext characters into Token IDs
        plain_text = data.get("plaintext", "")
        # Use 0 (or another designated token) as a fallback for unknown characters
        plain_ids = [self.char_to_id.get(char, 0) for char in plain_text] 
        
        # 3. Concatenate: [Cipher] + [SEP] + [Plaintext]
        input_ids = cipher_ids + [self.sep_token] + plain_ids
        
        # Ensure we don't exceed max context
        if len(input_ids) > cfg.max_context:
            input_ids = input_ids[:cfg.max_context]
            
        # For CausalLM optimizing the joint distribution, labels are same as input
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

def pad_collate(batch):
    """
    Pads sequences to the longest sequence in the current batch.
    Prevents cross-document contamination and uses -100 to mask padding from the loss.
    """
    # Find the max length in this specific batch (dynamic padding saves memory)
    max_len = max(len(item["input_ids"]) for item in batch)
    
    padded_inputs = []
    padded_labels = []
    
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        
        # Pad input_ids with 0 (or any safe vocab token)
        inputs = F.pad(item["input_ids"], (0, pad_len), value=0)
        
        # Pad labels with -100 so the CrossEntropyLoss completely ignores them
        labels = F.pad(item["labels"], (0, pad_len), value=-100)
        
        padded_inputs.append(inputs)
        padded_labels.append(labels)
        
    return {
        "input_ids": torch.stack(padded_inputs), # Shape: [batch_size, max_len]
        "labels": torch.stack(padded_labels)     # Shape: [batch_size, max_len]
    }

def train():
    model = get_model()
    
    train_ds = ArrowDatasetWrapper(cfg.data_dir)
    test_ds = ArrowDatasetWrapper(cfg.test_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, # 1, due to packing high seq len
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        
        # Checkpointing
        gradient_checkpointing=cfg.grad_checkpoint,
        
        # FP16/BF16
        bf16=cfg.bf16,
        
        # Logging
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_steps=cfg.eval_steps,
        
        # Speed
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=4,
        
        # DDP/FSDP
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=pad_collate
    )

    print(f"Starting training on {torch.cuda.device_count()} GPUs...")
    trainer.train()
    
    # Save manually since Trainer wrapper might fail on bare nn.Module without save_pretrained
    model_state = model.state_dict()
    # Save standard torch bin/safetensors
    output_path = str(cfg.output_dir) + "/final_model"
    os.makedirs(output_path, exist_ok=True)
    torch.save(model_state, output_path + "/pytorch_model.bin")

if __name__ == "__main__":
    train()

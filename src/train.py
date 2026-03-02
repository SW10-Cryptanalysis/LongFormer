import os
import torch
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

def packing_collate(batch):
    """
    Concatenates all sequences into one long buffer.
    In a real implementation we would preserve 'cu_seqlens' and pass it to the model 
    to reset attention masks. Our 'BlockSlidingWindowAttention' currently assumes 
    continuous relative positions or standard padding. 
    
    For now, we simply pad to the longest in the batch to be safe with the custom attention
    Wait: The Plan calls for Packing.
    """
    # 1. Concatenate inputs
    concat_input = torch.cat([item["input_ids"] for item in batch])
    concat_label = torch.cat([item["labels"] for item in batch])
    
    # Trim to nearest multiple of window size to avoid shape errors
    r = len(concat_input) % cfg.window_size
    if r != 0:
        concat_input = concat_input[:-r]
        concat_label = concat_label[:-r]
        
    # Check if we are too small (rare)
    if len(concat_input) == 0:
        return batch[0] # Fallback
        
    return {
        "input_ids": concat_input.unsqueeze(0), # Add batch dim [1, Seq]
        "labels": concat_label.unsqueeze(0)
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
        data_collator=packing_collate
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

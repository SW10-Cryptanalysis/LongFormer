import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from config import cfg
from model import get_model

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class ArrowDatasetWrapper(Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")
            
        self.max_homophone = cfg.unique_homophones 
        self.sep_token = self.max_homophone + 1
        self.unk_token = self.max_homophone + 2
        
        char_offset = self.unk_token + 1
        chars = "abcdefghijklmnopqrstuvwxyz "
        
        self.char_to_id = {char: i + char_offset for i, char in enumerate(chars)}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        data = self.hf_dataset[idx]
        
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        plain_text = data.get("plaintext", "")
        
        plain_ids = [self.char_to_id.get(char, self.unk_token) for char in plain_text] 
        input_ids = cipher_ids + [self.sep_token] + plain_ids
            
        if len(input_ids) > cfg.max_context:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"Safety constraint: Clipping sequence from {len(input_ids)} to {cfg.max_context}")
            input_ids = input_ids[:cfg.max_context]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

def varlen_collate(batch):
    """
    Concatenates batch into a single flat sequence to eliminate padding.
    Cleaned dict output removes the max_seqlen integer logic causing CPU syncs.
    """
    input_ids = []
    labels = []
    seqlens = []
    pos_ids = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        input_ids.append(item["input_ids"])
        labels.append(item["labels"])
        seqlens.append(seq_len)
        
        pos_ids.append(torch.arange(seq_len, dtype=torch.long))
        
    flat_input_ids = torch.cat(input_ids).unsqueeze(0)
    flat_labels = torch.cat(labels).unsqueeze(0)
    flat_pos_ids = torch.cat(pos_ids).unsqueeze(0)
    
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    
    return {
        "input_ids": flat_input_ids,
        "labels": flat_labels,
        "pos_ids": flat_pos_ids,
        "cu_seqlens": cu_seqlens.unsqueeze(0)
    }

def train():
    model = get_model()
    
    train_ds = ArrowDatasetWrapper(cfg.data_dir)
    test_ds = ArrowDatasetWrapper(cfg.test_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        gradient_checkpointing=cfg.grad_checkpoint,
        bf16=cfg.bf16,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_steps=cfg.eval_steps,
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=varlen_collate
    )

    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            last_checkpoint = os.path.join(cfg.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    if trainer.is_world_process_zero():
        print("Saving final model...")
        model_state = model.state_dict()
        output_path = os.path.join(str(cfg.output_dir), "final_model")
        os.makedirs(output_path, exist_ok=True)
        torch.save(model_state, os.path.join(output_path, "pytorch_model.bin"))
        print("Model saved successfully.")

if __name__ == "__main__":
    train()
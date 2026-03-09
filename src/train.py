import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from config import cfg
from model import get_model 

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class PretokenizedCipherDataset(Dataset):
    """
    Directly loads Arrow files mapped by the preprocessing pipeline.
    Because we use varlen_collate, we do NOT pad sequences here.
    """
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Enforce Equal Loss Weighting and truncate if necessary
        input_ids = item["input_ids"][:cfg.max_context]
        labels = item["labels"][:cfg.max_context]

        # MODIFIED: Ensure PAD tokens are explicitly stripped before Varlen Packing
        # This prevents the Tensor Cores from wasting FLOPs on padding logic
        valid_lengths = [i for i, token in enumerate(input_ids) if token != cfg.pad_token_id]
        if valid_lengths:
            actual_len = valid_lengths[-1] + 1
            input_ids = input_ids[:actual_len]
            labels = labels[:actual_len]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def varlen_collate(batch):
    """
    Packs variable-length sequences into a single flat 1D tensor.
    Calculates cu_seqlens required by flash_attn_varlen_func.
    No padding tokens (-100 or 0) are needed, optimizing Tensor Core usage!
    """
    input_ids = []
    labels = []
    seqlens = []
    pos_ids = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        input_ids.append(torch.tensor(item["input_ids"], dtype=torch.long))
        labels.append(torch.tensor(item["labels"], dtype=torch.long))
        seqlens.append(seq_len)
        
        # Absolute positional IDs for RoPE
        pos_ids.append(torch.arange(seq_len, dtype=torch.long))
        
    # Flatten across the batch into 1D contiguous tensors
    flat_input_ids = torch.cat(input_ids).unsqueeze(0)
    flat_labels = torch.cat(labels).unsqueeze(0)
    flat_pos_ids = torch.cat(pos_ids).unsqueeze(0)
    
    # Cumulative sequence lengths (starts with 0) for attention boundaries
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    actual_max_seqlen = max(seqlens)
    
    return {
        "input_ids": flat_input_ids,
        "labels": flat_labels,
        "pos_ids": flat_pos_ids,
        "cu_seqlens": cu_seqlens.unsqueeze(0),
        "max_seqlen": actual_max_seqlen
    }

def train():
    model = get_model()
    
    # Directly load the pre-tokenized offline datasets
    train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
    test_ds = PretokenizedCipherDataset(cfg.tokenized_test_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        bf16=cfg.bf16,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=8, 
        dataloader_pin_memory=True,
        fsdp="full_shard auto_wrap", 
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["CustomLayer"],
            "activation_checkpointing": cfg.grad_checkpoint
        },
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=varlen_collate
    )

    # Check if there is a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            last_checkpoint = os.path.join(cfg.output_dir, checkpoints[-1])
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(str(cfg.output_dir), "final_model"))

if __name__ == "__main__":
    train()
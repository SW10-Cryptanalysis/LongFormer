import os
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from config import cfg
from model import get_model 

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def varlen_collate(batch):
    """
    Packs variable-length sequences into a single flat 1D tensor to bypass padding tokens.
    Calculates cu_seqlens required by flash_attn_varlen_func.
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
        
    # Flatten across the batch
    flat_input_ids = torch.cat(input_ids).unsqueeze(0)
    flat_labels = torch.cat(labels).unsqueeze(0)
    flat_pos_ids = torch.cat(pos_ids).unsqueeze(0)
    
    # Cumulative sequence lengths (starts with 0)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    actual_max_seqlen = max(seqlens)
    
    return {
        "input_ids": flat_input_ids,
        "labels": flat_labels,
        "pos_ids": flat_pos_ids,
        "cu_seqlens": cu_seqlens.unsqueeze(0),
        "max_seqlen": actual_max_seqlen
    }

def get_datasets():
    # 1. Check if we already have the NEW tokenized data saved
    if os.path.exists(cfg.tokenized_data_dir) and os.path.exists(cfg.tokenized_test_dir):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("Found Varlen-ready tokenized datasets! Loading from disk...")
        train_ds = load_from_disk(str(cfg.tokenized_data_dir))
        test_ds = load_from_disk(str(cfg.tokenized_test_dir))
        return train_ds, test_ds

    # 2. If not, load the raw Arrow datasets and tokenize
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Tokenized datasets not found. Loading raw Arrow datasets...")
        
    train_ds = load_from_disk(str(cfg.data_dir))
    test_ds = load_from_disk(str(cfg.test_dir))

    max_homophone = cfg.unique_homophones 
    sep_token = max_homophone + 1
    unk_token = max_homophone + 2
    char_offset = unk_token + 1
    chars = "abcdefghijklmnopqrstuvwxyz "
    char_to_id = {char: i + char_offset for i, char in enumerate(chars)}

    def prepare_dataset(example):
        cipher_ids = [int(x) for x in example["ciphertext"].split()]
        plain_text = example.get("plaintext", "")
        plain_ids = [char_to_id.get(char, unk_token) for char in plain_text] 
        
        # [C1, C2...][SEP][P1, P2...] format
        input_ids = cipher_ids + [sep_token] + plain_ids
        if len(input_ids) > cfg.max_context:
            input_ids = input_ids[:cfg.max_context]
            
        return {"input_ids": input_ids, "labels": input_ids}

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Tokenizing datasets with UNK offset (this will only happen once!)...")
        
    train_ds = train_ds.map(
        prepare_dataset, 
        num_proc=8, # Reduced slightly for 64GB RAM safety
        remove_columns=train_ds.column_names
    )
    test_ds = test_ds.map(
        prepare_dataset, 
        num_proc=8, 
        remove_columns=test_ds.column_names
    )

    # 3. Save to disk so subsequent runs load instantly
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Saving tokenized training data to {cfg.tokenized_data_dir}...")
        train_ds.save_to_disk(str(cfg.tokenized_data_dir))
        print(f"Saving tokenized test data to {cfg.tokenized_test_dir}...")
        test_ds.save_to_disk(str(cfg.tokenized_test_dir))

    return train_ds, test_ds

def train():
    model = get_model()
    
    # Retrieve datasets (either loads cached or tokenizes and saves)
    train_ds, test_ds = get_datasets()

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
        eval_steps=cfg.eval_steps,
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=8, 
        dataloader_pin_memory=True,
        fsdp="full_shard auto_wrap", 
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["CustomLayer"],
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
import os
import torch
import logging
import transformers
import time
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerState,
    TrainerCallback,
    set_seed,
)
from src.config import cfg
from src.model import get_model
from easy_logging import EasyFormatter

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class HardwareOptimizationCallback(TrainerCallback):
    """Custom callback for AAU AI-Lab 4-8x L4 GPU performance telemetry."""

    def __init__(self) -> None:
        """Initialize the callback."""
        self.epoch_start_time = 0

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ) -> None:
        """Log the start time of the epoch."""
        self.epoch_start_time = time.time()
        self.epoch_start_step = state.global_step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ) -> None:
        """Log performance metrics at the end of the epoch."""
        epoch_time = time.time() - self.epoch_start_time
        steps_this_epoch = state.global_step - self.epoch_start_step
        num_devices = max(torch.cuda.device_count(), 1)

        # Tokens per second (Approximate based on max context)
        total_tokens_per_epoch = (
            steps_this_epoch
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * cfg.max_len
            * num_devices
        )
        tokens_per_sec = total_tokens_per_epoch / epoch_time if epoch_time > 0 else 0

        # VRAM Tracking (Memory Management)
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        logger.info(f"--- Epoch {state.epoch} Performance Metrics ---")
        logger.info(f"Wall-clock time: {epoch_time:.2f} seconds")
        logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.info(f"Peak VRAM: {peak_vram_gb:.2f} GB")
        logger.info(
            (
                f"Wall-clock time per 10k-token sample: {(epoch_time / (total_tokens_per_epoch / 10000)):.4f} seconds"
                if total_tokens_per_epoch > 0
                else "N/A"
            ),
        )
        logger.info("-" * 40)


class VarlenTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to argmax logits on-device before CPU offload."""
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs["loss"]
        logits = outputs.get("logits")

        if logits is not None:
            # Argmax on GPU — cheap, avoids offloading full float logits
            logits = logits.argmax(dim=-1)

        labels = inputs.get("labels")
        return (loss, logits, labels)


def log_environment_details(seed: int) -> None:
    """Logs SOTA Implementation library versions and hardware details."""
    logger.info("=== AAU AI-Lab Execution Environment ===")
    logger.info(f"Seed: {seed}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Transformers Version: {transformers.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {device_count}")
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        import flash_attn  # type: ignore

        logger.info(f"FlashAttention-2 Version: {flash_attn.__version__}")
    except ImportError:
        logger.warning("FlashAttention-2 not found. Sequence bottlenecks may occur.")
    logger.info("========================================")


class PretokenizedCipherDataset(Dataset):
    """Directly loads Arrow files mapped by the preprocessing pipeline.

    Because we use varlen_collate, we do NOT pad sequences here.
    """

    def __init__(self, directory_path: Path) -> None:
        """Load the pre-tokenized HuggingFace dataset from disk."""
        self.hf_dataset = load_from_disk(str(directory_path))

        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """Return a single sample with input_ids and labels, truncated and stripped of padding."""
        item = self.hf_dataset[idx]

        if len(item["input_ids"]) > cfg.max_len or len(item["labels"]) > cfg.max_len:
            logger.info(
                f"Sample {idx} truncated: input_ids {len(item['input_ids'])} -> {cfg.max_len}, labels {len(item['labels'])} -> {cfg.max_len}",
            )

        # Enforce Equal Loss Weighting and truncate if necessary
        input_ids = item["input_ids"][: cfg.max_len]
        labels = item["labels"][: cfg.max_len]

        # This prevents the Tensor Cores from wasting FLOPs on padding logic
        valid_lengths = [
            i for i, token in enumerate(input_ids) if token != cfg.pad_token_id
        ]
        if valid_lengths:
            actual_len = valid_lengths[-1] + 1
            input_ids = input_ids[:actual_len]
            labels = labels[:actual_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def varlen_collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor | int]:
    """Pack variable-length sequences into a single flat 1D tensor.

    Calculates cu_seqlens required by flash_attn_varlen_func.
    No padding tokens (-100 or 0) are needed, optimizing Tensor Core usage.
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
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    actual_max_seqlen = max(seqlens)

    return {
        "input_ids": flat_input_ids,
        "labels": flat_labels,
        "pos_ids": flat_pos_ids,
        "cu_seqlens": cu_seqlens.unsqueeze(0),
        "max_seqlen": actual_max_seqlen,
    }


def compute_metrics(
    eval_preds: EvalPrediction | tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Compute symbol error rate (SER) while ignoring padded labels."""
    if isinstance(eval_preds, tuple):
        predictions, labels = eval_preds
    else:
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    total_errors = 0
    total_symbols = 0

    for i in range(labels.shape[0]):
        mask = labels[i] != -100
        val_labels = labels[i][mask]
        val_preds = predictions[i][mask]
        total_errors += np.sum(val_labels != val_preds)
        total_symbols += len(val_labels)

    ser = total_errors / total_symbols if total_symbols > 0 else 0.0
    return {"SER": ser}


def train() -> None:
    # Safety check
    if cfg.vocab_size == 0 or cfg.max_len == 0 or cfg.unique_homophones == 0:
        raise ValueError(
            f"CRITICAL CONFIG ERROR: dimension was not initialized properly!\n"
            f"vocab_size: {cfg.vocab_size}\n"
            f"max_len: {cfg.max_len}\n"
            f"unique_homophones: {cfg.unique_homophones}\n"
            f"Check the Config class and load_homophones() method.",
        )

    # Seed Tracking
    run_seed = 42
    set_seed(run_seed)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        log_environment_details(run_seed)
    model = get_model()

    if cfg.use_spaces:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("Using space tokens in training.")
        train_ds = PretokenizedCipherDataset(cfg.tokenized_spaced_train_dir)
        val_ds = PretokenizedCipherDataset(cfg.tokenized_spaced_val_dir)
    else:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("Not using space tokens in training.")
        train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
        val_ds = PretokenizedCipherDataset(cfg.tokenized_val_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.grad_checkpoint,
        eval_accumulation_steps=1,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        seed=run_seed,
    )

    trainer = VarlenTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=varlen_collate,
        compute_metrics=compute_metrics,
        callbacks=[HardwareOptimizationCallback()],
    )

    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [
            d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(cfg.output_dir, checkpoints[-1])
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                pass

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_name = "final_model"
    if cfg.use_spaces:
        final_model_name += "_with_spaces"
    else:
        final_model_name += "_no_spaces"

    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(str(cfg.output_dir), final_model_name))


if __name__ == "__main__":
    train()

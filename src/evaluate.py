import os
from pathlib import Path
import torch
import Levenshtein
import logging
from datasets import load_from_disk
from src.model import get_model, RecurrenceModel
from easy_logging import EasyFormatter
from src.config import cfg

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def _resolve_model_path() -> Path | None:
    """Return the path to the best available model checkpoint, or None if none found."""
    model_path = cfg.output_dir
    if not os.path.exists(
        os.path.join(model_path, "model.safetensors"),
    ) and os.path.isdir(cfg.output_dir):
        checkpoints = [
            d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")
        ]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        model_path = Path(os.path.join(cfg.output_dir, checkpoints[-1]))
    return model_path


def _load_model(model_path: Path, device: torch.device) -> RecurrenceModel:
    """Load model weights from safetensors or pytorch_model.bin into a fresh RecurrenceModel."""
    model = get_model()
    state_dict_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file

        state_dict = load_file(state_dict_path)
    else:
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location=device)

    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    else:
        model = model.half()
    model.eval()
    return model


def _generate_tokens(
    model: RecurrenceModel,
    input_ids: list[int],
    chars_to_generate: int,
    eos_token: int,
    char_offset: int,
    device: torch.device,
) -> list[int]:
    """Auto-regressively generate up to chars_to_generate token IDs from input_ids."""
    generated_ids: list[int] = []
    curr_input_ids = input_ids[:]

    with torch.no_grad():
        for _ in range(chars_to_generate):
            seq_len = len(curr_input_ids)
            input_tensor = torch.tensor(
                [curr_input_ids],
                dtype=torch.long,
                device=device,
            )
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(
                0,
            )
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

            outputs = model(
                input_ids=input_tensor,
                cu_seqlens=cu_seqlens,
                pos_ids=pos_ids,
                max_seqlen=seq_len,
            )

            next_token = int(torch.argmax(outputs["logits"][0, -1, :]).item())
            generated_ids.append(next_token)
            curr_input_ids.append(next_token)

            if next_token == eos_token or next_token < char_offset:
                break

    return generated_ids


def evaluate() -> None:
    """Evaluate the trained model on the validation set and print SER for each sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = _resolve_model_path()
    if model_path is None:
        return

    model = _load_model(model_path, device)

    bos_token = cfg.bos_token_id
    sep_token = cfg.sep_token_id
    eos_token = cfg.eos_token_id
    char_offset = cfg.char_offset
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    try:
        val_ds = load_from_disk(str(cfg.tokenized_val_dir))
    except Exception:
        return

    eval_subset = val_ds.select(range(min(10, len(val_ds))))

    for _i, item in enumerate(eval_subset):
        cipher_ids = [int(x) for x in item["ciphertext"].split()]
        true_plain = item["plaintext"]

        max_cipher_len = cfg.max_context - 200
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        input_ids = [bos_token] + cipher_ids + [sep_token]

        generated_ids = _generate_tokens(
            model,
            input_ids,
            min(len(true_plain), 100),  # Limit generation to 100 chars
            eos_token,
            char_offset,
            device,
        )

        pred_plain = "".join(
            [id_to_char.get(idx, "?") for idx in generated_ids],
        ).replace("?", "")
        true_plain_subset = true_plain[: len(pred_plain)]
        lev_distance = Levenshtein.distance(true_plain_subset, pred_plain) / max(
            len(true_plain_subset),
            1,
        )
        logger.info(
            f"Sample {_i}: SER={lev_distance:.2f} | True: {true_plain_subset} | Pred: {pred_plain}",
        )


if __name__ == "__main__":
    evaluate()

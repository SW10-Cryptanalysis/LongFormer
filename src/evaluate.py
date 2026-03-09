import os
import torch
import Levenshtein
from datasets import load_from_disk
from model import get_model
from config import cfg

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Find the model directory
    model_path = cfg.output_dir
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        if os.path.isdir(cfg.output_dir):
            checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                model_path = os.path.join(cfg.output_dir, checkpoints[-1])
            else:
                print(f"No model.safetensors found in {cfg.output_dir} or its subfolders.")
                return

    print(f"Loading model architecture and weights from {model_path}...")
    model = get_model()
    
    state_dict_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
    else:
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
        else:
            print("Could not find model weights (safetensors or bin).")
            return

    # Remove the "_orig_mod." prefix if the model was saved with torch.compile
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    
    model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    else:
        model = model.half() 
    model.eval()

    # 2. Setup SOTA tokenization mapping
    bos_token = cfg.bos_token_id
    sep_token = cfg.sep_token_id
    eos_token = cfg.eos_token_id
    char_offset = cfg.char_offset
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    # 3. Load Test Data from Arrow
    print(f"Loading test dataset from {cfg.tokenized_val_dir}...")
    try:
        val_ds = load_from_disk(str(cfg.tokenized_val_dir))
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return

    num_tests = min(10, len(val_ds))
    eval_subset = val_ds.select(range(num_tests))
    
    print(f"\nTesting on {num_tests} sequences...\n")
    print("=" * 60)

    for i, item in enumerate(eval_subset):
        cipher_ids = [int(x) for x in item["ciphertext"].split()]
        true_plain = item["plaintext"]
        
        # Truncate cipher if it's too long to leave room for generation
        max_cipher_len = cfg.max_context - 200 
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        # MODIFIED: Explicitly inject BOS to match the Recurrence-Encoded format
        input_ids = [bos_token] + cipher_ids + [sep_token]
        
        print(f"File {i+1}/{num_tests}")
        print(f"Cipher length: {len(cipher_ids)} tokens")
        print(f"True Plaintext (first 100 chars): {true_plain[:100]}...")
        
        # 4. Generate plaintext
        generated_ids = []
        chars_to_generate = min(len(true_plain), 100) 
        curr_input_ids = input_ids[:] 
        
        print("Generating...", end="", flush=True)
        with torch.no_grad():
            for _ in range(chars_to_generate):
                seq_len = len(curr_input_ids)
                
                input_tensor = torch.tensor([curr_input_ids], dtype=torch.long, device=device)
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
                
                outputs = model(
                    input_ids=input_tensor,
                    cu_seqlens=cu_seqlens,
                    pos_ids=pos_ids,
                    max_seqlen=seq_len
                )
                
                next_token_logits = outputs["logits"][0, -1, :] 
                next_token = int(torch.argmax(next_token_logits).item())
                
                generated_ids.append(next_token)
                curr_input_ids.append(next_token)
                
                # MODIFIED: Explicitly halt on EOS to prevent context bleeding
                if next_token == eos_token or next_token < char_offset:
                    break
        
        # 5. Decode and Calculate SER
        pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])
        # Clean up any potential EOS token mapping in the string representation
        pred_plain = pred_plain.replace("?", "")
        
        true_plain_subset = true_plain[:len(pred_plain)]
        ser = Levenshtein.distance(true_plain_subset, pred_plain) / max(len(true_plain_subset), 1)
        
        print("\r" + " " * 20 + "\r", end="") 
        print(f"Pred Plaintext (first 100 chars): {pred_plain}")
        print(f"Symbol Error Rate (SER): {ser:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    evaluate()
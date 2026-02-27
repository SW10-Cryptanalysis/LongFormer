import json
import glob
import os
import torch
import Levenshtein
from model import get_model
from config import cfg

def evaluate():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Find the model directory
    model_path = cfg.output_dir
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        # Check if it's in a checkpoint folder
        if os.path.isdir(cfg.output_dir):
            checkpoints = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                model_path = os.path.join(cfg.output_dir, checkpoints[-1])
            else:
                print(f"No model.safetensors found in {cfg.output_dir} or its subfolders.")
                return

    print(f"Loading model from {model_path}...")
    
    # 2. Load the model
    # model = LongformerForCausalLM.from_pretrained(model_path)
    # Since we are using a custom nn.Module, we load architecture then weights
    model = get_model()
    
    # Check if safe tensors or bin
    state_dict_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
    else:
        # Fallback to bin
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
        else:
            print("Could not find model weights (safetensors or bin).")
            return

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # GTX 1660 doesn't support bfloat16 natively, but supports float16. 
    # Converting to float16 saves VRAM and runs faster.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    else:
        model = model.half() 
    model.eval()

    # 3. Setup tokenization (matching train.py)
    max_homophone = cfg.unique_homophones 
    sep_token = max_homophone + 1
    char_offset = sep_token + 1
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    # 4. Load test files (limit to 10)
    test_files = glob.glob(os.path.join(cfg.eval_dir, "*.json"))[:10]
    
    if not test_files:
        print(f"No test files found in {cfg.eval_dir}.")
        return

    print(f"\nTesting on {len(test_files)} files...\n")
    print("=" * 60)

    for file_path in test_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        true_plain = data["plaintext"]
        
        # Truncate cipher if it's too long to leave room for generation
        max_cipher_len = cfg.max_context - 200 
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        # Prepare input
        input_ids = cipher_ids + [sep_token]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        # attention_mask = torch.ones_like(input_tensor).to(device) # Unused in custom model which handles causal masking
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Cipher length: {len(cipher_ids)} tokens")
        print(f"True Plaintext (first 100 chars): {true_plain[:100]}...")
        
        # 5. Generate plaintext (Manual greedy decoding)
        generated_ids = []
        chars_to_generate = min(len(true_plain), 100) # Generate up to 100 chars for a quick test
        
        curr_input_ids = input_ids[:] # Copy list
        
        print("Generating...", end="", flush=True)
        with torch.no_grad():
            for _ in range(chars_to_generate):
                # Prepare tensor
                input_tensor = torch.tensor([curr_input_ids], dtype=torch.long).to(device)
                # Forward pass
                outputs = model(input_tensor)
                next_token_logits = outputs["logits"][0, -1, :] # Last token
                
                # Greedy choice
                next_token = int(torch.argmax(next_token_logits).item())
                
                # Append
                generated_ids.append(next_token)
                curr_input_ids.append(next_token)
                
                # Stop if padding (0)
                if next_token == 0:
                    break
            
        # The output includes the input prompt, so we slice it off to just get the new tokens
        # generated_ids = output_sequence[0][input_tensor.shape[1]:].tolist()

        # 6. Decode the generated tokens back to characters
        pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])
        
        # Calculate Symbol Error Rate (SER)
        true_plain_subset = true_plain[:len(pred_plain)]
        ser = Levenshtein.distance(true_plain_subset, pred_plain) / max(len(true_plain_subset), 1)
        
        print("\r" + " " * 20 + "\r", end="") # Clear "Generating..." text
        print(f"Pred Plaintext (first 100 chars): {pred_plain}")
        print(f"Symbol Error Rate (SER): {ser:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    evaluate()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from config import cfg

# 1. Try importing Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("FlashAttention-2 loaded successfully.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("FlashAttention-2 not found. Will fallback to PyTorch Native SDPA.")

# Try importing Liger Kernel optimizations
try:
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
        LigerRMSNorm,
        LigerSwiGLUMLP
    )
    LIGER_AVAILABLE = True
    print("Liger Kernel loaded successfully.")
except ImportError:
    LIGER_AVAILABLE = False
    print("Liger Kernel not found. Using standard PyTorch implementations.")

class RotatedEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, heads, head_dim]
        # Create position indices based on absolute position in the packed sequence causes issues
        # because we reset 'cu_seqlens'. But for RoPE in a sliding window, relative local is key.
        # Here we assume causal absolute positions provided by caller or inferred.
        if seq_len is None:
            seq_len = x.shape[1]
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(2) # [1, seq, 1, dim]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, freqs):
    # freqs: [1, seq, 1, dim]
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())

class FlashAttentionLayer(nn.Module):
    """
    Implements exact O(N^2) attention using FlashAttention-2.
    Removes chunking overhead and processes 16k tokens efficiently in SRAM.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.dims
        self.num_heads = config.att_heads
        self.head_dim = config.dims // config.att_heads
        self.rope = RotatedEmbedding(self.head_dim, config.rope_theta)
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, cu_seqlens=None, **kwargs):
        batch, seq_len, dim = hidden_states.shape
        
        # Project Q, K, V
        # Shape: [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        freqs = self.rope(q, seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # FlashAttention-2 requires inputs to be half-precision (fp16 or bf16)
        if q.dtype not in [torch.float16, torch.bfloat16]:
            target_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
            q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)

        # Compute Attention
        if FLASH_ATTN_AVAILABLE:
            # flash_attn_func handles causal masking natively and optimally
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        else:
            # Fallback to PyTorch Native SDPA (which also triggers FlashAttention under the hood on newer PyTorch)
            # SDPA expects shape: [batch, num_heads, seq_len, head_dim]
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, 
                is_causal=True
            )
            # Back to [batch, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2)

        # Reshape and project out
        attn_output = attn_output.contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class CustomLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FlashAttentionLayer(config)
        
        if LIGER_AVAILABLE and config.use_liger:
            self.mlp = LigerSwiGLUMLP(config) 
            self.att_norm = LigerRMSNorm(config.dims)
            self.mlp_norm = LigerRMSNorm(config.dims)
        else:
            # Standard PyTorch Fallback
            self.mlp = nn.Sequential(
                nn.Linear(config.dims, config.dims * 4),
                nn.SiLU(),
                nn.Linear(config.dims * 4, config.dims)
            )
            self.att_norm = nn.RMSNorm(config.dims)
            self.mlp_norm = nn.RMSNorm(config.dims)

    def forward(self, x, **kwargs):
        # Pre-Norm Architecture
        residual = x
        x = self.att_norm(x)
        x = self.attention(x, **kwargs)
        x = residual + x
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class RecurrenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dims)
        self.gradient_checkpointing = False
        
        self.layers = nn.ModuleList([
            CustomLayer(config) for _ in range(config.layers)
        ])
        
        if LIGER_AVAILABLE and config.use_liger:
            self.norm = LigerRMSNorm(config.dims)
            self.output_head = LigerFusedLinearCrossEntropyLoss()
        else:
            self.norm = nn.RMSNorm(config.dims)
            self.output_head = nn.Linear(config.dims, config.vocab_size, bias=False)
            
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(self, input_ids, output_hidden_states=False, labels=None, **kwargs):
        # input_ids: Recurrence distance IDs [batch, seq]
        x = self.embed(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False is the modern, safer PyTorch implementation
                x = checkpoint.checkpoint(layer, x, use_reentrant=False, **kwargs)
            else:
                x = layer(x, **kwargs)
            
        x = self.norm(x)
        
        loss = None
        logits = None
        
        if labels is not None:
            # If using Liger Fused Loss, we pass the hidden states directly
            if LIGER_AVAILABLE and self.config.use_liger:
                shift_hidden = x[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Pass the embedding weight matrix, hidden states, and labels
                loss = self.output_head(
                    self.embed.weight, 
                    shift_hidden.view(-1, self.config.dims), 
                    shift_labels.view(-1)
                )
            else:
                logits = self.output_head(x)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if logits is None and not self.training:
             # Basic inference support
             if hasattr(self.output_head, "lin"):
                 logits = self.output_head.lin(x)
             else:
                 # In standard mode, output_head is Linear
                 logits = self.output_head(x) if not isinstance(self.output_head, LigerFusedLinearCrossEntropyLoss) else torch.matmul(x, self.embed.weight.t())

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": x if output_hidden_states else None
        }

def get_model():
    model = RecurrenceModel(cfg)
    print(f"Custom Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model

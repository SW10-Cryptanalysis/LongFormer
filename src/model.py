import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from config import cfg

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("FlashAttention-2 Varlen loaded successfully.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("FlashAttention-2 not found. Will fallback to PyTorch Native SDPA.")

try:
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
        LigerRMSNorm,
        LigerSwiGLUMLP
    )
    LIGER_AVAILABLE = True
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Liger Kernel loaded successfully.")
except ImportError:
    LIGER_AVAILABLE = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Liger Kernel not found. Using standard PyTorch implementations.")

class RotatedEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # Optimized to accept pre-computed trig tensors
    return (x * cos) + (rotate_half(x) * sin)

class FlashAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.dims
        self.num_heads = config.att_heads
        self.head_dim = config.dims // config.att_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None, cos=None, sin=None, **kwargs):
        total_tokens, dim = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)

        # Apply pre-computed RoPE cleanly
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if FLASH_ATTN_AVAILABLE and cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int32)
            attn_output = flash_attn_varlen_func(
                q, k, v, 
                cu_seqlens_q=cu_seqlens, 
                cu_seqlens_k=cu_seqlens, 
                max_seqlen_q=max_seqlen, 
                max_seqlen_k=max_seqlen, 
                dropout_p=0.0, 
                causal=True
            )
        else:
            raise NotImplementedError("Varlen currently requires FlashAttention-2.")

        attn_output = attn_output.contiguous().view(total_tokens, self.hidden_size)
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
            self.mlp = nn.Sequential(
                nn.Linear(config.dims, config.dims * 4),
                nn.SiLU(),
                nn.Linear(config.dims * 4, config.dims)
            )
            self.att_norm = nn.RMSNorm(config.dims)
            self.mlp_norm = nn.RMSNorm(config.dims)

    def forward(self, x, **kwargs):
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
        
        # Centralized RoPE Instantiation
        self.head_dim = config.dims // config.att_heads
        self.rope = RotatedEmbedding(self.head_dim, config.rope_theta)
        
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
    
    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(self, input_ids, output_hidden_states=False, labels=None, cu_seqlens=None, pos_ids=None, **kwargs):
        if input_ids.dim() == 2 and input_ids.shape[0] == 1 and cu_seqlens is not None:
            input_ids = input_ids.squeeze(0)
            if labels is not None:
                labels = labels.squeeze(0)
            cu_seqlens = cu_seqlens.squeeze(0)
            if pos_ids is not None:
                pos_ids = pos_ids.squeeze(0)
            
        x = self.embed(input_ids)
        
        if self.gradient_checkpointing and self.training:
            x.requires_grad_(True)
            
        # 1. Eliminate CPU syncs by providing a static max sequence bound for FA-2 block allocation
        max_seqlen = self.config.max_context
        
        # 2. Compute RoPE Operations ONCE centrally to save immense VRAM and Tensor Core utilization
        cos, sin = None, None
        if pos_ids is not None:
            inv_freq = self.rope.inv_freq.to(x.device)
            freqs = torch.outer(pos_ids, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
            amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
            cos = emb.cos().to(amp_dtype)
            sin = emb.sin().to(amp_dtype)
        
        layer_kwargs = {
            "cu_seqlens": cu_seqlens, 
            "max_seqlen": max_seqlen, 
            "cos": cos, 
            "sin": sin
        }
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False, **layer_kwargs)
            else:
                x = layer(x, **layer_kwargs)
            
        x = self.norm(x)
        
        loss = None
        logits = None
        
        if labels is not None:
            shift_hidden = x[:-1, :].contiguous()
            shift_labels = labels[1:].contiguous().clone()
            
            if cu_seqlens is not None and len(cu_seqlens) > 2:
                boundary_indices = cu_seqlens[1:-1] - 1
                shift_labels[boundary_indices] = -100

            if LIGER_AVAILABLE and self.config.use_liger:
                loss = self.output_head(self.embed.weight, shift_hidden, shift_labels)
            else:
                logits_for_loss = self.output_head(shift_hidden)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits_for_loss, shift_labels)
        
        if not self.training:
            if isinstance(self.output_head, LigerFusedLinearCrossEntropyLoss):
                logits = torch.matmul(x, self.embed.weight.t())
            else:
                logits = self.output_head(x)
            
            logits = logits.unsqueeze(0)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": x.unsqueeze(0) if output_hidden_states else None
        }

def get_model():
    model = RecurrenceModel(cfg)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Custom Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model
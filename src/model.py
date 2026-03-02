import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from config import cfg

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

class BlockSlidingWindowAttention(nn.Module):
    """
    Implements O(N * W) attention by chunking the sequence into blocks of size window_size.
    Each block attends to itself and the previous block (Total window = 2 * window_size).
    Standard Softmax attention is applied for sharp deterministic mappings.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.dims
        self.num_heads = config.att_heads
        self.head_dim = config.dims // config.att_heads
        self.window_size = config.window_size
        self.rope = RotatedEmbedding(self.head_dim, config.rope_theta)
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # NOTE: ALiBi slopes removed. RoPE handles relative positioning implicitly.

    def forward(self, hidden_states, cu_seqlens=None, **kwargs):
        # hidden_states: [batch, total_seq_len, dim] (Packed)
        batch, seq_len, dim = hidden_states.shape
        seq_len_scalar = int(seq_len)
        
        q = self.q_proj(hidden_states).view(batch, seq_len_scalar, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len_scalar, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len_scalar, self.num_heads, self.head_dim)

        # Apply RoPE
        freqs = self.rope(q, seq_len_scalar)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Chunking Strategy for O(N) Complexity
        w = self.window_size
        bsz, seq_len, _, _ = q.shape
        
        # Calculate padding needed
        pad_len = (w - (seq_len % w)) % w
        
        if pad_len > 0:
            q = F.pad(q, (0,0,0,0,0,pad_len))
            k = F.pad(k, (0,0,0,0,0,pad_len))
            v = F.pad(v, (0,0,0,0,0,pad_len))
        
        padded_len = q.shape[1]
             
        num_chunks = int(padded_len // w)
        
        # Reshape to [batch, num_chunks, window_size, heads, dim]
        q_chunks = q.reshape(batch, num_chunks, w, self.num_heads, self.head_dim)
        k_chunks = k.reshape(batch, num_chunks, w, self.num_heads, self.head_dim)
        v_chunks = v.reshape(batch, num_chunks, w, self.num_heads, self.head_dim)

        out_chunks = []
        
        for i in range(num_chunks):
            q_i = q_chunks[:, i]     # [b, w, h, d]
            
            # Key/Value is concat of prev and curr
            if i == 0:
                k_cat = k_chunks[:, i] # [b, w, h, d]
                v_cat = v_chunks[:, i]
            else:
                k_cat = torch.cat([k_chunks[:, i-1], k_chunks[:, i]], dim=1) # [b, 2w, h, d]
                v_cat = torch.cat([v_chunks[:, i-1], v_chunks[:, i]], dim=1)
                
            # Attention scores: Q @ K.T
            # [b, w, h, d] @ [b, h, d, 2w] -> [b, h, w, 2w]
            attn_scores = torch.einsum("bwhd,bkhd->bhwk", q_i, k_cat)
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            
            # Causal Masking (Critical)
            if i == 0:
                 mask = torch.triu(torch.ones(w, w, device=q.device), diagonal=1).bool()
                 attn_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                 # NOTE: ALiBi Local removed.
                 
            else:
                causal_part = torch.triu(torch.ones(w, w, device=q.device), diagonal=1).bool()
                full_mask = torch.cat([torch.zeros(w, w, dtype=torch.bool, device=q.device), causal_part], dim=1)
                attn_scores.masked_fill_(full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                # NOTE: ALiBi Relative removed.

            # STANDARD SOFTMAX ATTENTION (Replaces Sigmoid)
            # Computed over the key dimension (dim=-1)
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Output
            out = torch.matmul(attn_probs, v_cat.transpose(1, 2)).transpose(1, 2)
            out_chunks.append(out)

        # Reassemble
        output = torch.cat(out_chunks, dim=1)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_len, :, :]
            
        output = output.reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(output)

class CustomLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BlockSlidingWindowAttention(config)
        
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

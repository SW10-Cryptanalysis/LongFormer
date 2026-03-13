import os
import math
from typing import cast
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from src.config import cfg, Config

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore

    FLASH_ATTN_AVAILABLE = True
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass

try:
    from liger_kernel.transformers import (  # type: ignore
        LigerFusedLinearCrossEntropyLoss,
        LigerRMSNorm,
        LigerSwiGLUMLP,
    )

    LIGER_AVAILABLE = True
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass
except ImportError:
    LIGER_AVAILABLE = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass


class RotatedEmbedding(nn.Module):
    """Precomputes inverse frequencies for Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, base: int | float = 10000) -> None:
        """Initialise inverse frequency buffer for the given head dimension."""
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting and negating the second half."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to tensor x using precomputed cos/sin."""
    return (x * cos) + (rotate_half(x) * sin)


class FlashAttentionLayer(nn.Module):
    """Multi-head self-attention using FlashAttention-2 varlen kernel."""

    def __init__(self, config: Config) -> None:
        """Initialise Q, K, V, and output projections from config."""
        super().__init__()
        self.hidden_size = config.dims
        self.num_heads = config.att_heads
        self.head_dim = config.dims // config.att_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run varlen flash attention over packed sequences."""
        total_tokens, dim = hidden_states.shape

        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)

        if cos is not None and sin is not None:
            # FlashAttention requires strictly contiguous tensors
            q = apply_rope(q, cos, sin).contiguous()
            k = apply_rope(k, cos, sin).contiguous()
            v = v.contiguous()

        if FLASH_ATTN_AVAILABLE and cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int32)
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=True,
            )
        else:
            raise NotImplementedError("Varlen currently requires FlashAttention-2.")

        attn_output = attn_output.contiguous().view(total_tokens, self.hidden_size)
        return self.o_proj(attn_output)


class CustomLayer(nn.Module):
    """Single transformer block: attention + MLP with pre-norm residuals."""

    def __init__(self, config: Config) -> None:
        """Initialise attention, MLP, and normalisation layers from config."""
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
                nn.Linear(config.dims * 4, config.dims),
            )
            self.att_norm = nn.RMSNorm(config.dims)
            self.mlp_norm = nn.RMSNorm(config.dims)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Apply pre-norm attention then pre-norm MLP with residual connections."""
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
    """Custom causal language model with RoPE, FlashAttention, and optional Liger kernels."""

    _no_split_modules = ["CustomLayer"]

    def __init__(self, config: Config) -> None:
        """Build embedding, transformer layers, norm, and output head from config."""
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dims)
        self.gradient_checkpointing = False

        self.head_dim = config.dims // config.att_heads
        self.rope = RotatedEmbedding(self.head_dim, config.rope_theta)

        self.layers = nn.ModuleList([CustomLayer(config) for _ in range(config.layers)])

        if LIGER_AVAILABLE and config.use_liger:
            self.norm = LigerRMSNorm(config.dims)
            self.output_head = LigerFusedLinearCrossEntropyLoss()
        else:
            self.norm = nn.RMSNorm(config.dims)
            self.output_head = nn.Linear(config.dims, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._apply_depth_scaling()

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _apply_depth_scaling(self) -> None:
        std = 0.02
        scaled_std = std / math.sqrt(2 * self.config.layers)

        for name, p in self.named_parameters():
            if (
                name.endswith("o_proj.weight")
                or name.endswith("mlp.2.weight")
                or name.endswith("down_proj.weight")
            ):
                nn.init.normal_(p, mean=0.0, std=scaled_std)

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        """Enable gradient checkpointing to trade compute for memory."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input token embedding table."""
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the input token embedding table."""
        self.embed = value

    def _compute_rope(
        self,
        pos_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin RoPE tensors for the given position IDs."""
        inv_freq = cast(torch.Tensor, self.rope.inv_freq).to(device)
        freqs = torch.outer(pos_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        return emb.cos().to(amp_dtype), emb.sin().to(amp_dtype)

    def _compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shift hidden states and labels, mask boundaries, then compute CE loss."""
        shift_hidden = x[:-1, :].contiguous()
        shift_labels = labels[1:].contiguous().clone()

        if cu_seqlens is not None and len(cu_seqlens) > 2:
            boundary_indices = cu_seqlens[1:-1] - 1
            shift_labels[boundary_indices] = -100

        if LIGER_AVAILABLE and self.config.use_liger:
            return self.output_head(self.embed.weight, shift_hidden, shift_labels)  # type: ignore[return-value]
        logits_for_loss = self.output_head(shift_hidden)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits_for_loss, shift_labels)  # type: ignore[return-value]

    def _run_layers(
        self,
        x: torch.Tensor,
        layer_kwargs: dict,
    ) -> torch.Tensor:
        """Pass hidden states through all transformer layers with optional gradient checkpointing."""
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = cast(
                    torch.Tensor,
                    checkpoint.checkpoint(
                        layer,
                        x,
                        use_reentrant=False,
                        **layer_kwargs,
                    ),
                )
            else:
                x = layer(x, **layer_kwargs)
        return x

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits for inference."""
        if isinstance(self.output_head, LigerFusedLinearCrossEntropyLoss):
            return torch.matmul(x, self.embed.weight.t())
        return self.output_head(x)  # type: ignore[return-value]

    def _squeeze_varlen_inputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        pos_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Remove the leading batch dimension added by the varlen collator."""
        input_ids = input_ids.squeeze(0)
        if labels is not None:
            labels = labels.squeeze(0)
        cu_seqlens = cu_seqlens.squeeze(0)
        if pos_ids is not None:
            pos_ids = pos_ids.squeeze(0)
        return input_ids, labels, cu_seqlens, pos_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        labels: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        """Run a full forward pass and optionally compute the cross-entropy loss."""
        if input_ids.dim() == 2 and input_ids.shape[0] == 1 and cu_seqlens is not None:
            input_ids, labels, cu_seqlens, pos_ids = self._squeeze_varlen_inputs(
                input_ids,
                labels,
                cu_seqlens,
                pos_ids,
            )

        x = self.embed(input_ids)

        if self.gradient_checkpointing and self.training:
            x.requires_grad_(True)

        if max_seqlen is None:
            max_seqlen = self.config.max_context

        cos, sin = (None, None)
        if pos_ids is not None:
            cos, sin = self._compute_rope(pos_ids, x.device)

        layer_kwargs = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "cos": cos,
            "sin": sin,
        }

        x = self._run_layers(x, layer_kwargs)
        x = self.norm(x)

        loss = None
        logits = None

        if labels is not None:
            loss = self._compute_loss(x, labels, cu_seqlens)

        if not self.training:
            logits = self._compute_logits(x).unsqueeze(0)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": x.unsqueeze(0) if output_hidden_states else None,
        }


def get_model() -> RecurrenceModel:
    """Instantiate and return a RecurrenceModel using the global config."""
    model = RecurrenceModel(cfg)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass
    return model

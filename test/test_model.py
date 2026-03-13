import pytest
import torch
import torch.nn as nn
from src.config import Config
from src.model import (
    RotatedEmbedding,
    rotate_half,
    apply_rope,
    FlashAttentionLayer,
    RecurrenceModel,
    get_model,
)


@pytest.fixture
def dummy_cfg():
    cfg = Config()
    cfg.dims = 64
    cfg.att_heads = 4
    cfg.layers = 2
    cfg.vocab_size = 128
    cfg.use_liger = False
    return cfg


def test_rotated_embedding_and_helpers():
    dim = 16
    seq_len = 10
    rope = RotatedEmbedding(dim=dim)
    assert rope.inv_freq.shape == (dim // 2,)

    # Test rotate_half
    x = torch.randn(2, seq_len, dim)
    rotated = rotate_half(x)
    assert rotated.shape == x.shape

    # Test apply_rope shape stability
    cos = torch.randn(2, seq_len, dim)
    sin = torch.randn(2, seq_len, dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_flash_attention_layer_fallback(mocker, dummy_cfg):
    """Test the FlashAttention layer. Uses create=True for conditional hardware imports."""
    layer = FlashAttentionLayer(dummy_cfg)

    mocker.patch("src.model.FLASH_ATTN_AVAILABLE", True)

    # create=True allows patching symbols that failed to import (e.g., no CUDA on test runner)
    mock_flash = mocker.patch("src.model.flash_attn_varlen_func", create=True)
    mock_flash.return_value = torch.randn(
        10, dummy_cfg.att_heads, dummy_cfg.dims // dummy_cfg.att_heads
    )

    hidden_states = torch.randn(10, dummy_cfg.dims)
    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int32)
    max_seqlen = 5

    out = layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    assert out.shape == (10, dummy_cfg.dims)


def test_recurrence_model_instantiation(dummy_cfg):
    model = RecurrenceModel(dummy_cfg)
    assert isinstance(model.embed, nn.Embedding)
    assert len(model.layers) == dummy_cfg.layers
    assert model.gradient_checkpointing is False


def test_recurrence_model_forward_pass(mocker, dummy_cfg):
    model = RecurrenceModel(dummy_cfg)

    mocker.patch("src.model.FLASH_ATTN_AVAILABLE", True)

    # Inject conditionally imported FlashAttention dependency
    mock_flash = mocker.patch("src.model.flash_attn_varlen_func", create=True)
    mock_flash.return_value = torch.randn(
        8, dummy_cfg.att_heads, dummy_cfg.dims // dummy_cfg.att_heads
    )

    input_ids = torch.randint(0, dummy_cfg.vocab_size, (1, 8))
    labels = torch.randint(0, dummy_cfg.vocab_size, (1, 8))
    cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32)
    pos_ids = torch.arange(8).unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        cu_seqlens=cu_seqlens,
        pos_ids=pos_ids,
        max_seqlen=4,
    )

    assert "loss" in outputs
    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0  # scalar
    assert outputs["logits"] is None  # Logits are None during training


def test_get_model(mocker, dummy_cfg):
    mocker.patch("src.model.cfg", dummy_cfg)
    model = get_model()
    assert isinstance(model, RecurrenceModel)

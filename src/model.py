 
import torch
from transformers import LongformerConfig, LongformerModel
from config import Config


def get_model():
    """Init model with params from config"""

    conf = LongformerConfig(
        vocab_size=Config.vocab_size,
        max_position_embeddings=Config.max_context,
        hidden_size=Config.dims,
        num_hidden_layers=Config.layers,
        intermediate_size=Config.dims * 4,
        num_attention_heads=Config.att_heads,
        attention_window=512,
        # Standard stuff
        hidden_act="silu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
    )

    model = LongformerModel(conf)
    print("Longformer Model loaded!")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model
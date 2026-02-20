import sys
import os
import torch

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model import get_model, LongformerForCausalLM
from config import cfg

def test_get_model_structure():
    model = get_model()
    assert isinstance(model, LongformerForCausalLM)
    assert model.config.vocab_size == cfg.vocab_size
    assert model.config.hidden_size == cfg.dims
    assert model.config.max_position_embeddings == cfg.max_context

def test_model_forward_pass_no_labels():
    model = get_model()
    # Create a small random input batch (batch_size=1, seq_len=10)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    attention_mask = torch.ones((1, 10))
    
    # Run forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check outputs
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape == (1, 10, cfg.vocab_size)
    assert outputs.loss is None

def test_model_forward_pass_with_labels():
    model = get_model()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    # For Causal LM, labels are typically same as inputs (model handles shifting)
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    
    # Ensure loss is calculated
    assert outputs.loss is not None
    assert isinstance(outputs.loss, torch.Tensor)
    assert outputs.loss.item() > 0


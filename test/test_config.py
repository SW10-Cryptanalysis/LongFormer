from pathlib import Path
from src.config import Config, cfg


def test_config_instance_types():
    """Verify that all configuration attributes possess the correct types."""
    assert isinstance(cfg, Config)

    # Architecture
    assert isinstance(cfg.unique_homophones, int)
    assert isinstance(cfg.unique_letters, int)
    assert isinstance(cfg.vocab_size, int)
    assert isinstance(cfg.max_context, int)
    assert isinstance(cfg.dims, int)
    assert isinstance(cfg.layers, int)
    assert isinstance(cfg.att_heads, int)
    assert isinstance(cfg.window_size, int)
    assert isinstance(cfg.rope_theta, float)
    assert isinstance(cfg.use_liger, bool)
    assert isinstance(cfg.packing, bool)
    assert isinstance(cfg.bf16, bool)
    assert isinstance(cfg.hidden_act, str)

    # Properties
    assert isinstance(cfg.hidden_size, int)
    assert isinstance(cfg.intermediate_size, int)

    # Training
    assert isinstance(cfg.batch_size, int)
    assert isinstance(cfg.grad_accum, int)
    assert isinstance(cfg.learning_rate, float)
    assert isinstance(cfg.epochs, int)
    assert isinstance(cfg.grad_checkpoint, bool)
    assert isinstance(cfg.log_steps, int)
    assert isinstance(cfg.save_steps, int)
    assert isinstance(cfg.eval_steps, int)
    assert isinstance(cfg.save_total_limit, int)

    # Token IDs
    assert isinstance(cfg.pad_token_id, int)
    assert isinstance(cfg.sep_token_id, int)
    assert isinstance(cfg.space_token_id, int)
    assert isinstance(cfg.bos_token_id, int)
    assert isinstance(cfg.eos_token_id, int)
    assert isinstance(cfg.char_offset, int)

    # System Paths
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.tokenized_training_dir, Path)
    assert isinstance(cfg.tokenized_test_dir, Path)
    assert isinstance(cfg.tokenized_val_dir, Path)


def test_config_properties_logic():
    """Validate dynamic properties against dimensional specifications."""
    test_cfg = Config(dims=256)
    assert test_cfg.hidden_size == 256
    assert test_cfg.intermediate_size == 1024

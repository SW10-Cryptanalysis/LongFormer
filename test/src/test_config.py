import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from config import cfg, Config

def test_global_cfg():
    assert isinstance(cfg, Config)
    # 8192 + 30 + 5 = 8227
    assert cfg.vocab_size == 8227


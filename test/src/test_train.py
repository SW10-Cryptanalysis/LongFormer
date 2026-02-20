import sys
import os
import pytest
from unittest.mock import patch, mock_open

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from train import CipherPlainData
from config import cfg

@patch('glob.glob')
def test_cipher_plain_data_init_and_len(mock_glob):
    # Mock finding two json files
    mock_glob.return_value = ['file1.json', 'file2.json']
    
    dataset = CipherPlainData('dummy_dir')
    
    assert len(dataset) == 2
    mock_glob.assert_called_with(os.path.join('dummy_dir', '*.json'))

def test_cipher_plain_data_getitem():
    # Mock glob to find one file
    with patch('glob.glob', return_value=['data/train/file1.json']), \
         patch('builtins.open', mock_open(read_data='{"ciphertext": "1 2 3", "plaintext": "abc"}')), \
         patch('json.load') as mock_json_load:
        
        # Ensure json.load returns the dictionary we expect
        # We need this because json.load might not use read_data directly if patched
        # Actually, if we patch json.load, it overrides the real json.load.
        mock_json_load.return_value = {
            'ciphertext': '1 2 3',
            'plaintext': 'abc'
        }
        
        dataset = CipherPlainData('data/train')
        item = dataset[0]
        
        # Check keys
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        
        # Check shapes (should be padded to max_context)
        assert item['input_ids'].shape[0] == cfg.max_context
        assert item['attention_mask'].shape[0] == cfg.max_context
        assert item['labels'].shape[0] == cfg.max_context
        
        # Check logic: Ciphertext IDs -> Separator -> Plaintext IDs
        input_ids = item['input_ids'].tolist()
        sep_token = cfg.unique_homophones + 1
        
        # 1. Separator exists
        assert sep_token in input_ids
        sep_index = input_ids.index(sep_token)
        
        # 2. Ciphertext comes before separator
        assert input_ids[:sep_index] == [1, 2, 3]
        
        # 3. Plaintext comes after separator
        # We expect 3 characters for 'abc', so 3 IDs
        plain_part = input_ids[sep_index+1 : sep_index+1+3]
        assert len(plain_part) == 3
        # Check that they are offset correctly (value > sep_token)
        assert all(id > sep_token for id in plain_part)

def test_cipher_plain_data_no_files():
    with patch('glob.glob', return_value=[]):
        with pytest.raises(ValueError, match='No .json files found'):
            CipherPlainData('empty_dir')

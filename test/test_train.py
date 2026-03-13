import pytest
import torch
from pathlib import Path
from src.train import PretokenizedCipherDataset, varlen_collate, train


@pytest.fixture
def mock_dataset_data():
    return [
        {"input_ids": [10, 20, 30, 0, 0], "labels": [10, 20, 30, 0, 0]},
        {"input_ids": [40, 50, 0], "labels": [40, 50, 0]},
    ]


def test_pretokenized_cipher_dataset(mocker, mock_dataset_data):
    mocker.patch("src.train.load_from_disk", return_value=mock_dataset_data)

    # Mock the global config padding logic
    mocker.patch("src.train.cfg.pad_token_id", 0)
    mocker.patch("src.train.cfg.max_context", 100)

    ds = PretokenizedCipherDataset(Path("dummy/path"))
    assert len(ds) == 2

    item_0 = ds[0]
    # Verify padding is stripped for varlen efficiency
    assert item_0["input_ids"] == [10, 20, 30]
    assert item_0["labels"] == [10, 20, 30]


def test_varlen_collate():
    batch = [
        {"input_ids": [10, 20, 30], "labels": [11, 21, 31]},
        {"input_ids": [40, 50], "labels": [41, 51]},
    ]

    collated = varlen_collate(batch)

    assert "input_ids" in collated
    assert "labels" in collated
    assert "cu_seqlens" in collated
    assert "pos_ids" in collated
    assert "max_seqlen" in collated

    # Check flattening behavior critical for FlashAttention
    input_ids = torch.as_tensor(collated["input_ids"])
    assert input_ids.shape == (1, 5)
    assert input_ids[0].tolist() == [10, 20, 30, 40, 50]

    # Check sequence boundaries
    cu_seqlens = torch.as_tensor(collated["cu_seqlens"])
    assert cu_seqlens.shape == (1, 3)
    assert cu_seqlens[0].tolist() == [0, 3, 5]

    assert collated["max_seqlen"] == 3


def test_train_execution_with_spaces(mocker):
    mocker.patch("src.train.cfg.use_spaces", True)
    mocker.patch("src.train.PretokenizedCipherDataset")
    mock_get_model = mocker.patch("src.train.get_model")
    mock_trainer = mocker.patch("src.train.Trainer")
    mock_train_args = mocker.patch("src.train.TrainingArguments")
    mocker.patch("src.train.os.path.isdir", return_value=False)

    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer.return_value = mock_trainer_instance

    train()

    mock_get_model.assert_called_once()
    mock_train_args.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_trainer_instance.save_model.assert_called_once()
    args, _ = mock_trainer_instance.save_model.call_args
    save_path = str(args[0])
    assert save_path.endswith("final_model_with_spaces")


def test_train_execution_no_spaces(mocker):
    mocker.patch("src.train.cfg.use_spaces", False)
    mocker.patch("src.train.PretokenizedCipherDataset")
    mock_get_model = mocker.patch("src.train.get_model")
    mock_trainer = mocker.patch("src.train.Trainer")
    mock_train_args = mocker.patch("src.train.TrainingArguments")
    mocker.patch("src.train.os.path.isdir", return_value=False)

    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer.return_value = mock_trainer_instance

    train()

    mock_get_model.assert_called_once()
    mock_train_args.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_trainer_instance.save_model.assert_called_once()
    args, _ = mock_trainer_instance.save_model.call_args
    save_path = str(args[0])
    assert save_path.endswith("final_model_no_spaces")

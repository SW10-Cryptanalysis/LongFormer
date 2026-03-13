import torch
from pathlib import Path
from src.evaluate import _resolve_model_path, _load_model, _generate_tokens, evaluate


def test_resolve_model_path(mocker, tmp_path):
    mocker.patch("src.evaluate.cfg.output_dir", str(tmp_path))

    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "other_dir").mkdir()

    path = _resolve_model_path()
    assert path == tmp_path / "checkpoint-500"


def test_load_model(mocker, tmp_path):
    mocker.patch("src.evaluate.cfg.dims", 64)
    mocker.patch("src.evaluate.cfg.vocab_size", 128)

    mock_model = mocker.Mock()
    mock_model.to.return_value = mock_model
    mock_model.bfloat16.return_value = mock_model
    mock_get_model = mocker.patch("src.evaluate.get_model", return_value=mock_model)

    mocker.patch(
        "src.evaluate.torch.load", return_value={"embed.weight": torch.randn(128, 64)}
    )
    mocker.patch("src.evaluate.torch.cuda.is_available", return_value=True)
    mocker.patch("src.evaluate.torch.cuda.is_bf16_supported", return_value=True)

    _load_model(tmp_path, torch.device("cpu"))

    mock_get_model.assert_called_once()
    mock_model.load_state_dict.assert_called_once()
    mock_model.eval.assert_called_once()


def test_generate_tokens(mocker):
    mock_model = mocker.Mock()
    mock_outputs = {"logits": torch.zeros((1, 1, 128))}
    mock_outputs["logits"][0, -1, 99] = 1.0
    mock_model.return_value = mock_outputs

    device = torch.device("cpu")
    input_ids = [10, 20]

    generated = _generate_tokens(
        model=mock_model,
        input_ids=input_ids,
        chars_to_generate=5,
        eos_token=99,
        char_offset=100,
        device=device,
    )

    assert generated == [99]


class DummyDataset:
    """A minimal stub to satisfy evaluation duck-typing without relying on MagicMock dunders."""

    def __len__(self) -> int:
        return 1

    def select(self, indices: range) -> list[dict[str, str]]:
        return [{"ciphertext": "1 2 3", "plaintext": "abc"}]


def test_evaluate_execution(mocker):
    mocker.patch("src.evaluate._resolve_model_path", return_value=Path("dummy/path"))
    mocker.patch("src.evaluate._load_model")

    mocker.patch("src.evaluate.load_from_disk", return_value=DummyDataset())

    mock_generate = mocker.patch(
        "src.evaluate._generate_tokens", return_value=[100, 101, 102]
    )

    mock_logger = mocker.patch("src.evaluate.logger")

    evaluate()

    mock_generate.assert_called_once()
    mock_logger.info.assert_called_once()

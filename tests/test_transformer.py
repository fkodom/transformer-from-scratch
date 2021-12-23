import pytest
import torch

from src.transformer import Transformer


@pytest.mark.parametrize("num_encoder_layers", [1, 6])
@pytest.mark.parametrize("num_decoder_layers", [1, 6])
@pytest.mark.parametrize("dim_model", [2, 8])
@pytest.mark.parametrize("num_heads", [1, 6])
@pytest.mark.parametrize("dim_feedforward", [2, 8])
def test_init(
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_model: int,
    num_heads: int,
    dim_feedforward: int,
):
    _ = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_model=dim_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("src_len", [2, 8])
@pytest.mark.parametrize("tgt_len", [2, 8])
@pytest.mark.parametrize("num_features", [2, 8])
@pytest.mark.parametrize("num_encoder_layers", [1, 6])
@pytest.mark.parametrize("num_decoder_layers", [1, 6])
@pytest.mark.parametrize("num_heads", [1, 6])
@pytest.mark.parametrize("dim_feedforward", [2, 8])
def test_forward(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    num_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    num_heads: int,
    dim_feedforward: int,
):
    model = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_model=num_features,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
    )

    src = torch.randn(batch_size, src_len, num_features)
    tgt = torch.randn(batch_size, tgt_len, num_features)
    out = model(src, tgt)

    _batch_size, seq_len, _num_features = out.shape
    assert batch_size == _batch_size
    assert seq_len == tgt_len
    assert _num_features == num_features

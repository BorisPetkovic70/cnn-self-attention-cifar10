"""CPU-fast shape and gradient smoke tests."""

import pytest
import torch

from src.models.attention import SpatialSelfAttention

# (channels, H = W) of every backbone stage output, see docs/INTERFACES.md
STAGE_SHAPES = [(64, 32), (64, 16), (128, 8), (256, 4), (512, 4)]


@pytest.mark.parametrize("channels, size", STAGE_SHAPES)
def test_attention_keeps_shape(channels: int, size: int) -> None:
    block = SpatialSelfAttention(channels)
    x = torch.randn(2, channels, size, size)
    assert block(x).shape == x.shape


def test_attention_starts_as_identity() -> None:
    block = SpatialSelfAttention(64)
    x = torch.randn(2, 64, 16, 16)
    assert torch.equal(block(x), x)


def test_attention_map_rows_sum_to_one() -> None:
    block = SpatialSelfAttention(64)
    x = torch.randn(2, 64, 16, 16)
    block(x)
    assert block.last_attn is None  # store_attn is off by default

    block.store_attn = True
    block(x)
    attn = block.last_attn
    assert attn.shape == (2, 256, 256)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 256))


def test_attention_gamma_gets_gradient() -> None:
    block = SpatialSelfAttention(64)
    block(torch.randn(2, 64, 16, 16)).sum().backward()
    assert block.gamma.grad is not None
    assert block.gamma.grad.abs().item() > 0

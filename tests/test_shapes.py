"""CPU-fast shape and gradient smoke tests."""

import pytest
import torch

from src.config import ExperimentConfig
from src.models.attention import SpatialSelfAttention
from src.models.baseline import BaselineVGG
from src.models.factory import build_model
from src.models.hybrid import AttentionVGG

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


def test_baseline_output_shape() -> None:
    assert BaselineVGG()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


@pytest.mark.parametrize(
    "attn_pos", ["none", "after_stem", "after_stage1", "after_stage4", "after_stage1,after_stage3"]
)
def test_hybrid_output_shape(attn_pos: str) -> None:
    assert AttentionVGG(attn_pos=attn_pos)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_hybrid_attn_keyed_by_stage() -> None:
    assert list(AttentionVGG(attn_pos="after_stage1, after_stage3").attn) == ["stage1", "stage3"]
    assert len(AttentionVGG(attn_pos="none").attn) == 0


@pytest.mark.parametrize("attn_pos", ["after_stage9", "stage1", "after_stage1,bogus", ""])
def test_hybrid_rejects_unknown_attn_pos(attn_pos: str) -> None:
    with pytest.raises(ValueError):
        AttentionVGG(attn_pos=attn_pos)


def test_hybrid_matches_baseline_at_init() -> None:
    # Same backbone and head weights + gamma = 0 -> the hybrid computes the baseline's function
    baseline = BaselineVGG().eval()
    hybrid = AttentionVGG(attn_pos="after_stage1,after_stage3").eval()
    missing, unexpected = hybrid.load_state_dict(baseline.state_dict(), strict=False)
    assert unexpected == []
    assert all(key.startswith("attn.") for key in missing)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(hybrid(x), baseline(x))


@pytest.mark.parametrize("model, cls", [("baseline", BaselineVGG), ("hybrid", AttentionVGG)])
def test_build_model_returns_registered_class(model: str, cls: type) -> None:
    assert isinstance(build_model(ExperimentConfig(model=model)), cls)


def test_build_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        build_model(ExperimentConfig(model="resnet"))

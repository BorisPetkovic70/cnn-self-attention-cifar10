"""VGG backbone with self-attention blocks inserted after selected stages."""

from __future__ import annotations

from torch import Tensor, nn

from src.models.attention import SpatialSelfAttention
from src.models.blocks import STAGE_CHANNELS, STAGE_ORDER, build_backbone, build_head


def parse_attn_pos(attn_pos: str) -> list[str]:
    """Turn ``"after_stage1,after_stage3"`` into ``["stage1", "stage3"]``; ``"none"`` gives ``[]``.

    Raises:
        ValueError: If an entry is not ``after_<stage>`` with a stage from ``STAGE_ORDER``.
    """
    if attn_pos == "none":
        return []
    stages = []
    for entry in attn_pos.split(","):
        entry = entry.strip()
        stage = entry.removeprefix("after_")
        if not entry.startswith("after_") or stage not in STAGE_ORDER:
            raise ValueError(
                f"Unknown attn_pos entry {entry!r}, expected after_<stage> with stage in {STAGE_ORDER}"
            )
        stages.append(stage)
    return stages


class AttentionVGG(nn.Module):
    """The baseline VGG with ``SpatialSelfAttention`` after the stages named in ``attn_pos``.

    Args:
        num_classes: Number of output classes.
        attn_pos: ``"none"`` or comma-separated ``after_<stage>`` values,
            e.g. ``"after_stage1,after_stage3"``. An unknown stage raises ``ValueError``.
        reduction: Q/K channel reduction of each attention block.
        dropout: Dropout probability in the classification head.

    Attributes:
        stages: Backbone from ``build_backbone()``, identical to ``BaselineVGG``.
        attn: Stage name (e.g. ``"stage1"``) -> ``SpatialSelfAttention``.
        head: Classification head from ``build_head(num_classes, dropout)``.
    """

    def __init__(
        self,
        num_classes: int = 10,
        attn_pos: str = "after_stage1",
        reduction: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stages = build_backbone()
        # ModuleDict, not dict: registers the blocks so their parameters are trained, moved and saved
        self.attn = nn.ModuleDict({
            name: SpatialSelfAttention(STAGE_CHANNELS[name], reduction)
            for name in parse_attn_pos(attn_pos)
        })
        self.head = build_head(num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Classify images: (B, 3, 32, 32) -> logits (B, num_classes)."""
        for name in STAGE_ORDER:
            x = self.stages[name](x)
            if name in self.attn:
                x = self.attn[name](x)
        return self.head(x)

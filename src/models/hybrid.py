"""VGG backbone with self-attention blocks inserted after selected stages."""

from __future__ import annotations

from torch import Tensor, nn


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
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Classify images: (B, 3, 32, 32) -> logits (B, num_classes)."""
        raise NotImplementedError

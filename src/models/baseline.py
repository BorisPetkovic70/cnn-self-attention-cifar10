"""Plain VGG baseline: the shared backbone and head with no attention."""

from torch import Tensor, nn

from src.models.blocks import STAGE_ORDER, build_backbone, build_head


class BaselineVGG(nn.Module):
    """VGG-style CNN built from ``build_backbone()`` and ``build_head()``.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability in the classification head.

    Attributes:
        stages: Backbone stages keyed by ``STAGE_ORDER``.
        head: Global average pooling -> Dropout -> Linear classifier.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.stages = build_backbone()
        self.head = build_head(num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Classify images: (B, 3, 32, 32) -> logits (B, num_classes)."""
        for name in STAGE_ORDER:
            x = self.stages[name](x)
        return self.head(x)

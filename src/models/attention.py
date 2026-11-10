"""SAGAN-style spatial self-attention block."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialSelfAttention(nn.Module):
    """Single-head spatial self-attention over the H*W positions of a feature map.

    Q and K are 1x1 convolutions to ``in_channels // reduction`` channels, V keeps
    ``in_channels``. The output is ``x + gamma * attention(x)``; ``gamma`` starts at 0,
    so the block starts as the identity.

    Attributes:
        gamma: Learnable scalar residual weight, initialized to 0.
        store_attn: When True, ``forward`` caches the attention matrix in ``last_attn``.
        last_attn: Last attention matrix, shape (B, N, N) with N = H*W, rows sum to 1.
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.store_attn: bool = False
        self.last_attn: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention: (B, C, H, W) -> (B, C, H, W)."""
        raise NotImplementedError

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
        inner = in_channels // reduction
        self.query = nn.Conv2d(in_channels, inner, kernel_size=1)
        self.key = nn.Conv2d(in_channels, inner, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.store_attn: bool = False
        self.last_attn: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention: (B, C, H, W) -> (B, C, H, W)."""
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)  # (B, N, C')  N = H*W, C' = C // reduction
        k = self.key(x).flatten(2)                    # (B, C', N)
        v = self.value(x).flatten(2)                  # (B, C, N)

        energy = torch.bmm(q, k)                      # (B, N, N)  energy[i, j] = q_i · k_j
        attn = torch.softmax(energy, dim=-1)          # (B, N, N)  softmax over keys j: rows sum to 1
        if self.store_attn:
            self.last_attn = attn.detach()            # kept only for plotting, outside the graph

        out = torch.bmm(v, attn.transpose(1, 2))      # (B, C, N)  out_i = sum_j attn[i, j] * v_j
        return x + self.gamma * out.view(b, c, h, w)

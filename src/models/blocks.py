"""Shared VGG spine: the backbone stages and classification head used by every model."""

from torch import nn

STAGE_ORDER: tuple[str, ...] = ("stem", "stage1", "stage2", "stage3", "stage4")

# Output channels per stage; each stage's input is the previous stage's output
STAGE_CHANNELS: dict[str, int] = {
    "stem": 64,
    "stage1": 64,
    "stage2": 128,
    "stage3": 256,
    "stage4": 512,
}


def vgg_block(in_channels: int, out_channels: int, n_convs: int, pool: bool) -> nn.Sequential:
    """Stack ``n_convs`` x (3x3 conv -> BN -> ReLU), optionally followed by 2x2 max pooling.

    The 3x3 convs use ``padding=1`` and keep H x W; only the pooling halves it.
    Conv bias is off because the following BatchNorm adds its own shift.
    """
    layers: list[nn.Module] = []
    for i in range(n_convs):
        layers += [
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def build_backbone() -> nn.ModuleDict:
    """Build the five VGG stages, keyed in ``STAGE_ORDER``, for (B, 3, 32, 32) inputs."""
    c = STAGE_CHANNELS
    return nn.ModuleDict({
        "stem": vgg_block(3, c["stem"], n_convs=1, pool=False),                # (B, 64, 32, 32)
        "stage1": vgg_block(c["stem"], c["stage1"], n_convs=1, pool=True),     # (B, 64, 16, 16)
        "stage2": vgg_block(c["stage1"], c["stage2"], n_convs=2, pool=True),   # (B, 128, 8, 8)
        "stage3": vgg_block(c["stage2"], c["stage3"], n_convs=2, pool=True),   # (B, 256, 4, 4)
        "stage4": vgg_block(c["stage3"], c["stage4"], n_convs=2, pool=False),  # (B, 512, 4, 4)
    })


def build_head(num_classes: int = 10, dropout: float = 0.3) -> nn.Sequential:
    """Global average pooling -> Dropout -> Linear(512, num_classes)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # (B, 512, 1, 1)
        nn.Flatten(),             # (B, 512)
        nn.Dropout(dropout),
        nn.Linear(STAGE_CHANNELS["stage4"], num_classes),
    )

import torch
import torch.nn as nn

from .lfsa import LFSA
from .coordinate_attention import CoordAttention


class CLNet(nn.Module):
    def __init__(
        self,
        channels: int,
        bias: bool = False,
        patch_size: int = 8,
        ca_reduction: int = 32,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        hidden_channels = max(channels * 2, 8)

        self.ca = CoordAttention(inp=channels, reduction=ca_reduction)
        self.lfsa = LFSA(dim=channels, bias=bias, patch_size=patch_size)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
        )

        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))
        self.output_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca_features = self.ca(x)
        lfsa_features = self.lfsa(x)

        fused_features = self.fusion(
            torch.cat([ca_features, lfsa_features], dim=1)
        )

        out = x + self.residual_scale * fused_features
        return self.output_act(out)

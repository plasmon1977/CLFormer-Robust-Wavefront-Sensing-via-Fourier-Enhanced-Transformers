import torch
import torch.nn as nn


class ConvAblationBackbone(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        stage1 = max(32, out_dim // 8)
        stage2 = max(64, out_dim // 4)
        stage3 = max(128, out_dim // 2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, stage1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage1),
            nn.GELU(),

            nn.Conv2d(stage1, stage2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage2),
            nn.GELU(),

            nn.Conv2d(stage2, stage3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage3),
            nn.GELU(),

            nn.Conv2d(stage3, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

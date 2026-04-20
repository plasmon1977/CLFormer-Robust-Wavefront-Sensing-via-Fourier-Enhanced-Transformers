import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.vit import VARIANT_TO_MODEL, create_timm_vit
from layers import (
    CBAMBlock,
    CLNet,
    ConvAblationBackbone,
)


FEATURE_DIM_BY_VARIANT = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}


class CLFormerPSFModel(nn.Module):
    def __init__(
        self,
        input_size: int = 112,
        patch_size: int = 16,
        num_images: int = 4,
        channels_per_image: int = 1,
        num_coefficients: int = 25,
        variant: str = "small",
        pretrained: bool = True,
        dropout: float = 0.1,
        use_vit_module: bool = True,
        lfsa_patch_size: int = 8,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        ca_reduction: int = 32,
    ):
        super().__init__()

        if variant not in VARIANT_TO_MODEL:
            raise ValueError(f"Unsupported ViT variant: {variant}")

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_images = num_images
        self.channels_per_image = channels_per_image
        self.num_coefficients = num_coefficients
        self.total_channels = num_images * channels_per_image
        self.variant = variant
        self.use_vit_module = use_vit_module
        self.lfsa_patch_size = lfsa_patch_size

        self.stem_norm = nn.BatchNorm2d(self.total_channels)

        self.clnet_module = CLNet(
            channels=self.total_channels,
            bias=False,
            patch_size=lfsa_patch_size,
            ca_reduction=ca_reduction,
        )

        self.cbam_module = CBAMBlock(
            channels=self.total_channels,
            reduction=cbam_reduction,
            spatial_kernel_size=cbam_kernel_size,
        )

        if use_vit_module:
            model_name = VARIANT_TO_MODEL[variant]
            self.backbone, self.using_pretrained = create_timm_vit(
                model_name=model_name,
                total_channels=self.total_channels,
                pretrained=pretrained,
            )
            with torch.no_grad():
                dummy_input = torch.randn(
                    1, self.total_channels, input_size, input_size
                )
                feature_dim = int(self.backbone(dummy_input).shape[1])
            self.conv_backbone = None
        else:
            self.using_pretrained = False
            self.backbone = None
            feature_dim = FEATURE_DIM_BY_VARIANT[variant]
            self.conv_backbone = ConvAblationBackbone(
                in_channels=self.total_channels,
                out_dim=feature_dim,
            )

        hidden_dim = max(feature_dim // 2, num_coefficients * 2)
        bottleneck_dim = max(feature_dim // 4, num_coefficients)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_coefficients),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels, got {x.shape[1]}"
            )

        x = self.stem_norm(x)
        x = self.clnet_module(x)
        x = self.cbam_module(x)

        if self.use_vit_module:
            return self.backbone(x)
        return self.conv_backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(x)
        return self.head(features)


def create_model(
    input_size: int = 112,
    patch_size: int = 16,
    num_images: int = 4,
    channels_per_image: int = 1,
    num_coefficients: int = 25,
    variant: str = "small",
    pretrained: bool = True,
    dropout: float = 0.1,
    use_vit_module: bool = True,
    lfsa_patch_size: int = 8,
    cbam_reduction: int = 16,
    cbam_kernel_size: int = 7,
    ca_reduction: int = 32,
) -> CLFormerPSFModel:
    return CLFormerPSFModel(
        input_size=input_size,
        patch_size=patch_size,
        num_images=num_images,
        channels_per_image=channels_per_image,
        num_coefficients=num_coefficients,
        variant=variant,
        pretrained=pretrained,
        dropout=dropout,
        use_vit_module=use_vit_module,
        lfsa_patch_size=lfsa_patch_size,
        cbam_reduction=cbam_reduction,
        cbam_kernel_size=cbam_kernel_size,
        ca_reduction=ca_reduction,
    )


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "size_mb": total * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("CLFormer PSF-to-Zernike model smoke test")
    print("=" * 80)

    ablation_settings = [
        {"use_vit_module": True,  "variant": "small", "num_images": 5, "num_coefficients": 77},
        {"use_vit_module": True,  "variant": "base",  "num_images": 5, "num_coefficients": 77},
        {"use_vit_module": True,  "variant": "tiny",  "num_images": 4, "num_coefficients": 25},
        {"use_vit_module": False, "variant": "small", "num_images": 5, "num_coefficients": 77},
    ]

    for settings in ablation_settings:
        print("-" * 80)
        print(f"Settings: {settings}")
        print("-" * 80)

        model = create_model(
            input_size=112,
            patch_size=8,
            pretrained=False,
            dropout=0.1,
            **settings,
        )
        params = count_parameters(model)

        x = torch.randn(2, settings["num_images"], 112, 112)
        with torch.no_grad():
            y = model(x)

        print(f"Total params:       {params['total']:,}")
        print(f"Trainable params:   {params['trainable']:,}")
        print(f"Model size:         {params['size_mb']:.2f} MB")
        print(f"Input shape:        {tuple(x.shape)}")
        print(f"Output shape:       {tuple(y.shape)}")

    print("=" * 80)
    print("Smoke test completed")
    print("=" * 80)

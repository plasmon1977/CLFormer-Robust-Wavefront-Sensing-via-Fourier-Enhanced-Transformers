"""
Vision Transformer model for PSF-to-Zernike regression.
"""

import warnings

import timm
import torch
import torch.nn as nn


VARIANT_TO_MODEL = {
    "tiny": "vit_tiny_patch16_224",
    "small": "vit_small_patch16_224",
    "base": "vit_base_patch16_224",
    "large": "vit_large_patch16_224",
}


def create_timm_vit(model_name: str, total_channels: int, pretrained: bool):
    common_kwargs = {
        "img_size": 112,
        "in_chans": total_channels,
        "num_classes": 0,
        "global_pool": "avg",
    }

    if not pretrained:
        return timm.create_model(model_name, pretrained=False, **common_kwargs), False

    try:
        backbone = timm.create_model(model_name, pretrained=True, **common_kwargs)
        return backbone, True
    except Exception as exc:
        warnings.warn(
            f"Failed to load pretrained weights for {model_name}: {exc}. "
            "Falling back to random initialization."
        )
        backbone = timm.create_model(model_name, pretrained=False, **common_kwargs)
        return backbone, False


class ViTPSFModel(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_images: int = 4,
        channels_per_image: int = 1,
        num_coefficients: int = 25,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_images = num_images
        self.channels_per_image = channels_per_image
        self.num_coefficients = num_coefficients
        self.total_channels = num_images * channels_per_image

        self.backbone, self.using_pretrained = create_timm_vit(
            model_name=model_name,
            total_channels=self.total_channels,
            pretrained=pretrained,
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, self.total_channels, 112, 112)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def create_vit_model(
    variant: str = "small",
    num_images: int = 4,
    channels_per_image: int = 1,
    num_coefficients: int = 25,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> ViTPSFModel:
    if variant not in VARIANT_TO_MODEL:
        raise ValueError(f"Unsupported ViT variant: {variant}")

    return ViTPSFModel(
        model_name=VARIANT_TO_MODEL[variant],
        num_images=num_images,
        channels_per_image=channels_per_image,
        num_coefficients=num_coefficients,
        pretrained=pretrained,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "size_mb": total * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("ViT PSF-to-Zernike smoke test")
    print("=" * 80)

    for variant in ["tiny", "small", "base"]:
        print(f"\n{'=' * 60}")
        print(f"Testing variant: ViT-{variant}")
        print(f"{'=' * 60}")

        try:
            model = create_vit_model(
                variant=variant,
                num_images=5,
                channels_per_image=1,
                num_coefficients=77,
                pretrained=False,
                dropout=0.1,
            )

            params = count_parameters(model)
            print(f"Total params: {params['total']:,}")
            print(f"Trainable params: {params['trainable']:,}")
            print(f"Model size: {params['size_mb']:.2f} MB")

            x = torch.randn(2, 5, 112, 112)
            with torch.no_grad():
                y = model(x)

            print(f"Input shape: {x.shape}")
            print(f"Output shape: {y.shape}")
            print(f"Output stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        except Exception as exc:
            print(f"Variant {variant} smoke test failed: {exc}")

    print(f"\n{'=' * 80}")
    print("Smoke test completed")
    print(f"{'=' * 80}")

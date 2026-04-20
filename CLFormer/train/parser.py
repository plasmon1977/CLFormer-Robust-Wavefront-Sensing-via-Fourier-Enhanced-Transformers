import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CLFormer Model",
    )

    parser.add_argument(
        "--train_dir", type=str, required=True,
        help="Training data directory (contains .npy files)",
    )
    parser.add_argument(
        "--val_dir", type=str, required=True,
        help="Validation data directory",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Test data directory",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="./checkpoints_fcas_vit",
        help="Directory to save model checkpoints",
    )

    parser.add_argument(
        "--input_size", type=int, default=112,
        help="Input image size",
    )
    parser.add_argument(
        "--patch_size", type=int, default=8,
        help="ViT patch size",
    )
    parser.add_argument(
        "--fsas_patch_size", type=int, default=8,
        help="LFSA internal patch size",
    )
    parser.add_argument(
        "--cbam_reduction", type=int, default=16,
        help="CBAM channel attention reduction ratio",
    )
    parser.add_argument(
        "--cbam_kernel_size", type=int, default=7,
        help="CBAM spatial attention convolution kernel size",
    )
    parser.add_argument(
        "--ca_reduction", type=int, default=32,
        help="CA (Coordinate Attention) channel reduction ratio",
    )
    parser.add_argument(
        "--num_images", type=int, default=4, choices=[4, 5],
        help="Number of input PSF images (4 -> 25 coefficients, 5 -> 77 coefficients)",
    )
    parser.add_argument(
        "--num_coefficients", type=int, default=25, choices=[25, 77],
        help="Number of output Zernike coefficients",
    )
    parser.add_argument(
        "--variant", type=str, default="small",
        choices=["tiny", "small", "base", "large"],
        help="ViT backbone variant",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--use_vit_module", action="store_true", default=True,
        help="Use ViT backbone (disable to use pure convolution backbone)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--use_amp", action="store_true", default=True,
        help="Enable AMP mixed precision training",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="EarlyStopping patience",
    )

    return parser

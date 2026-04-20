import math
import os
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

import sys
from pathlib import Path

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.clformer import create_model, count_parameters
from train.callbacks import EarlyStopping
from train.dataset import PSFDataset
from train.engine import evaluate, train_one_epoch
from train.loss import MaskedMSELoss
from train.utils import save_training_artifacts, setup_logger


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    warmup_epochs = epochs // 10

    def lr_fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = epoch - warmup_epochs
        total = epochs - warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * progress / total))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.checkpoint_dir, f"train_{timestamp}.log")
    logger = setup_logger(log_path)

    logger.info("=" * 80)
    logger.info("CLFormer Model Training")
    logger.info(
        f"Mode: {args.num_images} PSF images -> "
        f"{args.num_coefficients} Zernike coefficients"
    )
    if args.num_coefficients == 77:
        logger.info("First 3 entries in row 7 are 0, excluded from loss computation (74 valid coefficients)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Train dir: {args.train_dir}")
    logger.info(f"Val dir: {args.val_dir}")
    logger.info(f"Test dir: {args.test_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"LR: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"LFSA patch size: {args.fsas_patch_size}")
    logger.info(f"CBAM reduction: {args.cbam_reduction}")
    logger.info(f"CBAM kernel size: {args.cbam_kernel_size}")
    logger.info(f"Use ViT: {args.use_vit_module}")
    logger.info(f"Use AMP: {args.use_amp}")
    logger.info("=" * 80)

    train_dataset = PSFDataset(
        args.train_dir, args.num_images, args.num_coefficients,
    )
    val_dataset = PSFDataset(
        args.val_dir, args.num_images, args.num_coefficients,
    )
    test_dataset = PSFDataset(
        args.test_dir, args.num_images, args.num_coefficients,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    logger.info(
        f"Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    model = create_model(
        input_size=args.input_size,
        patch_size=args.patch_size,
        num_images=args.num_images,
        channels_per_image=1,
        num_coefficients=args.num_coefficients,
        variant=args.variant,
        pretrained=False,
        dropout=args.dropout,
        use_vit_module=args.use_vit_module,
        lfsa_patch_size=args.fsas_patch_size,
        cbam_reduction=args.cbam_reduction,
        cbam_kernel_size=args.cbam_kernel_size,
        ca_reduction=args.ca_reduction,
    )

    params = count_parameters(model)
    logger.info(
        f"Model parameters: {params['total']:,} ({params['size_mb']:.2f} MB)"
    )

    model = model.to(device)

    # Only use DataParallel if we have enough data for multiple batches
    min_samples_for_multigpu = args.batch_size * torch.cuda.device_count()
    if torch.cuda.device_count() > 1 and len(train_dataset) >= min_samples_for_multigpu:
        logger.info(
            f"Using DataParallel with {torch.cuda.device_count()} GPUs"
        )
        torch.cuda.empty_cache()
        model = nn.DataParallel(model)
        logger.info(
            f"Effective batch size per GPU: "
            f"{args.batch_size // torch.cuda.device_count()}"
        )
    else:
        if torch.cuda.device_count() > 1:
            logger.warning(
                f"Dataset too small ({len(train_dataset)} samples, need >={min_samples_for_multigpu}) "
                f"for multi-GPU training with batch_size={args.batch_size}. Using single GPU."
            )

    if args.num_coefficients == 77:
        criterion = MaskedMSELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scaler = GradScaler() if args.use_amp else None

    best_val = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "model_best.pth")
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler, args.use_amp, args.max_grad_norm,
        )
        val_loss = evaluate(
            model, val_loader, criterion, device, args.use_amp,
        )
        scheduler.step()
        early_stopping(val_loss, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        if epoch % 10 == 0:
            save_training_artifacts(history, args.checkpoint_dir)
            epoch_path = os.path.join(
                args.checkpoint_dir, f"model_epoch_{epoch:03d}.pth",
            )
            torch.save(model.state_dict(), epoch_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("  -> Best model saved")

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))

    test_loss = evaluate(
        model, test_loader, criterion, device, args.use_amp,
    )
    logger.info(f"Test RMSE: {test_loss:.4f}")

    final_path = os.path.join(args.checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    save_training_artifacts(history, args.checkpoint_dir)
    logger.info("Training completed!")

from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0

    try:
        from torch.amp import autocast
    except ImportError:
        from torch.cuda.amp import autocast

    progress = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}",
        ncols=120,
    )

    for batch_idx, (inputs, labels) in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        try:
            if use_amp and scaler is not None:
                with autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                ):
                    outputs = model(inputs)
                    mse = criterion(outputs, labels)
                    loss = torch.sqrt(mse + 1e-8)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                mse = criterion(outputs, labels)
                loss = torch.sqrt(mse + 1e-8)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm,
                )
                optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    f"\nWarning: Batch {batch_idx} out of GPU memory, skipping this batch"
                )
                torch.cuda.empty_cache()
                continue
            raise e

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.eval()
    running_rmse = 0.0

    try:
        from torch.amp import autocast
    except ImportError:
        from torch.cuda.amp import autocast

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
            ):
                outputs = model(inputs)
                loss_mse = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss_mse = criterion(outputs, labels)

        loss_rmse = torch.sqrt(loss_mse + 1e-8)
        running_rmse += loss_rmse.item()

    return running_rmse / len(dataloader)

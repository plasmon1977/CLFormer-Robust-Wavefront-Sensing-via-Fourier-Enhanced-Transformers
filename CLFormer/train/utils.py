import logging
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("FCASViT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_training_artifacts(
    history: List[Dict],
    checkpoint_dir: str,
) -> None:
    if not history:
        return

    df = pd.DataFrame(history)
    excel_path = os.path.join(checkpoint_dir, "train_history.xlsx")
    df.to_excel(excel_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df["epoch"], df["train_loss"],
        label="Train Loss", marker="o", markersize=3, linewidth=1.2,
    )
    plt.plot(
        df["epoch"], df["val_loss"],
        label="Val Loss", marker="s", markersize=3, linewidth=1.2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (RMSE)")
    plt.title("Training vs Validation Loss (CLFormer)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "train_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

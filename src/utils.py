"""Shared training utilities: reproducibility seeding, device selection,
and checkpointing. Model-agnostic — the same functions work for A1, A2,
and A3, since a checkpoint just stores a state dict plus enough metadata
(model_config) to reconstruct whichever model class produced it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix every source of randomness this project touches (Python,
    NumPy, PyTorch CPU/GPU). Call this once, before building the model,
    optimizer, or dataloaders — reproducibility is a course requirement,
    not just good practice."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, float],
    tokenizer_char_to_idx: dict[str, int],
    model_config: dict[str, Any],
) -> None:
    """Save everything needed to resume training or reproduce evaluation
    from this checkpoint alone: weights, optimizer state, epoch, the
    validation metrics that earned this checkpoint, the exact vocabulary
    used, and the model's constructor config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "tokenizer_char_to_idx": tokenizer_char_to_idx,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint saved by save_checkpoint(). `model` must already
    be constructed with the same architecture (use the returned
    `model_config` / `tokenizer_char_to_idx` to reconstruct it correctly
    if loading "cold", without already knowing those values)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

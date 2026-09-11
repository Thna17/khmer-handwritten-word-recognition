"""Model-agnostic training loop for CTC-based Khmer word recognition.

Works with any model implementing:
    forward(images: Tensor[B,1,H,W]) -> log_probs: Tensor[T,B,num_classes]
    compute_sequence_length(width: int) -> int

A1 (BaselineCRNN) implements this today; A2 and A3 will too, so `fit()`
does not change when those approaches are added — only the model object
(and its `model_config` dict, used for checkpointing) passed in differs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.decoder import greedy_decode
from src.metrics import evaluate_predictions
from src.tokenizer import KhmerTokenizer
from src.utils import get_device, save_checkpoint


@dataclasses.dataclass
class TrainConfig:
    learning_rate: float = 1e-3          # locked baseline default (PROJECT_CONTEXT.md)
    max_epochs: int = 100
    patience: int = 10                   # early stopping: epochs without val CER improvement
    checkpoint_dir: str = "checkpoints"
    checkpoint_prefix: str = "a1_baseline"


def _compute_input_lengths(model: nn.Module, image_widths: torch.Tensor) -> torch.Tensor:
    """Per-sample true (pre-padding) CTC input lengths — see
    model_baseline.py's compute_sequence_length docstring for why the
    padded batch width must never be used here directly."""
    return torch.tensor(
        [model.compute_sequence_length(w.item()) for w in image_widths],
        dtype=torch.long,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths = _compute_input_lengths(model, batch["image_widths"]).to(device)

        optimizer.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, tokenizer: KhmerTokenizer, criterion, device) -> dict:
    """Runs one full pass over `loader`, returning {loss, cer, wer, word_accuracy}."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[str] = []
    all_predictions: list[str] = []

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths = _compute_input_lengths(model, batch["image_widths"]).to(device)

        log_probs = model(images)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = greedy_decode(log_probs, tokenizer, input_lengths=input_lengths)
        all_predictions.extend(predictions)
        all_targets.extend(batch["labels"])

    metrics = evaluate_predictions(all_targets, all_predictions)
    metrics["loss"] = total_loss / total_samples
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: KhmerTokenizer,
    model_config: dict,
    config: TrainConfig = TrainConfig(),
    device: torch.device | None = None,
) -> list[dict]:
    """Train `model` with AdamW and early stopping on validation CER.

    Saves `{checkpoint_prefix}_latest.pt` every epoch and
    `{checkpoint_prefix}_best.pt` whenever validation CER improves, so
    training can survive a Colab disconnect without losing progress.
    Returns the per-epoch history (for plotting loss/metric curves).
    """
    device = device or get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    history: list[dict] = []
    best_cer = float("inf")
    epochs_without_improvement = 0

    checkpoint_dir = Path(config.checkpoint_dir)
    latest_path = checkpoint_dir / f"{config.checkpoint_prefix}_latest.pt"
    best_path = checkpoint_dir / f"{config.checkpoint_prefix}_best.pt"

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, tokenizer, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(
            f"epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"| val_CER={val_metrics['cer']*100:.2f}% | val_WordAcc={val_metrics['word_accuracy']*100:.2f}%"
        )

        save_checkpoint(latest_path, model, optimizer, epoch, val_metrics, tokenizer.char_to_idx, model_config)

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, tokenizer.char_to_idx, model_config)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch}: no val CER improvement for {config.patience} epochs.")
                break

    return history

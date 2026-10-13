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
import math
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

    # --- fine-tuning controls -------------------------------------------------
    # All default to the previous behaviour, so an existing command line that
    # does not set them trains exactly as it did before.
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    scheduler: str = "plateau"           # "plateau" (as before) or "cosine"
    warmup_epochs: int = 0               # linear LR ramp, cosine schedule only
    min_lr_factor: float = 0.02          # cosine floor, as a fraction of learning_rate
    backbone_lr_mult: float = 1.0        # <1 fine-tunes the CNN slower than the head
    freeze_backbone_epochs: int = 0      # train head only for this many epochs first
    ema_decay: float = 0.0               # 0 disables; 0.999 is a good small-data value


class WeightEMA:
    """Exponential moving average of the model weights.

    Averaging the last few thousand steps is worth more here than in a
    normal run: with 177 training lines an epoch is ~11 optimizer steps, so
    individual steps are noisy and the "best epoch" chosen by validation CER
    is partly luck. The EMA weights are what get evaluated and checkpointed.

    Integer buffers (BatchNorm's `num_batches_tracked`) and any other
    non-floating-point state are copied rather than averaged — averaging a
    counter is meaningless and `lerp_` rejects integer tensors anyway.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.steps = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _effective_decay(self) -> float:
        """Ramp the decay up from 0, so the average is not anchored to the
        starting weights.

        The shadow copy starts as a clone of the model at step 0. At a fixed
        decay of 0.99 that initial copy still carries 0.99**steps of the
        weight, which over a 3-epoch 156-step fine-tune leaves 21% of the
        PRE-TRAINING weights (or, from scratch, the random init) inside the
        checkpoint being validated — measured, not hypothetical: a smoke run
        reported val_CER 100% while train_loss fell normally. This is the
        standard `(1 + t) / (10 + t)` warmup: it equals the requested decay
        once enough steps have accumulated to justify it.
        """
        return min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self._effective_decay()
        self.steps += 1
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value.detach())

    def swapped_into(self, model: nn.Module) -> "_EMASwap":
        return _EMASwap(model, self.shadow)


class _EMASwap:
    """Context manager that evaluates `model` with the EMA weights in place
    and restores the live training weights on exit, even if evaluation
    raises — otherwise a mid-evaluation error would silently leave the
    averaged weights in the model and corrupt the rest of training."""

    def __init__(self, model: nn.Module, shadow: dict):
        self.model = model
        self.shadow = shadow
        self.backup: dict | None = None

    def __enter__(self) -> nn.Module:
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)
        return self.model

    def __exit__(self, *exc_info) -> None:
        if self.backup is not None:
            self.model.load_state_dict(self.backup)
            self.backup = None


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze the CNN backbone (`model.cnn` on A1, A2 and A3).

    Freezing means BOTH no gradients AND eval-mode BatchNorm: a backbone
    whose weights are fixed but whose running statistics keep drifting
    toward the current batches is not actually frozen, and on 177 images
    those statistics move a long way.
    """
    backbone = getattr(model, "cnn", None)
    if backbone is None:
        return

    # A2 built with --freeze-backbone is a deliberate linear probe: its
    # backbone must stay frozen for the WHOLE run. Without this guard, the
    # staged unfreeze would quietly turn that experiment into a full
    # fine-tune at epoch `freeze_backbone_epochs + 1` and the results would
    # be filed under the wrong name.
    if getattr(backbone, "frozen", False):
        backbone.eval()
        return

    for parameter in backbone.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        backbone.eval()


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the backbone in its own parameter group.

    A pretrained backbone carries most of what synthetic pre-training
    learned about Khmer glyph shapes; the classifier has to adapt to a new
    ink domain. Giving the backbone a smaller LR keeps the first group from
    being washed out while the second one moves.
    """
    backbone = getattr(model, "cnn", None)
    backbone_parameters = list(backbone.parameters()) if backbone is not None else []
    backbone_ids = {id(p) for p in backbone_parameters}
    head_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]

    groups = [{"params": head_parameters, "lr": config.learning_rate, "name": "head"}]
    if backbone_parameters:
        groups.append({
            "params": backbone_parameters,
            "lr": config.learning_rate * config.backbone_lr_mult,
            "name": "backbone",
        })
    return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    """Returns (scheduler, steps_on_metric). `steps_on_metric` tells `fit`
    whether to call `scheduler.step(val_cer)` or plain `scheduler.step()`."""
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5), True

    if config.scheduler != "cosine":
        raise ValueError(f"unknown scheduler {config.scheduler!r}; expected 'plateau' or 'cosine'")

    warmup = max(0, config.warmup_epochs)
    total = max(1, config.max_epochs)
    floor = config.min_lr_factor

    def lr_factor(epoch_index: int) -> float:
        if warmup and epoch_index < warmup:
            return (epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor), False


def _compute_input_lengths(model: nn.Module, image_widths: torch.Tensor) -> torch.Tensor:
    """Per-sample true (pre-padding) CTC input lengths — see
    model_baseline.py's compute_sequence_length docstring for why the
    padded batch width must never be used here directly."""
    return torch.tensor(
        [model.compute_sequence_length(w.item()) for w in image_widths],
        dtype=torch.long,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device,
    grad_clip: float = 5.0,
    freeze_backbone: bool = False,
    ema: "WeightEMA | None" = None,
) -> float:
    model.train()
    # Re-applied every epoch because `model.train()` above puts the whole
    # module tree back into training mode, BatchNorm layers included.
    set_backbone_frozen(model, freeze_backbone)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

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

    optimizer = build_optimizer(model, config)
    scheduler, scheduler_needs_metric = build_scheduler(optimizer, config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    ema = WeightEMA(model, config.ema_decay) if config.ema_decay > 0 else None

    history: list[dict] = []
    best_cer = float("inf")
    epochs_without_improvement = 0

    checkpoint_dir = Path(config.checkpoint_dir)
    latest_path = checkpoint_dir / f"{config.checkpoint_prefix}_latest.pt"
    best_path = checkpoint_dir / f"{config.checkpoint_prefix}_best.pt"

    for epoch in range(1, config.max_epochs + 1):
        freeze_backbone = epoch <= config.freeze_backbone_epochs
        if freeze_backbone and epoch == 1 and config.freeze_backbone_epochs:
            print(f"Backbone frozen for the first {config.freeze_backbone_epochs} epoch(s); head adapts first.")
        if not freeze_backbone and epoch == config.freeze_backbone_epochs + 1 and config.freeze_backbone_epochs:
            print(f"Backbone unfrozen at epoch {epoch}.")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.grad_clip, freeze_backbone=freeze_backbone, ema=ema,
        )

        # Validate (and checkpoint) the EMA weights when EMA is on — those
        # are the weights that will be used at test time, so selecting the
        # best epoch on the live weights instead would pick the wrong one.
        if ema is not None:
            with ema.swapped_into(model) as averaged:
                val_metrics = evaluate(averaged, val_loader, tokenizer, criterion, device)
                checkpoint_state = {k: v.detach().clone() for k, v in averaged.state_dict().items()}
        else:
            val_metrics = evaluate(model, val_loader, tokenizer, criterion, device)
            checkpoint_state = None

        if scheduler_needs_metric:
            scheduler.step(val_metrics["cer"])
        else:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "frozen_backbone": freeze_backbone,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | lr={current_lr:.1e} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"| val_CharAcc={(1.0 - val_metrics['cer'])*100:.2f}% | val_CER={val_metrics['cer']*100:.2f}% "
            f"| val_WordLevel={val_metrics['word_level_accuracy']*100:.2f}% "
            f"| val_ExactLine={val_metrics['word_accuracy']*100:.2f}%"
        )

        _save(latest_path, model, checkpoint_state, optimizer, epoch, val_metrics, tokenizer, model_config)

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            epochs_without_improvement = 0
            _save(best_path, model, checkpoint_state, optimizer, epoch, val_metrics, tokenizer, model_config)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch}: no val CER improvement for {config.patience} epochs.")
                break

    return history


def _save(path, model, ema_state, optimizer, epoch, val_metrics, tokenizer, model_config) -> None:
    """Write a checkpoint holding whichever weights were validated.

    With EMA on, `ema_state` holds the averaged weights that produced
    `val_metrics`; they are swapped in only for the duration of the save so
    the checkpoint on disk and the metrics beside it describe the same
    model.
    """
    if ema_state is None:
        save_checkpoint(path, model, optimizer, epoch, val_metrics, tokenizer.char_to_idx, model_config)
        return

    live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state)
    try:
        save_checkpoint(path, model, optimizer, epoch, val_metrics, tokenizer.char_to_idx, model_config)
    finally:
        model.load_state_dict(live_state)

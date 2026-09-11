"""
Training loop mechanics: checkpointing, early stopping, seed-fixing.

This deliberately does NOT re-prove convergence quality (test_overfit.py
already does that, at real cost — ~4 minutes). This file uses a tiny
dataset and very few epochs, and only checks that train.py's plumbing
(checkpoints saved/loadable, history shape, early stopping trigger) works.

Run:
    .venv/bin/pytest tests/test_train.py -v
"""

import csv

import torch
from torch.utils.data import DataLoader

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.model_baseline import BaselineCRNN
from src.tokenizer import KhmerTokenizer
from src.train import TrainConfig, evaluate, fit, train_one_epoch
from src.utils import get_device, load_checkpoint, set_seed
from tests.test_dataset import WORDS, WRITER_IDS, _render_word_image

TINY_SIZE = 8


def _build_tiny_dataset(tmp_path, tokenizer):
    root = tmp_path / "train_synthetic"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    rows = []
    for i, word in enumerate(WORDS[:TINY_SIZE]):
        filename = f"{i:04d}.png"
        _render_word_image(word, 32, images_dir / filename)
        rows.append({"image": filename, "label": word, "writer_id": WRITER_IDS[i % len(WRITER_IDS)]})

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)

    samples = load_metadata_csv(csv_path, images_dir)
    return KhmerWordDataset(samples, tokenizer, image_height=48, max_width=256)


def test_set_seed_makes_model_init_deterministic():
    set_seed(123)
    model_a = BaselineCRNN(num_classes=10)
    set_seed(123)
    model_b = BaselineCRNN(num_classes=10)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b), "same seed must produce identical initial weights"


def test_train_one_epoch_and_evaluate_run_without_error(tmp_path):
    set_seed(0)
    tokenizer = KhmerTokenizer.build_from_labels(WORDS)
    dataset = _build_tiny_dataset(tmp_path, tokenizer)
    loader = DataLoader(dataset, batch_size=TINY_SIZE, shuffle=True, collate_fn=collate_fn)

    model = BaselineCRNN(num_classes=tokenizer.vocab_size)
    device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    loss_epoch_1 = train_one_epoch(model, loader, optimizer, criterion, device)
    loss_epoch_2 = train_one_epoch(model, loader, optimizer, criterion, device)
    assert loss_epoch_1 > 0 and loss_epoch_2 > 0

    metrics = evaluate(model, loader, tokenizer, criterion, device)
    assert set(metrics.keys()) == {"cer", "wer", "word_accuracy", "loss"}
    assert 0.0 <= metrics["cer"]
    assert 0.0 <= metrics["word_accuracy"] <= 1.0


def test_fit_saves_loadable_checkpoints_with_required_fields(tmp_path):
    set_seed(0)
    tokenizer = KhmerTokenizer.build_from_labels(WORDS)
    dataset = _build_tiny_dataset(tmp_path, tokenizer)
    loader = DataLoader(dataset, batch_size=TINY_SIZE, shuffle=True, collate_fn=collate_fn)

    model_config = {"model_class": "BaselineCRNN", "num_classes": tokenizer.vocab_size, "hidden_size": 256, "num_lstm_layers": 2, "dropout": 0.2}
    model = BaselineCRNN(num_classes=tokenizer.vocab_size)

    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainConfig(max_epochs=3, patience=10, checkpoint_dir=str(checkpoint_dir), checkpoint_prefix="test_a1")

    history = fit(model, loader, loader, tokenizer, model_config, config=config)

    assert len(history) == 3
    assert all("train_loss" in h and "val_cer" in h and "val_word_accuracy" in h for h in history)

    latest_path = checkpoint_dir / "test_a1_latest.pt"
    best_path = checkpoint_dir / "test_a1_best.pt"
    assert latest_path.exists()
    assert best_path.exists()

    fresh_model = BaselineCRNN(num_classes=tokenizer.vocab_size)
    checkpoint = load_checkpoint(latest_path, fresh_model)

    for key in ("model_state_dict", "optimizer_state_dict", "epoch", "val_metrics", "tokenizer_char_to_idx", "model_config"):
        assert key in checkpoint, f"checkpoint is missing required field: {key}"

    assert checkpoint["tokenizer_char_to_idx"] == tokenizer.char_to_idx
    assert checkpoint["model_config"] == model_config

    # `fit()` moved `model` onto whatever device it trained on (e.g. MPS);
    # `fresh_model` was never moved, so compare values on a common device.
    for p_orig, p_loaded in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_orig.cpu(), p_loaded.cpu()), "loaded checkpoint must exactly restore the trained weights"


def test_early_stopping_triggers_before_max_epochs_when_patience_is_tiny(tmp_path):
    set_seed(0)
    tokenizer = KhmerTokenizer.build_from_labels(WORDS)
    dataset = _build_tiny_dataset(tmp_path, tokenizer)
    loader = DataLoader(dataset, batch_size=TINY_SIZE, shuffle=True, collate_fn=collate_fn)

    model_config = {"model_class": "BaselineCRNN", "num_classes": tokenizer.vocab_size}
    model = BaselineCRNN(num_classes=tokenizer.vocab_size)

    # A learning rate this large destabilizes training badly enough that
    # val CER will not keep improving every epoch, so patience=1 should
    # force a stop well before a generous max_epochs is reached.
    config = TrainConfig(
        learning_rate=5.0,
        max_epochs=50,
        patience=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_prefix="test_early_stop",
    )

    history = fit(model, loader, loader, tokenizer, model_config, config=config)
    assert len(history) < 50, "early stopping should have triggered well before max_epochs with patience=1"

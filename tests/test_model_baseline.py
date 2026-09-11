"""
Shape and integration verification for the A1 baseline CRNN
(VGG CNN + BiLSTM + CTC head), using SYNTHETIC PLACEHOLDER images from
the same rendering helper as test_dataset.py — no real handwriting exists
yet, this only proves the model/dataset/CTC wiring is correct.

Run:
    .venv/bin/pytest tests/test_model_baseline.py -v
"""

import csv

import pytest
import torch
import torch.nn as nn

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.model_baseline import BaselineCRNN, VGGFeatureExtractor
from src.tokenizer import KhmerTokenizer
from tests.test_dataset import WORDS, WRITER_IDS, _render_word_image


@pytest.fixture(scope="module")
def tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(WORDS)


@pytest.fixture(scope="module")
def model(tokenizer) -> BaselineCRNN:
    torch.manual_seed(0)
    return BaselineCRNN(num_classes=tokenizer.vocab_size)


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory, tokenizer) -> KhmerWordDataset:
    root = tmp_path_factory.mktemp("model_baseline_synthetic")
    images_dir = root / "images"
    images_dir.mkdir()

    rows = []
    for i, word in enumerate(WORDS):
        font_size = 28 + (i % 5) * 6
        filename = f"{i:04d}.png"
        _render_word_image(word, font_size, images_dir / filename)
        rows.append({"image": filename, "label": word, "writer_id": WRITER_IDS[i % len(WRITER_IDS)]})

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)

    samples = load_metadata_csv(csv_path, images_dir)
    return KhmerWordDataset(samples, tokenizer, image_height=48, max_width=256)


def test_cnn_collapses_height_to_one_and_preserves_batch(model):
    x = torch.randn(3, 1, 48, 128)
    features = model.cnn(x)
    assert features.shape[0] == 3
    assert features.shape[1] == 512
    assert features.shape[2] == 1, "CNN must collapse height 48 -> 1"


@pytest.mark.parametrize("width", [16, 32, 64, 100, 128, 200, 256])
def test_compute_sequence_length_matches_actual_forward_pass(model, width):
    x = torch.randn(2, 1, 48, width)
    log_probs = model(x)
    actual_t = log_probs.shape[0]
    predicted_t = model.compute_sequence_length(width)
    assert actual_t == predicted_t, (
        f"compute_sequence_length({width}) predicted {predicted_t} but the "
        f"real forward pass produced T={actual_t} — the analytic formula "
        "must exactly match the CNN, not just approximate it."
    )


def test_forward_pass_output_shape(tokenizer, model):
    batch_size, width = 4, 96
    x = torch.randn(batch_size, 1, 48, width)
    log_probs = model(x)
    expected_t = width // BaselineCRNN.WIDTH_DOWNSAMPLE_FACTOR
    assert log_probs.shape == (expected_t, batch_size, tokenizer.vocab_size)


def test_log_softmax_output_is_a_valid_log_probability_distribution(model):
    x = torch.randn(2, 1, 48, 64)
    log_probs = model(x)
    probs = log_probs.exp()
    total_prob_per_timestep = probs.sum(dim=2)  # should sum to ~1 across classes
    assert torch.allclose(total_prob_per_timestep, torch.ones_like(total_prob_per_timestep), atol=1e-4)


def test_too_narrow_image_raises_clear_error(model):
    x = torch.randn(1, 1, 48, 3)  # far below the minimum width the CNN needs
    with pytest.raises(ValueError):
        model(x)


def test_trainable_parameter_count_is_reported_and_nonzero(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0
    print(f"\nBaselineCRNN trainable parameters: {num_params:,}")


def test_full_integration_ctc_loss_and_backward_pass(tokenizer, model, synthetic_dataset):
    """The real end-to-end smoke test: real (synthetic-image) batch ->
    dataset -> collate_fn -> model -> CTCLoss -> backward(). This is the
    exact pipeline the tiny overfit test (Gate #5) will scale up later."""
    from torch.utils.data import DataLoader

    loader = DataLoader(synthetic_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    log_probs = model(batch["images"])  # [T, B, num_classes]
    t, b, _ = log_probs.shape

    # input_lengths must reflect each sample's TRUE (pre-padding) width,
    # not the batch's padded max — otherwise CTC would treat white padding
    # columns as real, potentially-informative time steps.
    input_lengths = torch.tensor(
        [model.compute_sequence_length(w.item()) for w in batch["image_widths"]],
        dtype=torch.long,
    )
    target_lengths = batch["target_lengths"]

    assert torch.all(input_lengths <= t), "no sample's true input length should exceed the padded batch's T"
    assert torch.all(input_lengths >= target_lengths), (
        "CTC requires input_length >= target_length for every sample — if this "
        "fails, the words are rendered too small/narrow for the current CNN "
        "downsampling factor and need larger synthetic font sizes."
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, batch["targets"], input_lengths, target_lengths)

    assert torch.isfinite(loss), f"CTC loss must be finite, got {loss.item()}"

    model.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "backward() should have produced non-zero gradients somewhere in the model"

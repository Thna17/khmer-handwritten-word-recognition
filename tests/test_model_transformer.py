"""
Shape and integration verification for the A3 Transformer-sequence CRNN
(same VGG CNN as A1 + Transformer encoder replacing the BiLSTM + CTC head).

Run:
    .venv/bin/pytest tests/test_model_transformer.py -v
"""

import csv

import pytest
import torch
import torch.nn as nn

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.model_transformer import SinusoidalPositionalEncoding, TransformerCRNN
from src.tokenizer import KhmerTokenizer
from tests.test_dataset import WORDS, WRITER_IDS, _render_word_image


@pytest.fixture(scope="module")
def tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(WORDS)


@pytest.fixture(scope="module")
def model(tokenizer) -> TransformerCRNN:
    torch.manual_seed(0)
    return TransformerCRNN(num_classes=tokenizer.vocab_size)


# ---------------------------------------------------------------------
# Basic shape / construction checks
# ---------------------------------------------------------------------

def test_rejects_mismatched_d_model(tokenizer):
    with pytest.raises(ValueError):
        TransformerCRNN(num_classes=tokenizer.vocab_size, d_model=256)  # CNN outputs 512, not 256


def test_rejects_d_model_not_divisible_by_nhead(tokenizer):
    with pytest.raises(ValueError):
        TransformerCRNN(num_classes=tokenizer.vocab_size, d_model=512, nhead=7)


@pytest.mark.parametrize("width", [16, 32, 64, 100, 128, 200, 256])
def test_compute_sequence_length_matches_actual_forward_pass(model, width):
    x = torch.randn(2, 1, 48, width)
    log_probs = model(x)
    actual_t = log_probs.shape[0]
    predicted_t = model.compute_sequence_length(width)
    assert actual_t == predicted_t == width // 4


def test_forward_pass_output_shape(tokenizer, model):
    batch_size, width = 4, 96
    x = torch.randn(batch_size, 1, 48, width)
    log_probs = model(x)
    expected_t = width // TransformerCRNN.WIDTH_DOWNSAMPLE_FACTOR
    assert log_probs.shape == (expected_t, batch_size, tokenizer.vocab_size)


def test_log_softmax_output_is_a_valid_log_probability_distribution(model):
    x = torch.randn(2, 1, 48, 64)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=2), torch.ones_like(probs.sum(dim=2)), atol=1e-4)


def test_too_narrow_image_raises_clear_error(model):
    x = torch.randn(1, 1, 48, 3)
    with pytest.raises(ValueError):
        model(x)


def test_trainable_parameter_count_is_reported_and_nonzero(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0
    print(f"\nTransformerCRNN trainable parameters: {num_params:,}")


# ---------------------------------------------------------------------
# Sinusoidal positional encoding: correctness in isolation
# ---------------------------------------------------------------------

def test_positional_encoding_at_position_zero_is_sin0_cos0():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=16)
    pos0 = pe.pe[0, 0]  # [d_model]
    assert torch.allclose(pos0[0::2], torch.zeros_like(pos0[0::2]), atol=1e-6)  # sin(0) = 0
    assert torch.allclose(pos0[1::2], torch.ones_like(pos0[1::2]), atol=1e-6)   # cos(0) = 1


def test_positional_encoding_differs_across_positions():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=32)
    assert not torch.allclose(pe.pe[0, 0], pe.pe[1, 0])
    assert not torch.allclose(pe.pe[5, 0], pe.pe[10, 0])


def test_positional_encoding_forward_adds_correctly():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=16)
    x = torch.zeros(5, 2, 8)  # [T=5, B=2, d_model=8]
    out = pe(x)
    assert torch.equal(out, pe.pe[:5].expand(5, 2, 8))


def test_positional_encoding_rejects_sequences_longer_than_max_len():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=4)
    x = torch.zeros(5, 1, 8)
    with pytest.raises(ValueError):
        pe(x)


# ---------------------------------------------------------------------
# WHY positional encoding matters: permutation-equivariance proofs.
# Pure self-attention (no positional info) treats a sequence as a SET —
# permuting the input permutes the output the same way. Adding positional
# encoding breaks that symmetry, which is exactly why it's needed for a
# task where character ORDER is the entire point.
# ---------------------------------------------------------------------

def test_transformer_alone_is_permutation_equivariant_without_positional_encoding():
    torch.manual_seed(0)
    d_model = 512
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.0, batch_first=False)
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    transformer.eval()

    t, b = 6, 2
    x = torch.randn(t, b, d_model)
    permutation = torch.randperm(t)

    with torch.no_grad():
        out_original = transformer(x)
        out_permuted_input = transformer(x[permutation])

    # Permuting the INPUT time order produces the exact same permutation
    # of the OUTPUT -- proving plain self-attention has no notion of order,
    # it only relates content to content, not position to position.
    assert torch.allclose(out_permuted_input, out_original[permutation], atol=1e-5)


def test_positional_encoding_breaks_transformer_permutation_equivariance():
    torch.manual_seed(0)
    d_model = 512
    pos_enc = SinusoidalPositionalEncoding(d_model, max_len=32)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.0, batch_first=False)
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    transformer.eval()

    t, b = 6, 2
    x = torch.randn(t, b, d_model)
    permutation = torch.randperm(t)

    with torch.no_grad():
        out_original = transformer(pos_enc(x))
        out_permuted_input = transformer(pos_enc(x[permutation]))

    # Now that each time step's content is tagged with ITS OWN position
    # before permuting, permuting-then-encoding is NOT the same as
    # encoding-then-permuting -- position now carries real information.
    assert not torch.allclose(out_permuted_input, out_original[permutation], atol=1e-4)


# ---------------------------------------------------------------------
# Full integration: same CTC smoke test as A1/A2
# ---------------------------------------------------------------------

def test_full_integration_ctc_loss_and_backward_pass(tokenizer, model):
    from torch.utils.data import DataLoader
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
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
        dataset = KhmerWordDataset(samples, tokenizer, image_height=48, max_width=256)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        batch = next(iter(loader))

    log_probs = model(batch["images"])
    t = log_probs.shape[0]

    input_lengths = torch.tensor(
        [model.compute_sequence_length(w.item()) for w in batch["image_widths"]],
        dtype=torch.long,
    )
    target_lengths = batch["target_lengths"]

    assert torch.all(input_lengths <= t)
    assert torch.all(input_lengths >= target_lengths)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, batch["targets"], input_lengths, target_lengths)
    assert torch.isfinite(loss)

    model.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0

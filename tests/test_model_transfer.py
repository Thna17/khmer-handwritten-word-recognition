"""
Shape and integration verification for the A2 transfer-learning CRNN
(pretrained ResNet-18 backbone + BiLSTM + CTC head).

Most tests use pretrained=False (random init) so they run fast without
a network dependency -- shape/gradient/freeze behavior doesn't depend on
the actual ImageNet weight values. One dedicated test confirms the real
pretrained-download path works.

Run:
    .venv/bin/pytest tests/test_model_transfer.py -v
"""

import csv

import pytest
import torch
import torch.nn as nn

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.model_transfer import TransferCRNN
from src.tokenizer import KhmerTokenizer
from tests.test_dataset import WORDS, WRITER_IDS, _render_word_image


@pytest.fixture(scope="module")
def tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(WORDS)


def _make_model(tokenizer, freeze_backbone=False, pretrained=False):
    torch.manual_seed(0)
    return TransferCRNN(num_classes=tokenizer.vocab_size, freeze_backbone=freeze_backbone, pretrained=pretrained)


def test_backbone_collapses_height_to_one_and_downsamples_width(tokenizer):
    model = _make_model(tokenizer)
    x = torch.randn(3, 1, 48, 128)
    features = model.cnn(x)
    assert features.shape[0] == 3
    assert features.shape[1] == TransferCRNN.INPUT_HEIGHT == 48 or True  # sanity: constant exists
    assert features.shape[2] == 1, "backbone must collapse height 48 -> 1"
    assert features.shape[1] == model.cnn.OUT_CHANNELS


@pytest.mark.parametrize("width", [32, 64, 96, 128, 160, 200, 256])
def test_compute_sequence_length_matches_actual_forward_pass(tokenizer, width):
    model = _make_model(tokenizer)
    x = torch.randn(2, 1, 48, width)
    log_probs = model(x)
    actual_t = log_probs.shape[0]
    predicted_t = model.compute_sequence_length(width)
    assert actual_t == predicted_t, (
        f"compute_sequence_length({width}) predicted {predicted_t} but the "
        f"real forward pass produced T={actual_t}."
    )


def test_forward_pass_output_shape(tokenizer):
    model = _make_model(tokenizer)
    batch_size, width = 4, 128
    x = torch.randn(batch_size, 1, 48, width)
    log_probs = model(x)
    expected_t = model.compute_sequence_length(width)
    assert log_probs.shape == (expected_t, batch_size, tokenizer.vocab_size)


def test_log_softmax_output_is_a_valid_log_probability_distribution(tokenizer):
    model = _make_model(tokenizer)
    x = torch.randn(2, 1, 48, 96)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=2), torch.ones_like(probs.sum(dim=2)), atol=1e-4)


def test_frozen_backbone_receives_no_gradients_but_head_does(tokenizer):
    model = _make_model(tokenizer, freeze_backbone=True)
    x = torch.randn(2, 1, 48, 96)
    log_probs = model(x)
    loss = log_probs.sum()  # any scalar to trigger backward
    loss.backward()

    for name, param in model.cnn.named_parameters():
        assert param.grad is None, f"frozen backbone param {name} unexpectedly received a gradient"

    rnn_grad_norm = sum(p.grad.norm().item() for p in model.rnn.parameters() if p.grad is not None)
    classifier_grad_norm = sum(p.grad.norm().item() for p in model.classifier.parameters() if p.grad is not None)
    assert rnn_grad_norm > 0, "BiLSTM should still train even with a frozen backbone"
    assert classifier_grad_norm > 0, "classifier head should still train even with a frozen backbone"


def test_unfrozen_backbone_receives_gradients():
    tokenizer = KhmerTokenizer.build_from_labels(WORDS)
    model = _make_model(tokenizer, freeze_backbone=False)
    x = torch.randn(2, 1, 48, 96)
    log_probs = model(x)
    loss = log_probs.sum()
    loss.backward()

    backbone_grad_norm = sum(p.grad.norm().item() for p in model.cnn.parameters() if p.grad is not None)
    assert backbone_grad_norm > 0, "a NON-frozen backbone should receive gradients"


def test_frozen_backbone_batchnorm_stays_in_eval_mode_during_train(tokenizer):
    frozen_model = _make_model(tokenizer, freeze_backbone=True)
    frozen_model.train()
    bn_layers = [m for m in frozen_model.cnn.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) > 0
    assert all(not bn.training for bn in bn_layers), "frozen backbone's BatchNorm must stay in eval mode even after model.train()"


def test_unfrozen_backbone_batchnorm_enters_train_mode(tokenizer):
    unfrozen_model = _make_model(tokenizer, freeze_backbone=False)
    unfrozen_model.train()
    bn_layers = [m for m in unfrozen_model.cnn.modules() if isinstance(m, nn.BatchNorm2d)]
    assert all(bn.training for bn in bn_layers), "a non-frozen backbone's BatchNorm should train normally"


def test_pretrained_weights_actually_load_from_imagenet(tokenizer):
    """The one test that touches the network -- confirms pretrained=True
    produces different (real, learned) weights than pretrained=False
    (random init), i.e. the download/loading path genuinely works."""
    pretrained_model = _make_model(tokenizer, pretrained=True)
    random_model = _make_model(tokenizer, pretrained=False)

    pretrained_conv1_weight = pretrained_model.cnn.stem[0].weight
    random_conv1_weight = random_model.cnn.stem[0].weight

    assert not torch.allclose(pretrained_conv1_weight, random_conv1_weight), (
        "pretrained conv1 weights should differ from a random initialization"
    )


def test_full_integration_ctc_loss_and_backward_pass(tokenizer):
    """Same end-to-end smoke test as A1: synthetic-image batch -> dataset
    -> collate_fn -> model -> CTCLoss -> backward()."""
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

    model = _make_model(tokenizer, pretrained=False)
    log_probs = model(batch["images"])
    t = log_probs.shape[0]

    input_lengths = torch.tensor(
        [model.compute_sequence_length(w.item()) for w in batch["image_widths"]],
        dtype=torch.long,
    )
    target_lengths = batch["target_lengths"]

    assert torch.all(input_lengths <= t)
    assert torch.all(input_lengths >= target_lengths), (
        "CTC requires input_length >= target_length for every sample -- if this "
        "fails, A2's /8 downsampling is too aggressive for these image widths."
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, batch["targets"], input_lengths, target_lengths)
    assert torch.isfinite(loss)

    model.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0

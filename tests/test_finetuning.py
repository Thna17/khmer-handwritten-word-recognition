"""Tests for the Stage-2 fine-tuning machinery: EMA, backbone freezing,
differential learning rates, schedulers, and synthetic data mixing.

Each test pins a behaviour that, when it silently broke, produced a plausible
looking training log and a bad model — the failure mode this whole stage has
already suffered once.
"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn

from src.dataset import Sample
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.run_experiment import build_training_pool
from src.tokenizer import KhmerTokenizer
from src.train import (
    TrainConfig,
    WeightEMA,
    build_optimizer,
    build_scheduler,
    set_backbone_frozen,
)

LABELS = ["ទឹក", "ភ្លើង", "ផ្ទះ"]


def _tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(LABELS)


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

def test_ema_warmup_does_not_anchor_the_average_to_the_initial_weights():
    """Without the decay warmup, a short fine-tune validates weights that are
    still largely the initialization — the bug that made a 3-epoch smoke run
    report val_CER 100% while its training loss fell normally."""
    model = nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.fill_(0.0)

    ema = WeightEMA(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for _ in range(20):
        ema.update(model)

    # 20 steps at a flat 0.99 would leave 0.99**20 = 0.82 of the zeros behind.
    assert ema.shadow["weight"][0, 0].item() > 0.99


def test_ema_decay_converges_to_the_requested_value():
    ema = WeightEMA(nn.Linear(2, 2), decay=0.9)
    assert ema._effective_decay() == pytest.approx(0.1)
    ema.steps = 10_000
    assert ema._effective_decay() == pytest.approx(0.9)


def test_ema_copies_integer_buffers_instead_of_averaging_them():
    """BatchNorm's num_batches_tracked is an integer counter; averaging it is
    meaningless and `lerp`-style updates reject integer tensors outright."""
    model = nn.BatchNorm2d(3)
    ema = WeightEMA(model, decay=0.99)
    model(torch.rand(2, 3, 4, 4))
    ema.update(model)

    assert ema.shadow["num_batches_tracked"].dtype == torch.int64
    assert ema.shadow["num_batches_tracked"].item() == int(model.num_batches_tracked)


def test_ema_swap_restores_the_live_weights_even_when_evaluation_raises():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema = WeightEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)

    live = model.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with ema.swapped_into(model):
            raise RuntimeError("evaluation blew up")

    assert torch.equal(model.weight, live)


# --------------------------------------------------------------------------
# Backbone freezing
# --------------------------------------------------------------------------

def test_freezing_stops_gradients_and_pins_batchnorm_to_eval():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    model.train()
    set_backbone_frozen(model, True)

    assert all(not p.requires_grad for p in model.cnn.parameters())
    assert all(not m.training for m in model.cnn.modules() if isinstance(m, nn.BatchNorm2d))
    # The head must stay trainable, otherwise the frozen epochs do nothing.
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfreezing_restores_training_for_the_backbone():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    set_backbone_frozen(model, True)
    set_backbone_frozen(model, False)
    model.train()

    assert all(p.requires_grad for p in model.cnn.parameters())


def test_frozen_backbone_weights_do_not_move_during_a_step():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    model.train()
    set_backbone_frozen(model, True)
    before = model.cnn.block1[0].weight.detach().clone()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(torch.rand(2, 1, 48, 64)).sum().backward()
    optimizer.step()

    assert torch.equal(model.cnn.block1[0].weight, before)


# --------------------------------------------------------------------------
# Optimizer / scheduler
# --------------------------------------------------------------------------

def test_backbone_gets_its_own_lower_learning_rate():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    config = TrainConfig(learning_rate=1e-3, backbone_lr_mult=0.1)
    optimizer = build_optimizer(model, config)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["head"]["lr"] == pytest.approx(1e-3)
    assert groups["backbone"]["lr"] == pytest.approx(1e-4)

    # Every parameter must land in exactly one group — none silently dropped.
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == len(list(model.parameters()))


def test_cosine_schedule_warms_up_then_decays():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    config = TrainConfig(learning_rate=1.0, scheduler="cosine", warmup_epochs=2,
                         max_epochs=10, min_lr_factor=0.0)
    optimizer = build_optimizer(model, config)
    scheduler, needs_metric = build_scheduler(optimizer, config)
    assert needs_metric is False

    seen = []
    for _ in range(10):
        seen.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert seen[0] < seen[1]                 # warming up
    assert seen[1] == pytest.approx(1.0)     # full LR at the end of warmup
    assert seen[-1] < seen[2]                # decaying afterwards
    assert all(lr >= 0 for lr in seen)


def test_plateau_schedule_still_consumes_the_validation_metric():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    _, needs_metric = build_scheduler(build_optimizer(model, TrainConfig()), TrainConfig())
    assert needs_metric is True


def test_unknown_scheduler_is_rejected():
    model = BaselineCRNN(num_classes=_tokenizer().vocab_size)
    config = TrainConfig(scheduler="triangular")
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(build_optimizer(model, config), config)


# --------------------------------------------------------------------------
# A2's sequence-length probe must not disturb BatchNorm
# --------------------------------------------------------------------------

def test_sequence_length_probe_leaves_batchnorm_statistics_untouched():
    """compute_sequence_length() runs an all-zeros image through the
    backbone. In training mode a BatchNorm layer updates its running
    statistics on every forward pass regardless of no_grad, so this probe
    used to fold "the image is uniformly black" into the statistics that
    eval-mode inference depends on — once per unseen width, and line widths
    are nearly all distinct."""
    model = TransferCRNN(num_classes=_tokenizer().vocab_size, pretrained=False)
    model.train()
    # Give the statistics something realistic to hold first.
    for _ in range(3):
        model(torch.rand(2, 1, 48, 256))

    batchnorm = model.cnn.stem[1]
    mean_before = batchnorm.running_mean.detach().clone()
    var_before = batchnorm.running_var.detach().clone()
    tracked_before = int(batchnorm.num_batches_tracked)

    for width in (128, 192, 256, 320, 384):
        model.compute_sequence_length(width)

    assert torch.equal(batchnorm.running_mean, mean_before)
    assert torch.equal(batchnorm.running_var, var_before)
    assert int(batchnorm.num_batches_tracked) == tracked_before
    # And the model must be left in training mode, as it was found.
    assert model.cnn.training


def test_sequence_length_probe_still_returns_the_true_width():
    model = TransferCRNN(num_classes=_tokenizer().vocab_size, pretrained=False)
    model.eval()
    for width in (128, 256):
        expected = model(torch.rand(1, 1, 48, width)).shape[0]
        assert model.compute_sequence_length(width) == expected


# --------------------------------------------------------------------------
# Training-pool assembly
# --------------------------------------------------------------------------

def _args(**overrides) -> argparse.Namespace:
    base = dict(real_oversample=1, mix_csv=None, mix_images_dir=None, mix_size=0, seed=42)
    base.update(overrides)
    return argparse.Namespace(**base)


def _samples(n: int) -> list[Sample]:
    return [Sample(image_path=f"img_{i}.png", label=LABELS[i % len(LABELS)], writer_id="W001") for i in range(n)]


def test_oversampling_repeats_the_real_samples():
    pool = build_training_pool(_samples(5), _args(real_oversample=4), _tokenizer())
    assert len(pool) == 20


def test_no_oversampling_leaves_the_pool_alone():
    originals = _samples(5)
    pool = build_training_pool(originals, _args(), _tokenizer())
    assert pool == originals
    assert pool is not originals  # a copy, so callers cannot mutate the split


def test_mixing_requires_an_images_dir():
    with pytest.raises(ValueError, match="--mix-images-dir"):
        build_training_pool(_samples(3), _args(mix_csv="somewhere.csv"), _tokenizer())


def test_mix_corpus_with_unknown_characters_is_rejected_up_front(tmp_path):
    """A mix corpus outside the vocabulary would otherwise fail thousands of
    steps into training, inside a DataLoader worker."""
    csv_path = tmp_path / "mix.csv"
    csv_path.write_text("image,label,writer_id\na.png,ZZZ,SYNTH_00\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from the vocabulary"):
        build_training_pool(
            _samples(3),
            _args(mix_csv=str(csv_path), mix_images_dir=str(tmp_path)),
            _tokenizer(),
        )


def test_mix_size_subsamples_deterministically(tmp_path):
    csv_path = tmp_path / "mix.csv"
    rows = ["image,label,writer_id"]
    rows += [f"s{i}.png,{LABELS[i % len(LABELS)]},SYNTH_00" for i in range(50)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def pool(seed):
        return build_training_pool(
            _samples(4),
            _args(mix_csv=str(csv_path), mix_images_dir=str(tmp_path), mix_size=10, seed=seed),
            _tokenizer(),
        )

    first, second = pool(42), pool(42)
    assert len(first) == 4 + 10
    assert [s.image_path for s in first] == [s.image_path for s in second]
    assert [s.image_path for s in pool(7)] != [s.image_path for s in first]


def test_linear_probe_backbone_is_never_unfrozen_by_the_staged_schedule():
    """A2 built with freeze_backbone=True is a linear probe for the whole
    run. The staged unfreeze must not silently convert it into a full
    fine-tune partway through and file the result under the probe's name."""
    model = TransferCRNN(num_classes=_tokenizer().vocab_size, pretrained=False, freeze_backbone=True)

    set_backbone_frozen(model, False)  # what the schedule does after the frozen epochs
    model.train()

    assert all(not p.requires_grad for p in model.cnn.parameters())
    assert all(not m.training for m in model.cnn.modules() if isinstance(m, nn.BatchNorm2d))

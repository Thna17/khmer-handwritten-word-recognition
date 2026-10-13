"""Tests for the fine-tuning augmentation pipeline.

The property that matters most here is NOT that the images change — it is
that they stay readable. Augmentation is paired with an unchanged CTC
target, so a transform that erases a character turns a training sample into
a lie. These tests pin the ink-retention guard that enforces that, using
the dataset's own smallest, faintest line images as the hard case.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.augment import MIN_INK_RETENTION, HandwritingAugmentor, _ink_fraction, build_augmentor


def _synthetic_line(width: int = 400, height: int = 30, stroke: int = 1) -> Image.Image:
    """A pale, thin, low-resolution line — the W001 regime (native 30px)."""
    array = np.full((height, width), 245, dtype=np.uint8)
    for x in range(10, width - 10, 9):
        array[height // 3: 2 * height // 3, x: x + stroke] = 90
    return Image.fromarray(array)


def test_zero_strength_returns_the_image_unchanged():
    assert build_augmentor(0.0) is None
    assert build_augmentor(-1.0) is None

    image = _synthetic_line()
    assert HandwritingAugmentor(strength=0.0)(image) is image


def test_augmentor_produces_a_different_but_still_grayscale_image():
    image = _synthetic_line()
    augmented = HandwritingAugmentor(strength=1.0, rng=random.Random(0))(image)

    assert augmented.mode == "L"
    assert augmented.size != image.size or np.any(np.array(augmented) != np.array(image))


def test_augmentation_is_deterministic_for_a_fixed_seed():
    image = _synthetic_line()
    first = HandwritingAugmentor(strength=1.0, rng=random.Random(7))(image)
    second = HandwritingAugmentor(strength=1.0, rng=random.Random(7))(image)

    assert first.size == second.size
    assert np.array_equal(np.array(first), np.array(second))


@pytest.mark.parametrize("height", [28, 48, 96])
def test_ink_is_never_destroyed_across_many_draws(height):
    """The guard that keeps augmented images honest, exercised at the three
    resolutions the real writers actually span (W001 ~30px, W008 ~49px,
    W007 ~83px)."""
    image = _synthetic_line(height=height, stroke=max(1, height // 20))
    baseline_ink = _ink_fraction(image)

    retentions = [
        _ink_fraction(HandwritingAugmentor(strength=1.0, rng=random.Random(seed))(image)) / baseline_ink
        for seed in range(150)
    ]
    assert min(retentions) >= MIN_INK_RETENTION


def test_reading_order_is_preserved_left_to_right():
    """Ink must not be mirrored or reordered: a line whose ink sits only in
    the left half has to still have it in the left half afterwards, or the
    CTC target would describe the characters in the wrong order."""
    array = np.full((40, 400), 250, dtype=np.uint8)
    array[12:28, 20:180] = 40  # ink in the LEFT half only
    image = Image.fromarray(array)

    for seed in range(40):
        augmented = HandwritingAugmentor(strength=1.0, rng=random.Random(seed))(image)
        pixels = np.array(augmented, dtype=np.float32)
        midpoint = pixels.shape[1] // 2
        left_ink = (255 - pixels[:, :midpoint]).sum()
        right_ink = (255 - pixels[:, midpoint:]).sum()
        assert left_ink > right_ink, f"seed {seed} moved the ink to the right half"


def test_detail_budget_protects_small_images_more_than_large_ones():
    augmentor = HandwritingAugmentor(strength=1.0, rng=random.Random(0))
    assert augmentor._detail_budget(28) < augmentor._detail_budget(96)
    assert 0.0 < augmentor._detail_budget(10) <= 1.0
    assert augmentor._detail_budget(1000) == 1.0


def test_strength_must_not_be_negative():
    with pytest.raises(ValueError, match="strength"):
        HandwritingAugmentor(strength=-0.5)

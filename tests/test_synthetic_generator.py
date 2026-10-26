"""Unit tests for Parallel Synthetic Generator & Augmentation Physics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_synthetic_data import (
    CANDIDATE_FONTS,
    apply_handwriting_physics,
    build_100k_text_corpus,
    init_worker,
    render_khmer_text,
)


@pytest.fixture(scope="module", autouse=True)
def setup_worker_fonts():
    init_worker(CANDIDATE_FONTS)


def test_synthetic_corpus_vocabulary_coverage():
    # Build a small corpus of 500 samples
    samples = build_100k_text_corpus(num_samples=500)
    assert len(samples) >= 500
    all_chars = set("".join(samples))

    # Test that common and rare Khmer Unicode code points are present
    assert "ក" in all_chars
    assert "្" in all_chars  # Coeng sign
    assert "ា" in all_chars  # Sra Aa


def test_handwriting_physics_augmentations():
    # Synthetic white image with a dark horizontal line
    canvas = np.full((48, 200), 255, dtype=np.uint8)
    canvas[20:28, 20:180] = 30  # Simulated text stroke

    augmented = apply_handwriting_physics(canvas)
    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == (48, 200)
    assert augmented.dtype == np.uint8
    # Should still contain both dark ink and light paper
    assert np.min(augmented) < 100
    assert np.max(augmented) > 180


def test_harfbuzz_rendering_validity():
    from scripts.generate_synthetic_data import _AVAILABLE_FONTS
    if not _AVAILABLE_FONTS:
        pytest.skip("No TrueType Khmer fonts available on current host.")

    font_path = _AVAILABLE_FONTS[0]
    test_phrase = "ការសិក្សាអក្សរ"
    rendered = render_khmer_text(test_phrase, font_path, font_size=28)

    assert rendered is not None
    assert rendered.ndim == 2
    assert rendered.shape[0] > 10
    assert rendered.shape[1] > 20

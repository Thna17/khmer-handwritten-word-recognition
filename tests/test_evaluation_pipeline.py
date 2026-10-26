"""Unit tests for Benchmark Evaluation & Confusion Analysis."""

from __future__ import annotations

import pytest

from scripts.evaluate_full_benchmark import compute_character_confusions


def test_character_confusion_alignment():
    # Test pair with substitution and deletion
    targets = ["ប្រធាន", "សិស្ស"]
    # Model confused Coeng Ro with Coeng Lo and missed a consonant
    predictions = ["ប្លធាន", "សិស"]

    confusions = compute_character_confusions(targets, predictions, top_k=10)
    assert len(confusions) > 0

    error_pairs = [pair for pair, count in confusions]
    # Check that omission or substitution was recorded
    assert any(p[1] == "<omitted>" or p[0] != p[1] for p in error_pairs)

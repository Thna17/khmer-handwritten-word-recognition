"""Unit tests for Add New Writer & Transcription Pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.add_new_writer import deskew_image, estimate_skew_angle, segment_text_lines
from scripts.transcribe_new_lines import update_writer_split, validate_khmer_transcription


def test_skew_estimation_and_deskew():
    # Create white canvas with a black horizontal bar
    canvas = np.full((200, 400), 255, dtype=np.uint8)
    canvas[80:120, 50:350] = 0

    angle = estimate_skew_angle(canvas)
    assert abs(angle) < 5.0, f"Angle should be near 0 for horizontal bar, got {angle}"

    rotated = deskew_image(canvas, angle)
    assert rotated.shape == canvas.shape


def test_segment_text_lines():
    # Create synthetic sheet with 3 horizontal bands
    canvas = np.full((300, 500), 255, dtype=np.uint8)
    canvas[40:80, 50:450] = 0    # Line 1
    canvas[130:170, 50:450] = 0  # Line 2
    canvas[220:260, 50:450] = 0  # Line 3

    spans = segment_text_lines(canvas, min_line_height=20, max_line_height=100, vertical_padding=5)
    assert len(spans) == 3, f"Expected 3 segmented lines, got {len(spans)}"


def test_validate_khmer_transcription():
    valid_chars = {"ក", "ា", "រ", "ស", "ិ", "ក", "្", "ស", "ា", " "}

    # Valid text
    valid_text = "ការសិក្សា"
    is_valid, norm, invalid = validate_khmer_transcription(valid_text, valid_chars)
    assert is_valid is True
    assert norm == "ការសិក្សា"
    assert len(invalid) == 0

    # Text containing unknown character (Latin 'A')
    invalid_text = "ការសិក្សា A"
    is_valid, norm, invalid = validate_khmer_transcription(invalid_text, valid_chars)
    assert is_valid is False
    assert any("'A'" in inv for inv in invalid)


def test_update_writer_split_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = Path(tmpdir) / "writer_split.json"
        initial_splits = {
            "train_writers": ["W001", "W002"],
            "val_writers": ["W003"],
            "test_writers": ["W007", "W0011"],
        }
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(initial_splits, f)

        # 1. Register new writer
        update_writer_split("W012", split_path=split_file)

        with open(split_file, "r", encoding="utf-8") as f:
            updated = json.load(f)

        assert "W012" in updated["train_writers"]
        assert updated["test_writers"] == ["W007", "W0011"]

        # 2. Safety violation: Attempting to register test writer should raise ValueError
        with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION"):
            update_writer_split("W007", split_path=split_file)

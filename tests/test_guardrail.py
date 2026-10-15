"""Unit tests for Input Guardrail & Non-Handwriting Detection."""

import numpy as np
import pytest
from PIL import Image

from src.guardrail import validate_ctc_prediction, validate_image_input


def test_validate_blank_image():
    blank = np.full((60, 600), 255, dtype=np.uint8)
    is_valid, reason = validate_image_input(blank)
    assert not is_valid
    assert "blank" in reason.lower() or "contrast" in reason.lower()


def test_validate_solid_dark_image():
    dark = np.full((60, 600), 10, dtype=np.uint8)
    is_valid, reason = validate_image_input(dark)
    assert not is_valid
    assert "blank" in reason.lower() or "contrast" in reason.lower()


def test_validate_vertical_crop():
    tall = np.full((500, 40), 255, dtype=np.uint8)
    tall[100:400, 15:25] = 0  # vertical line
    is_valid, reason = validate_image_input(tall)
    assert not is_valid
    assert "vertical" in reason.lower()


def test_validate_too_small_image():
    tiny = np.full((10, 10), 255, dtype=np.uint8)
    is_valid, reason = validate_image_input(tiny)
    assert not is_valid
    assert "small" in reason.lower()


def test_validate_real_handwriting_passes():
    # Synthetic line image on white background
    img = np.full((60, 500), 255, dtype=np.uint8)
    # Draw several strokes
    for x in range(30, 450, 20):
        img[20:45, x : x + 4] = 0
        img[30:34, x : x + 12] = 0
    is_valid, reason = validate_image_input(img)
    assert is_valid, f"Expected valid handwriting candidate, got {reason}"


def test_validate_ctc_empty_prediction():
    is_valid, reason = validate_ctc_prediction(
        predicted_text="",
        confidence=0.8,
        image_width=400,
    )
    assert not is_valid
    assert "could not detect" in reason.lower() or "empty" in reason.lower()


def test_validate_ctc_low_confidence():
    is_valid, reason = validate_ctc_prediction(
        predicted_text="កខគឃ",
        confidence=0.25,
        image_width=400,
        min_confidence=0.40,
    )
    assert not is_valid
    assert "too low" in reason.lower()


def test_validate_ctc_repetitive_hallucination():
    is_valid, reason = validate_ctc_prediction(
        predicted_text="កកកកកកកកកកកក",
        confidence=0.75,
        image_width=400,
    )
    assert not is_valid
    assert "repetitive" in reason.lower()


def test_validate_ctc_filler_dots():
    is_valid, reason = validate_ctc_prediction(
        predicted_text="......................",
        confidence=0.85,
        image_width=400,
    )
    assert not is_valid
    assert "filler dots" in reason.lower()


def test_validate_ctc_valid_prediction():
    is_valid, reason = validate_ctc_prediction(
        predicted_text="ប្រធាន: នៅក្នុងច្បាប់ល្បើកថ្មី",
        confidence=0.88,
        image_width=800,
    )
    assert is_valid

"""Topic & Handwriting Input Guardrail for Khmer Handwritten Text Recognition.

Protects the inference pipeline from out-of-domain (OOD) inputs:
  1. Pre-inference physical & visual checks:
     - Blanks / solid colors (flat standard deviation)
     - Extreme aspect ratios (e.g., tall vertical portrait crops)
     - Extreme ink density (too dark / solid blocks or < 1.2% ink)
     - Random noise / textured backgrounds
  2. Post-inference CTC domain validation:
     - Empty predictions on non-trivial images
     - Very low model confidence (< 40%)
     - Degenerate repetitive hallucinations (e.g. repeating a single token indefinitely)
     - Character density incompatible with natural handwriting lines
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from PIL import Image


def validate_image_input(img_input: Image.Image | np.ndarray) -> tuple[bool, str]:
    """Performs fast visual and geometric pre-checks to determine if an image
    is a plausible candidate for a handwritten Khmer line or word crop.

    Returns:
        (is_valid: bool, reason: str)
    """
    if isinstance(img_input, Image.Image):
        # Convert to grayscale array
        if img_input.mode in ("RGBA", "LA") or (img_input.mode == "P" and "transparency" in img_input.info):
            bg = Image.new("RGB", img_input.size, (255, 255, 255))
            rgba = img_input.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            img_gray = np.array(bg.convert("L"))
        else:
            img_gray = np.array(img_input.convert("L"))
    elif isinstance(img_input, np.ndarray):
        if len(img_input.shape) == 3:
            img_gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_input
    else:
        return False, "Unsupported image format."

    if img_gray is None or img_gray.size == 0:
        return False, "Image is empty or unreadable."

    h, w = img_gray.shape

    # 1. Minimum resolution
    if h < 16 or w < 16:
        return False, f"Image resolution ({w}x{h} px) is too small to contain legible handwriting."

    # 2. Aspect ratio (text lines/words are horizontal)
    aspect = w / float(h)
    if aspect < 0.35:
        return False, f"Aspect ratio ({aspect:.2f}) is too vertical. Khmer text is written horizontally in lines or words."
    if aspect > 45.0:
        return False, f"Aspect ratio ({aspect:.2f}) is abnormally wide."

    # 3. Flatness / Contrast (detects blank white paper or solid black blocks)
    std = float(np.std(img_gray))
    if std < 10.0:
        return False, "Image is completely blank, solid color, or lacks sufficient contrast. No handwriting ink detected."

    # 4. Foreground ink coverage via Otsu binarization
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(np.count_nonzero(thresh) / thresh.size)
    if ink_ratio < 0.012:
        return False, "No handwriting ink detected (less than 1.2% dark pixels). The image appears blank."
    if ink_ratio > 0.58:
        return False, "Excessive dark coverage (> 58%). The image looks like a dark photograph or solid graphic rather than handwriting on paper."

    # 5. Background texture check (paper vs natural scene)
    # In handwriting on paper, the top 25% brightest pixels (the paper) have low variance
    bright_pixels = img_gray[img_gray >= np.percentile(img_gray, 75)]
    if len(bright_pixels) > 50:
        bg_std = float(np.std(bright_pixels))
        if bg_std > 32.0:
            return False, f"Complex textured background detected (std {bg_std:.1f} > 32). Expected relatively uniform paper background."

    # 6. Connected components & noise speckle analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    areas = [s[cv2.CC_STAT_AREA] for s in stats[1:]]
    if not areas:
        return False, "No ink components found."

    # In random salt-and-pepper noise, > 80% of components are tiny 1-4 pixel specks
    if len(areas) >= 40:
        tiny_specks = sum(1 for a in areas if a < 6)
        speck_ratio = tiny_specks / float(len(areas))
        if speck_ratio > 0.82:
            return False, "Image consists of random noise or grain rather than coherent handwritten strokes."

    # Valid strokes resembling characters
    valid_strokes = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 5 and s[cv2.CC_STAT_HEIGHT] >= 3]
    if len(valid_strokes) < 2 and w > 80:
        return False, "No distinct character strokes detected."

    return True, "Valid handwriting image candidate"


def validate_ctc_prediction(
    predicted_text: str,
    confidence: float,
    image_width: int,
    min_confidence: float = 0.40,
) -> tuple[bool, str]:
    """Validates model prediction against out-of-domain CTC failure signatures.

    Returns:
        (is_valid: bool, reason: str)
    """
    clean_text = predicted_text.strip()

    # 1. Empty prediction on non-trivial image
    if len(clean_text) == 0:
        return False, "The model could not detect any legible Khmer characters in this image."

    # 2. Confidence threshold
    if confidence < min_confidence:
        return (
            False,
            f"Recognition confidence is too low ({confidence*100:.1f}% < {min_confidence*100:.0f}%). "
            f"The image does not match learned Khmer handwriting patterns.",
        )

    # 3. Long strings of placeholder dots or punctuation
    if re.search(r"[\.]{4,}", clean_text):
        return False, "Prediction contains continuous filler dots rather than valid Khmer words."

    # 4. Degenerate repeating character loops (OOD hallucination)
    # e.g. "កកកកកកកកកកកក"
    if len(clean_text) >= 6:
        # Check if single character repeats for majority of string
        char_counts = {}
        for c in clean_text:
            if not c.isspace():
                char_counts[c] = char_counts.get(c, 0) + 1
        if char_counts:
            most_freq_char, max_count = max(char_counts.items(), key=lambda x: x[1])
            repeat_ratio = max_count / float(len(clean_text))
            if repeat_ratio > 0.45 and len(clean_text) > 8:
                return (
                    False,
                    f"Degenerate repetitive prediction detected ('{most_freq_char}' repeated {max_count} times). "
                    f"This indicates the input image is outside the domain of Khmer handwriting.",
                )

    # 5. Plausible character-to-width density
    # A line of width 800px typically has 15 to 80 chars; 1 char in 800px is noise.
    if image_width > 150 and len(clean_text) <= 1:
        return False, f"Only 1 character predicted across a wide image ({image_width}px). The image appears to be non-text or noise."

    return True, "Valid Khmer text prediction"

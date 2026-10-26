#!/usr/bin/env python3
"""Add New Writer & Automated Line Segmentation Pipeline.

Automates the ingestion of newly scanned student handwriting sheets into the
Khmer Handwritten Text Recognition (KHTR) benchmark:
1. Deskews scanned sheets with automatic angle estimation.
2. Segments text lines using smoothed horizontal projection profiling.
3. Automatically applies vertical padding to preserve Khmer 3-tier vertical
   stacking (subscript Coeng ligatures and floating upper vowels).
4. Saves line crops to:
   data/handwritten_external/{writer_id}/lines_raw/{writer_id}_P{page:03d}_L{line:03d}.png
5. Saves a visual segmentation audit map for human verification.

Usage:
  python scripts/add_new_writer.py \
    --writer-id W012 \
    --input-page path/to/scanned_page.png \
    --page-num 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image


def estimate_skew_angle(image: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimates the skew angle of a scanned document in degrees using minAreaRect."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize with Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge character glyphs into continuous text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) < 100:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle between -45 and 45
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Clamp to reasonable skew limit to prevent accidental 90 degree flips
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image by the given angle around its center with white border padding."""
    if abs(angle) < 0.2:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
    )
    return rotated


def segment_text_lines(
    gray_image: np.ndarray,
    min_line_height: int = 28,
    max_line_height: int = 240,
    smooth_sigma: float = 4.0,
    valley_threshold_factor: float = 0.08,
    vertical_padding: int = 12,
) -> list[tuple[int, int]]:
    """Segments horizontal text lines using smoothed projection profile valleys.

    Returns:
        List of (y_start, y_end) tuples in pixel coordinates.
    """
    h, w = gray_image.shape

    # Invert binary so text ink is 1 and paper is 0
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile: sum across row width
    profile = np.sum(binary, axis=1, dtype=np.float32) / 255.0

    # Gaussian smoothing of the profile to eliminate within-line diacritic dips
    ksize = int(round(smooth_sigma * 6)) | 1
    kernel = cv2.getGaussianKernel(ksize, smooth_sigma)
    smooth_profile = cv2.filter2D(profile, -1, kernel).flatten()

    peak_val = np.max(smooth_profile)
    if peak_val <= 0:
        return []

    threshold = peak_val * valley_threshold_factor
    is_text = smooth_profile > threshold

    # Find contiguous text bands
    line_spans: list[tuple[int, int]] = []
    in_text = False
    start_y = 0

    for y in range(h):
        if is_text[y] and not in_text:
            in_text = True
            start_y = y
        elif not is_text[y] and in_text:
            in_text = False
            line_height = y - start_y
            if min_line_height <= line_height <= max_line_height:
                # Add vertical safety padding for Khmer Coeng subscript & upper vowels
                pad_top = max(0, start_y - vertical_padding)
                pad_bottom = min(h, y + vertical_padding)
                line_spans.append((pad_top, pad_bottom))

    # Catch line at the very bottom edge
    if in_text:
        line_height = h - start_y
        if min_line_height <= line_height <= max_line_height:
            pad_top = max(0, start_y - vertical_padding)
            pad_bottom = min(h, h + vertical_padding)
            line_spans.append((pad_top, pad_bottom))

    # Merge overlapping boxes caused by padding
    merged_spans: list[tuple[int, int]] = []
    for span in line_spans:
        if not merged_spans:
            merged_spans.append(span)
        else:
            prev_start, prev_end = merged_spans[-1]
            curr_start, curr_end = span
            if curr_start < prev_end:
                merged_spans[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged_spans.append(span)

    return merged_spans


def process_writer_page(
    image_path: Path | str,
    writer_id: str,
    page_num: int = 1,
    output_base_dir: Path | str = REPO_ROOT / "data" / "handwritten_external",
    auto_deskew: bool = True,
    vertical_padding: int = 12,
) -> list[Path]:
    """Ingests a single scanned page, segments lines, and saves crops."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Scanned page image not found: {image_path}")

    writer_dir = Path(output_base_dir) / writer_id
    raw_dir = writer_dir / "raw"
    lines_dir = writer_dir / "lines_raw"
    audit_dir = writer_dir / "annotated"

    raw_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image with OpenCV: {image_path}")

    # Copy raw page to writer archive
    raw_target = raw_dir / f"{writer_id}_P{page_num:03d}.png"
    cv2.imwrite(str(raw_target), img_bgr)
    print(f"✓ Archived raw page to: {raw_target}")

    # Deskew
    if auto_deskew:
        angle = estimate_skew_angle(img_bgr)
        print(f"📐 Estimated skew angle: {angle:.2f}°")
        processed_img = deskew_image(img_bgr, angle)
    else:
        processed_img = img_bgr

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    # Segment lines
    spans = segment_text_lines(gray, vertical_padding=vertical_padding)
    print(f"✂️  Segmented {len(spans)} text lines from page {page_num}.")

    # Save line crops and create audit visual
    audit_img = processed_img.copy()
    saved_crops: list[Path] = []

    for idx, (y_start, y_end) in enumerate(spans, start=1):
        line_crop = processed_img[y_start:y_end, :]
        line_filename = f"{writer_id}_P{page_num:03d}_L{idx:03d}.png"
        crop_path = lines_dir / line_filename
        cv2.imwrite(str(crop_path), line_crop)
        saved_crops.append(crop_path)

        # Draw green bounding boxes and line index on audit sheet
        cv2.rectangle(audit_img, (0, y_start), (audit_img.shape[1], y_end), (0, 200, 0), 2)
        cv2.putText(
            audit_img,
            f"L{idx:03d} ({y_end - y_start}px)",
            (15, y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 150, 0),
            2,
        )

    # Save visual segmentation map
    audit_path = audit_dir / f"{writer_id}_P{page_num:03d}_segmentation_audit.png"
    cv2.imwrite(str(audit_path), audit_img)
    print(f"🔍 Saved visual segmentation map to: {audit_path}")
    print(f"🎉 Successfully saved {len(saved_crops)} line crops in {lines_dir}")

    return saved_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Add new writer scanned sheets and segment lines.")
    parser.add_argument("--writer-id", required=True, help="Writer identifier (e.g. W012)")
    parser.add_argument("--input-page", required=True, help="Path to scanned handwriting image file")
    parser.add_argument("--page-num", type=int, default=1, help="Page number (default: 1)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable automatic deskewing")
    parser.add_argument("--padding", type=int, default=12, help="Vertical line padding in pixels (default: 12)")
    args = parser.parse_args()

    process_writer_page(
        image_path=args.input_page,
        writer_id=args.writer_id,
        page_num=args.page_num,
        auto_deskew=not args.no_deskew,
        vertical_padding=args.padding,
    )


if __name__ == "__main__":
    main()

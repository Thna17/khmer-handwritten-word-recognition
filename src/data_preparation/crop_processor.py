"""Image cropping and preprocessing for raw and processed training candidates."""

import cv2
import numpy as np
import os
from typing import Dict, Any

def extract_raw_crop(img: np.ndarray, box: Dict[str, int]) -> np.ndarray:
    """Extracts unaltered raw image crop matching bounding box coordinates."""
    x1, y1 = box["x_min"], box["y_min"]
    x2, y2 = box["x_max"], box["y_max"]
    return img[y1:y2, x1:x2].copy()

def preprocess_training_crop(raw_crop: np.ndarray) -> np.ndarray:
    """
    Applies gentle training-friendly preprocessing:
    - Grayscale conversion
    - Mild contrast normalization (CLAHE with gentle clip limit 1.5)
    - Mild bilateral filtering to suppress background paper noise while preserving delicate Khmer strokes
    - Preserves original aspect ratio and geometry (no forced resizing)
    """
    if len(raw_crop.shape) == 3:
        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_crop.copy()

    # Gentle CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Mild bilateral filter (preserves sharp stroke edges and thin diacritics)
    filtered = cv2.bilateralFilter(enhanced, d=3, sigmaColor=20, sigmaSpace=20)
    return filtered

def save_crop_pair(
    img: np.ndarray,
    box: Dict[str, int],
    raw_path: str,
    processed_path: str
) -> bool:
    """Extracts and saves both raw and processed versions of a word crop."""
    raw = extract_raw_crop(img, box)
    if raw.size == 0:
        return False

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Save original raw crop (PNG lossless)
    cv2.imwrite(raw_path, raw)

    # Process and save training version
    processed = preprocess_training_crop(raw)
    cv2.imwrite(processed_path, processed)

    return True

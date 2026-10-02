"""Page preprocessing, ROI masking, and ink binarization."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

def load_and_validate_page(image_path: str) -> np.ndarray:
    """Loads an image from disk and validates its dimensions and channels."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image from {image_path}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {img.shape}")
    return img

def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimates document skew angle in degrees using Hough line transform on horizontal ruling/strokes.
    Returns angle in degrees (-45 to 45).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        coords = line.flatten()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 2 * abs(dy):  # Near horizontal
            angle = np.degrees(np.arctan2(dy, dx))
            if -10.0 < angle < 10.0:
                angles.append(angle)

    if not angles:
        return 0.0
    median_angle = float(np.median(angles))
    return median_angle

def deskew_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image around center by angle degrees with border replication or white padding."""
    if abs(angle) < 0.1:
        return img.copy()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_header_divider_y(gray_img: np.ndarray) -> int:
    """
    Detects the lowest major horizontal printed divider/box line in the top region (y < 380)
    separating printed header/forms from the handwriting area.
    """
    h, w = gray_img.shape[:2]
    blur = cv2.GaussianBlur(gray_img[:min(400, h), :], (3, 3), 0)
    _, bin_top = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect horizontal lines with width > 350
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horiz = cv2.morphologyEx(bin_top, cv2.MORPH_OPEN, kernel_h)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz)
    
    dividers = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if cw > 350 and ch < 25:
            dividers.append(y + ch)
            
    if dividers:
        return max(dividers)
    return 350

def extract_handwriting_roi(
    img: np.ndarray,
    top_y_start: int = None,
    bottom_y_end: int = None,
    left_x_start: int = None,
    right_x_end: int = None,
    writer_id: str = "W001"
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extracts the handwritten essay body region of interest (ROI),
    excluding printed header, form boxes, student names, signatures,
    vertical left margin watermark ('សៅ ម៉ាឌី'), and bottom page numbers.
    Supports writer-specific document dimensions (e.g. W001 vs W002).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if writer_id == "W0011":
        # Square notebook page with a large handwritten prompt box followed by
        # a title and essay body; writing reaches close to the bottom edge.
        if top_y_start is None:
            top_y_start = round(h * 0.015)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.98)
        if left_x_start is None:
            left_x_start = round(w * 0.02)
        if right_x_end is None:
            right_x_end = round(w * 0.99)
    elif writer_id == "W0010":
        # Structured essay form: remove the personal-information area and
        # retain the centered handwritten title plus the ruled essay body.
        if top_y_start is None:
            top_y_start = round(h * 0.325)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.97)
        if left_x_start is None:
            left_x_start = round(w * 0.025)
        if right_x_end is None:
            right_x_end = round(w * 0.97)
    elif writer_id == "W009":
        # Large full-page notebook scan: handwriting, including the boxed
        # prompt and final continuation line, reaches close to every edge.
        if top_y_start is None:
            top_y_start = round(h * 0.01)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.99)
        if left_x_start is None:
            left_x_start = round(w * 0.01)
        if right_x_end is None:
            right_x_end = round(w * 0.99)
    elif writer_id == "W008":
        # Compact squared-notebook page with a handwritten prompt inside a
        # printed box. Start below the faint form header/top border while
        # retaining both prompt baselines and the centered essay title.
        if top_y_start is None:
            top_y_start = round(h * 0.0625)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.95)
        if left_x_start is None:
            left_x_start = round(w * 0.04)
        if right_x_end is None:
            right_x_end = round(w * 0.96)
    elif writer_id == "W007":
        # Near-full-page notebook photograph: retain tall opening strokes while
        # excluding the centered page number and faint app watermark.
        if top_y_start is None:
            top_y_start = round(h * 0.015)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.955)
        if left_x_start is None:
            left_x_start = round(w * 0.02)
        if right_x_end is None:
            right_x_end = round(w * 0.98)
    elif writer_id == "W006":
        # Photographed notebook form: omit the faint printed header and the
        # handwritten date box, but retain the prompt and essay body.
        if top_y_start is None:
            top_y_start = round(h * 0.22)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.96)
        if left_x_start is None:
            left_x_start = round(w * 0.04)
        if right_x_end is None:
            right_x_end = round(w * 0.97)
    elif writer_id == "W005":
        # Ruled essay form: skip the printed name/title/header area while
        # preserving the first handwritten baseline.  The lower bound avoids
        # the centered printed page number and scanner border.
        if top_y_start is None:
            top_y_start = round(h * 0.14)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.94)
        if left_x_start is None:
            left_x_start = round(w * 0.04)
        if right_x_end is None:
            right_x_end = round(w * 0.96)
    elif writer_id == "W002":
        if top_y_start is None:
            top_y_start = 38
        if bottom_y_end is None:
            bottom_y_end = 985
        if left_x_start is None:
            left_x_start = 25
        if right_x_end is None:
            right_x_end = 700
    elif writer_id == "W001":
        if top_y_start is None:
            divider_y = detect_header_divider_y(gray)
            top_y_start = divider_y + 3
        if bottom_y_end is None:
            bottom_y_end = 1008
        if left_x_start is None:
            left_x_start = 89
        if right_x_end is None:
            right_x_end = 680
    else:
        # Generic public-page profile: retain nearly the full page while
        # excluding scanner borders and the usual bottom page number.  New
        # writers must never inherit W001's form-specific header crop.
        if top_y_start is None:
            top_y_start = round(h * 0.03)
        if bottom_y_end is None:
            bottom_y_end = round(h * 0.95)
        if left_x_start is None:
            left_x_start = round(w * 0.04)
        if right_x_end is None:
            right_x_end = round(w * 0.96)

    x1 = max(0, min(left_x_start, w - 1))
    x2 = max(x1 + 10, min(right_x_end, w))
    y1 = max(0, min(top_y_start, h - 1))
    y2 = max(y1 + 10, min(bottom_y_end, h))

    roi_img = img[y1:y2, x1:x2].copy()
    roi_coords = {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
    return roi_img, roi_coords

def binarize_ink(gray_roi: np.ndarray) -> np.ndarray:
    """
    Binarizes dark ink strokes against off-white lined paper.
    Returns binary mask where ink strokes are 255 and background is 0.
    """
    blur = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    # Otsu thresholding inverted (ink=255)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optional morphological cleanup to suppress tiny single-pixel scanner speckles
    clean_mask = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    return clean_mask

"""Line segmentation for handwritten Khmer text."""

import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class HandwrittenLine:
    line_index: int
    center_y: int
    y_min: int
    y_max: int
    components: List[Tuple[int, int, int, int, int]]  # (x, y, w, h, area)


def line_bounding_box(
    line: HandwrittenLine,
    image_shape: Tuple[int, int],
    padding_x: int = 8,
    padding_y: int = 5,
) -> Dict[str, int]:
    """Return a padded crop box containing all components in one line.

    ``image_shape`` is ``(height, width)``.  Clamping makes the function safe
    for lines touching page edges while preserving ascenders and descenders.
    """
    if not line.components:
        raise ValueError("cannot build a bounding box for a line without components")
    image_h, image_w = image_shape
    x_min = min(component[0] for component in line.components)
    x_max = max(component[0] + component[2] for component in line.components)
    return {
        "x_min": max(0, x_min - padding_x),
        "y_min": max(0, line.y_min - padding_y),
        "x_max": min(image_w, x_max + padding_x),
        "y_max": min(image_h, line.y_max + padding_y),
    }


def line_bounding_boxes(
    lines: List[HandwrittenLine],
    image_shape: Tuple[int, int],
    roi_y_min: int,
    roi_y_max: int,
    padding_x: int = 8,
    boundary_overlap: int = 2,
) -> List[Dict[str, int]]:
    """Create non-destructive line bands separated at center midpoints.

    Component extents can cross into adjacent Khmer lines because tall vowels
    and diacritics overlap vertically.  Midpoints between detected baselines
    provide stable boundaries; a tiny overlap protects edge strokes without
    merging two full lines into one training crop.
    """
    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: item.center_y)
    boxes: List[Dict[str, int]] = []
    for index, line in enumerate(ordered):
        top = (
            max(roi_y_min, line.y_min)
            if index == 0
            else (ordered[index - 1].center_y + line.center_y) // 2
        )
        bottom = (
            min(roi_y_max, line.y_max)
            if index == len(ordered) - 1
            else (line.center_y + ordered[index + 1].center_y) // 2
        )
        box = line_bounding_box(line, image_shape, padding_x=padding_x, padding_y=0)
        box["y_min"] = max(0, top - boundary_overlap)
        box["y_max"] = min(image_shape[0], bottom + boundary_overlap)
        boxes.append(box)
    return boxes

def segment_lines(
    bin_img: np.ndarray,
    roi_coords: Dict[str, int],
    min_peak_distance: int = 18,
    peak_prominence: int = 20
) -> List[HandwrittenLine]:
    """
    Segments handwritten text lines using horizontal projection profile
    and Connected Component grouping to prevent cutting ascenders and descenders.
    """
    x1, y1 = roi_coords["x_min"], roi_coords["y_min"]
    x2, y2 = roi_coords["x_max"], roi_coords["y_max"]
    roi_bin = bin_img[y1:y2, x1:x2]

    # Calculate horizontal projection profile of ink
    proj = np.sum(roi_bin, axis=1) / 255.0
    smoothed = ndimage.gaussian_filter1d(proj, sigma=2.0)
    peaks, _ = find_peaks(smoothed, distance=min_peak_distance, prominence=peak_prominence)

    if len(peaks) == 0:
        return []

    line_centers = [y1 + int(p) for p in peaks]

    # Extract connected components from ROI
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_bin)

    line_comps: Dict[int, List[Tuple[int, int, int, int, int]]] = {
        i: [] for i in range(len(line_centers))
    }

    for i in range(1, num_labels):
        cx_rel, cy_rel, cw, ch, carea = stats[i]
        
        # Filter out tiny noise specks
        if carea < 6:
            continue
            
        abs_x = x1 + cx_rel
        abs_y = y1 + cy_rel
        
        # Filter out long horizontal ruling lines or box borders (w > 180 and h < 8)
        if cw > 180 and ch < 8:
            continue
            
        # Filter out vertical margin lines or multi-line spanning ink bridges (ch > 55)
        if ch > 55:
            continue
            
        # Filter out bottom-right circled page marker ① or ② (y > 960 and x > 655)
        if abs_y > 960 and abs_x > 655:
            continue

        comp_cy = abs_y + ch / 2.0
        # Assign to nearest line center
        dists = [abs(comp_cy - lc) for lc in line_centers]
        best_line = int(np.argmin(dists))

        line_comps[best_line].append((abs_x, abs_y, cw, ch, carea))

    lines: List[HandwrittenLine] = []
    for idx, center in enumerate(line_centers):
        comps = line_comps[idx]
        if not comps:
            continue

        comp_y1s = [c[1] for c in comps]
        comp_y2s = [c[1] + c[3] for c in comps]

        min_y = max(0, min(comp_y1s))
        max_y = min(bin_img.shape[0], max(comp_y2s))

        lines.append(HandwrittenLine(
            line_index=idx,
            center_y=center,
            y_min=min_y,
            y_max=max_y,
            components=comps
        ))

    return lines

"""Word segmentation module for handwritten Khmer lines."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from .line_segmentation import HandwrittenLine

def segment_words_in_line(
    line: HandwrittenLine,
    bin_img: np.ndarray,
    page_shape: Tuple[int, int],
    h_padding: int = 8,
    v_padding: int = 6
) -> List[Dict[str, Any]]:
    """
    Segments a handwritten line into candidate short-word bounding boxes.
    
    Adheres strictly to Khmer script characteristics:
    - Clusters overlapping/connected components (subscripts, vowels, base consonants).
    - Preserves safe whitespace margins (5-15px).
    - Avoids slicing through strokes, diacritics, or coeng components.
    """
    page_h, page_w = page_shape[:2]
    comps = sorted(line.components, key=lambda c: c[0])
    if not comps:
        return []

    # 1. Cluster overlapping components horizontally (intra-syllable distance <= 3px)
    clusters = []
    for c in comps:
        cx1, cy1, cw, ch, carea = c
        cx2 = cx1 + cw
        cy2 = cy1 + ch
        if not clusters:
            clusters.append([cx1, cy1, cx2, cy2, [c]])
        else:
            last = clusters[-1]
            if cx1 <= last[2] + 3:
                last[0] = min(last[0], cx1)
                last[1] = min(last[1], cy1)
                last[2] = max(last[2], cx2)
                last[3] = max(last[3], cy2)
                last[4].append(c)
            else:
                clusters.append([cx1, cy1, cx2, cy2, [c]])

    # 2. Inspect clusters: If a cluster is wide (> 125px), check for clean vertical valleys
    raw_candidates: List[Tuple[int, int, int, int]] = []
    for cl in clusters:
        cl_x1, cl_y1, cl_x2, cl_y2, cl_comps = cl
        width = cl_x2 - cl_x1
        
        if width > 125:
            # Check vertical projection inside cluster
            cl_mask = bin_img[cl_y1:cl_y2, cl_x1:cl_x2]
            v_proj = np.sum(cl_mask, axis=0) / 255.0
            
            splits = [0]
            in_gap = False
            gap_start = 0
            for col_idx, val in enumerate(v_proj):
                if val == 0:
                    if not in_gap:
                        in_gap = True
                        gap_start = col_idx
                else:
                    if in_gap:
                        gap_len = col_idx - gap_start
                        if gap_len >= 3 and (gap_start - splits[-1]) >= 28 and (width - col_idx) >= 28:
                            splits.append(gap_start + gap_len // 2)
                        in_gap = False
            splits.append(width)
            
            for s_idx in range(len(splits) - 1):
                sub_x1 = cl_x1 + splits[s_idx]
                sub_x2 = cl_x1 + splits[s_idx + 1]
                sub_mask = bin_img[cl_y1:cl_y2, sub_x1:sub_x2]
                cols = np.where(np.sum(sub_mask, axis=0) > 0)[0]
                rows = np.where(np.sum(sub_mask, axis=1) > 0)[0]
                if len(cols) > 0 and len(rows) > 0:
                    actual_x1 = sub_x1 + cols[0]
                    actual_x2 = sub_x1 + cols[-1] + 1
                    actual_y1 = cl_y1 + rows[0]
                    actual_y2 = cl_y1 + rows[-1] + 1
                    raw_candidates.append((actual_x1, actual_y1, actual_x2, actual_y2))
        else:
            raw_candidates.append((cl_x1, cl_y1, cl_x2, cl_y2))

    # 3. Add safe whitespace padding and evaluate segmentation confidence
    candidates: List[Dict[str, Any]] = []
    for idx, (bx1, by1, bx2, by2) in enumerate(raw_candidates):
        # Calculate safe margins without extending beyond image boundaries
        pad_x1 = max(0, bx1 - h_padding)
        pad_y1 = max(0, by1 - v_padding)
        pad_x2 = min(page_w, bx2 + h_padding)
        pad_y2 = min(page_h, by2 + v_padding)

        cw = bx2 - bx1
        ch = by2 - by1

        # Check neighbor clearance
        prev_clearance = bx1 - (raw_candidates[idx - 1][2]) if idx > 0 else 50
        next_clearance = (raw_candidates[idx + 1][0]) - bx2 if idx + 1 < len(raw_candidates) else 50

        # Confidence heuristic
        conf = 0.95
        review_req = False
        notes = []

        if cw < 18 or ch < 12:
            conf -= 0.15
            notes.append("narrow component")
        if prev_clearance < 4 or next_clearance < 4:
            conf -= 0.10
            notes.append("tight spacing with neighbor")
        if width > 160:
            conf -= 0.10
            notes.append("compound word or continuous phrase")

        if conf < 0.85:
            review_req = True

        candidates.append({
            "line_index": line.line_index,
            "x_min": pad_x1,
            "y_min": pad_y1,
            "x_max": pad_x2,
            "y_max": pad_y2,
            "ink_x_min": bx1,
            "ink_y_min": by1,
            "ink_x_max": bx2,
            "ink_y_max": by2,
            "segmentation_confidence": round(conf, 2),
            "review_required": review_req,
            "notes": "; ".join(notes)
        })

    return candidates

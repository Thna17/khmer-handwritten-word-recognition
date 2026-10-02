"""Visual annotation generator for human verification of crops."""

import cv2
import numpy as np
import os
from typing import List, Dict, Any

def create_annotated_page(
    source_img: np.ndarray,
    crops_metadata: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Generates an annotated preview of the original page with:
    - Bounding rectangle around every candidate word crop
    - Numeric crop ID (e.g. '0001', '0002') positioned cleanly above the box
    - Labels never obscure the handwriting
    """
    annotated = source_img.copy()
    h_page, w_page = annotated.shape[:2]

    for crop in crops_metadata:
        x1, y1 = crop["x_min"], crop["y_min"]
        x2, y2 = crop["x_max"], crop["y_max"]
        crop_id = crop["crop_id"]
        
        # Extract 4-digit numeric ID (e.g. 'W001_P001_0042' -> '0042')
        num_str = crop_id.split("_")[-1]

        # Determine box color based on review requirement
        if crop.get("review_required", False):
            box_color = (0, 140, 255)  # Orange/amber for review required
        else:
            box_color = (0, 180, 0)    # Green for high confidence

        # Draw crisp bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)

        # Place label above rectangle if space permits, else below
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.32
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(num_str, font, font_scale, thickness)

        if y1 - th - 3 > 5:
            text_y = y1 - 2
            bg_y1 = y1 - th - 4
            bg_y2 = y1
        else:
            text_y = y2 + th + 2
            bg_y1 = y2
            bg_y2 = y2 + th + 4

        text_x = max(0, min(x1, w_page - tw))
        bg_x2 = min(w_page, text_x + tw + 2)

        # Draw small background pill to guarantee label readability without hiding strokes
        cv2.rectangle(annotated, (text_x, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.putText(annotated, num_str, (text_x + 1, text_y), font, font_scale, (0, 0, 180), thickness, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path

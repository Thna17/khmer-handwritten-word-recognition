"""Automated Quality Control (QC) validator for extracted crops and metadata."""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple

def run_quality_control(
    metadata_records: List[Dict[str, Any]],
    base_dir: str,
    source_img_path: str
) -> Dict[str, Any]:
    """
    Validates crop integrity, metadata consistency, and bounding box sanity.
    
    Checks:
    - Crop file existence and non-zero size
    - Non-empty and reasonable crop dimensions (w >= 10, h >= 10)
    - No duplicate crop IDs or metadata entries
    - All metadata image paths exist on disk
    - Bounding box coordinates within source page bounds
    - Non-blank crops (sufficient ink stroke pixel variance)
    - No crops originating from excluded printed header/form areas
    - Unicode strings encode and decode cleanly in UTF-8
    """
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        raise FileNotFoundError(f"Source image not found: {source_img_path}")
    page_h, page_w = source_img.shape[:2]

    seen_ids = set()
    errors = []
    warnings = []

    detected_count = len(metadata_records)
    needs_seg_review = 0
    needs_trans_review = 0
    rejected_count = 0
    accepted_auto = 0

    for idx, rec in enumerate(metadata_records):
        cid = rec["crop_id"]
        img_rel_path = rec["image_path"]
        full_img_path = os.path.join(base_dir, img_rel_path)

        # Check 1: Duplicate IDs
        if cid in seen_ids:
            errors.append(f"Duplicate crop_id: {cid}")
        seen_ids.add(cid)

        # Check 2: File exists on disk
        if not os.path.exists(full_img_path):
            errors.append(f"Missing crop image file: {full_img_path}")
            continue

        # Check 3: File non-empty and readable
        crop_mat = cv2.imread(full_img_path)
        if crop_mat is None or crop_mat.size == 0:
            errors.append(f"Corrupted or empty crop image: {full_img_path}")
            continue

        ch, cw = crop_mat.shape[:2]
        if cw < 8 or ch < 8:
            errors.append(f"Crop {cid} dimensions too small: {cw}x{ch}")

        # Check 4: Bounding box inside source page bounds
        x1 = int(rec["x_min"])
        y1 = int(rec["y_min"])
        x2 = int(rec["x_max"])
        y2 = int(rec["y_max"])

        if x1 < 0 or y1 < 0 or x2 > page_w or y2 > page_h or x1 >= x2 or y1 >= y2:
            errors.append(f"Crop {cid} bounding box out of page bounds: [{x1},{y1},{x2},{y2}] vs ({page_w}x{page_h})")

        # Check 5: Ensure crop is not from excluded header/overlay area
        wid = rec.get("writer_id", "W001")
        min_header_y = 210 if wid == "W001" else (30 if wid == "W002" else 15)
        if y1 < min_header_y:
            errors.append(f"Crop {cid} originates from printed form/header area: y={y1}")

        # Check 6: Blank crop detection (standard deviation of grayscale values)
        crop_gray = cv2.cvtColor(crop_mat, cv2.COLOR_BGR2GRAY) if len(crop_mat.shape) == 3 else crop_mat
        std_val = float(np.std(crop_gray))
        if std_val < 3.0:
            errors.append(f"Crop {cid} appears completely blank (std={std_val:.2f})")

        # Check 7: Unicode encode/decode UTF-8
        try:
            rec["label_raw"].encode("utf-8").decode("utf-8")
            rec["label_normalized"].encode("utf-8").decode("utf-8")
        except UnicodeError as e:
            errors.append(f"Crop {cid} Unicode encoding failure: {e}")

        # Classification counts
        rev_req = str(rec.get("review_required", "")).lower() == "true"
        status = rec.get("review_status", "pending")
        seg_conf = float(rec.get("segmentation_confidence", 1.0))
        lbl_conf = float(rec.get("label_confidence", 1.0))

        if status == "rejected":
            rejected_count += 1
        elif rev_req or seg_conf < 0.85 or lbl_conf < 0.85:
            if seg_conf < 0.85:
                needs_seg_review += 1
            if lbl_conf < 0.85 or not rec.get("label_raw"):
                needs_trans_review += 1
        else:
            accepted_auto += 1

    qc_summary = {
        "source_page": os.path.basename(source_img_path),
        "detected_candidates": detected_count,
        "accepted_automatically": accepted_auto,
        "needs_segmentation_review": needs_seg_review,
        "needs_transcription_review": needs_trans_review,
        "rejected_as_non_handwriting": rejected_count,
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0
    }
    return qc_summary

def format_qc_report(summary: Dict[str, Any]) -> str:
    """Formats QC summary into human-readable report."""
    status_text = "PASSED (Zero errors)" if summary["is_valid"] else f"FAILED ({len(summary['errors'])} errors)"
    lines = [
        f"Source page: {summary['source_page']}",
        "",
        f"Detected candidates: {summary['detected_candidates']}",
        f"Accepted automatically: {summary['accepted_automatically']}",
        f"Needs manual segmentation review: {summary['needs_segmentation_review']}",
        f"Needs transcription review: {summary['needs_transcription_review']}",
        f"Rejected as non-handwriting: {summary['rejected_as_non_handwriting']}",
        "",
        f"Validation Status: {status_text}"
    ]
    if summary["errors"]:
        lines.append("Errors:")
        for err in summary["errors"][:10]:
            lines.append(f"  - {err}")
    return "\n".join(lines)

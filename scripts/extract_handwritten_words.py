#!/usr/bin/env python3
"""
Master CLI for Khmer handwritten line/short-text extraction.

Line mode is the project default.  The old word mode remains available only
for reproducing legacy artifacts; natural Khmer handwriting does not expose
reliable visual word boundaries.

Usage:
    python scripts/extract_handwritten_words.py \
        --input data/handwritten_external/W001/source/page_001.png \
        --writer-id W001 \
        --output data/handwritten_external/W001/ \
        --mode line
"""

import argparse
import os
import sys
import cv2

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preparation.page_preprocessing import (
    load_and_validate_page,
    estimate_skew_angle,
    deskew_image,
    extract_handwriting_roi,
    binarize_ink
)
from src.data_preparation.line_segmentation import line_bounding_boxes, segment_lines
from src.data_preparation.word_segmentation import segment_words_in_line
from src.data_preparation.crop_processor import save_crop_pair
from src.data_preparation.transcription import assign_transcription
from src.data_preparation.metadata import write_metadata_csv
from src.data_preparation.annotated_visualizer import create_annotated_page
from src.data_preparation.quality_control import run_quality_control, format_qc_report

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Khmer handwritten lines or legacy word candidates.")
    parser.add_argument("--input", required=True, help="Path to input source page image.")
    parser.add_argument("--writer-id", default="W001", help="Anonymous writer ID (e.g. W001).")
    parser.add_argument("--output", required=True, help="Target output directory (e.g. data/handwritten_external/W001/).")
    parser.add_argument(
        "--mode",
        choices=("line", "word"),
        default="line",
        help="line is the project default; word is retained for legacy reproduction",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    writer_id = args.writer_id
    output_dir = os.path.abspath(args.output)

    print(f"=== Starting Khmer {args.mode.title()} Extraction Pipeline ===")
    print(f"Source image: {input_path}")
    print(f"Writer ID:    {writer_id}")
    print(f"Output dir:   {output_dir}")

    # Create directory structure
    source_dir = os.path.join(output_dir, "source")
    annotated_dir = os.path.join(output_dir, "annotated")
    raw_dir_name = "lines_raw" if args.mode == "line" else "crops_raw"
    proc_dir_name = "lines_processed" if args.mode == "line" else "crops_processed"
    crops_raw_dir = os.path.join(output_dir, raw_dir_name)
    crops_proc_dir = os.path.join(output_dir, proc_dir_name)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(crops_raw_dir, exist_ok=True)
    os.makedirs(crops_proc_dir, exist_ok=True)

    # 1. Load image
    source_img = load_and_validate_page(input_path)
    page_filename = os.path.basename(input_path)
    
    # Save a clean copy to output source/ if not already there
    target_source_path = os.path.join(source_dir, page_filename)
    if not os.path.exists(target_source_path) or os.path.abspath(target_source_path) != input_path:
        cv2.imwrite(target_source_path, source_img)

    # 2. Skew detection
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    skew_angle = estimate_skew_angle(gray)
    print(f"Detected document skew angle: {skew_angle:.2f} degrees")
    if abs(skew_angle) > 0.5:
        print(f"Applying slight deskew ({skew_angle:.2f}°)...")
        source_img = deskew_image(source_img, skew_angle)
        gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    # 3. Extract handwriting ROI & binarize ink
    roi_overrides = {}
    page_stem = os.path.splitext(page_filename)[0]
    if writer_id == "W0010" and page_stem != "page_001":
        # Continuation sheet replaces the personal-information header with a
        # large printed image placeholder. Start immediately below that panel.
        roi_overrides["top_y_start"] = round(source_img.shape[0] * 0.27)
        if page_stem == "page_003":
            # The essay ends in the upper half; exclude the remaining blank
            # dotted form rows so they cannot create projection peaks.
            roi_overrides["bottom_y_end"] = round(source_img.shape[0] * 0.61)
    elif writer_id == "W005" and page_stem != "page_001":
        # W005 continuation sheets use printed panels of different heights.
        # Start below each known panel while retaining the first handwritten
        # baseline; unreviewed continuation layouts keep the page-2 profile.
        top_ratios = {
            "page_002": 0.17,
            "page_003": 0.245,
            "page_004": 0.235,
            "page_005": 0.25,
        }
        top_ratio = top_ratios.get(page_stem, 0.17)
        roi_overrides["top_y_start"] = round(source_img.shape[0] * top_ratio)
        if page_stem == "page_005":
            # This page ends early and leaves many blank ruled rows; stop just
            # below the handwriting to exclude those rules and the page number.
            roi_overrides["bottom_y_end"] = round(source_img.shape[0] * 0.68)
    elif writer_id == "W008" and page_stem != "page_001":
        # Continuation sheets omit the boxed prompt and begin close to the
        # top edge; retain the first baseline while skipping the faint form
        # header. The writer-level bottom bound still excludes the watermark.
        roi_overrides["top_y_start"] = round(source_img.shape[0] * 0.03)
        if page_stem == "page_003":
            # This final sheet has only a short Khmer continuation; exclude
            # both the faint form header and the unrelated English source note
            # in the otherwise blank area.
            roi_overrides["top_y_start"] = round(source_img.shape[0] * 0.12)
            roi_overrides["bottom_y_end"] = round(source_img.shape[0] * 0.47)
    elif writer_id == "W006" and page_stem != "page_001":
        # Continuation pages have no date box, but their faint printed headers
        # have different heights.
        top_ratios = {"page_002": 0.10, "page_003": 0.145}
        roi_overrides["top_y_start"] = round(
            source_img.shape[0] * top_ratios.get(page_stem, 0.10)
        )
        if page_stem == "page_003":
            roi_overrides["bottom_y_end"] = round(source_img.shape[0] * 0.98)
    roi_img, roi_coords = extract_handwriting_roi(
        source_img,
        writer_id=writer_id,
        **roi_overrides,
    )
    bin_img = binarize_ink(gray)
    print(f"Handwriting ROI: x=[{roi_coords['x_min']}, {roi_coords['x_max']}], y=[{roi_coords['y_min']}, {roi_coords['y_max']}]")

    # 4. Line segmentation
    line_peak_distances = {
        "W007": 40,
        "W008": 38,
        "W009": 40,
        "W0010": 36,
        "W0011": 45,
    }
    line_peak_distance = line_peak_distances.get(writer_id, 18)
    lines = segment_lines(
        bin_img,
        roi_coords,
        min_peak_distance=line_peak_distance,
    )
    print(f"Segmented {len(lines)} handwritten text lines.")

    # 5. Line or legacy word-candidate cropping
    metadata_records = []
    crop_counter = 1

    # Extract page identifier code (e.g. 'P001' from 'page_001.png')
    base_name_no_ext = os.path.splitext(page_filename)[0]
    import re
    page_digits = re.findall(r'\d+', base_name_no_ext)
    page_code = f"P{int(page_digits[0]):03d}" if page_digits else "P001"

    line_boxes = line_bounding_boxes(
        lines,
        source_img.shape[:2],
        roi_y_min=roi_coords["y_min"],
        roi_y_max=roi_coords["y_max"],
    )
    if writer_id == "W005":
        # The ruled form contains touching Khmer strokes and faint ink that can
        # fragment connected components.  Preserve each complete writing line
        # by using the validated form width instead of component x-extents.
        for box in line_boxes:
            box["x_min"] = roi_coords["x_min"]
            box["x_max"] = roi_coords["x_max"]
    elif writer_id in {"W007", "W008", "W009", "W0010", "W0011"}:
        # Preserve long right-leaning or faint text while retaining adaptive
        # left edges that avoid photographed margins and fingers.
        for box in line_boxes:
            box["x_max"] = roi_coords["x_max"]
            if writer_id == "W0011" and page_stem != "page_001":
                # Continuation sheets have a stable text margin and no prompt
                # border; a fixed left edge protects faint opening characters.
                box["x_min"] = max(
                    roi_coords["x_min"], round(source_img.shape[1] * 0.09)
                )
    for line_number, line in enumerate(lines, start=1):
        if args.mode == "line":
            box = line_boxes[line_number - 1]
            crop_id = f"{writer_id}_{page_code}_L{line_number:03d}"
            raw_rel_path = f"{raw_dir_name}/{crop_id}.png"
            proc_rel_path = f"{proc_dir_name}/{crop_id}.png"
            save_crop_pair(
                source_img,
                box,
                os.path.join(output_dir, raw_rel_path),
                os.path.join(output_dir, proc_rel_path),
            )
            line_note = "Line crop; human transcription required"
            if writer_id == "W002" and line_number == 1 and box["y_min"] <= roi_coords["y_min"] + 2:
                line_note += "; inspect possible header or source-overlay contamination"
            metadata_records.append({
                "crop_id": crop_id,
                "image_path": raw_rel_path,
                "source_page": page_filename,
                "writer_id": writer_id,
                "x_min": box["x_min"],
                "y_min": box["y_min"],
                "x_max": box["x_max"],
                "y_max": box["y_max"],
                "label_raw": "",
                "label_normalized": "",
                "label_confidence": 0.0,
                "segmentation_confidence": 0.95,
                "review_required": True,
                "review_status": "pending",
                "notes": line_note,
            })
            continue

        candidates = segment_words_in_line(line, bin_img, source_img.shape[:2])
        # Filter out tiny isolated noise (less than 10x10)
        filtered_cands = []
        for c in candidates:
            cw = c["ink_x_max"] - c["ink_x_min"]
            ch = c["ink_y_max"] - c["ink_y_min"]
            if cw < 10 and ch < 10:
                continue
            filtered_cands.append(c)

        for cand_idx, cand in enumerate(filtered_cands):
            crop_id = f"{writer_id}_{page_code}_{crop_counter:04d}"
            raw_rel_path = f"{raw_dir_name}/{crop_id}.png"
            proc_rel_path = f"{proc_dir_name}/{crop_id}.png"

            raw_abs_path = os.path.join(output_dir, raw_rel_path)
            proc_abs_path = os.path.join(output_dir, proc_rel_path)

            # Extract and save crop pair
            save_crop_pair(source_img, cand, raw_abs_path, proc_abs_path)

            # Assign transcription
            trans = assign_transcription(page_filename, line.line_index, cand_idx, len(filtered_cands), cand, writer_id=writer_id)

            rec = {
                "crop_id": crop_id,
                "image_path": raw_rel_path,
                "source_page": page_filename,
                "writer_id": writer_id,
                "x_min": cand["x_min"],
                "y_min": cand["y_min"],
                "x_max": cand["x_max"],
                "y_max": cand["y_max"],
                "label_raw": trans["label_raw"],
                "label_normalized": trans["label_normalized"],
                "label_confidence": trans["label_confidence"],
                "segmentation_confidence": cand["segmentation_confidence"],
                "review_required": trans["review_required"],
                "review_status": trans["review_status"],
                "notes": trans["notes"]
            }
            metadata_records.append(rec)
            crop_counter += 1

    # 6. Save metadata.csv (merge with existing pages if present)
    metadata_filename = "line_metadata.csv" if args.mode == "line" else "metadata.csv"
    metadata_path = os.path.join(output_dir, metadata_filename)
    from src.data_preparation.metadata import read_metadata_csv
    all_records = []
    if os.path.exists(metadata_path):
        try:
            prior_records = read_metadata_csv(metadata_path)
            # Retain records from other pages
            all_records = [r for r in prior_records if r.get("source_page") != page_filename]
        except Exception:
            all_records = []
    all_records.extend(metadata_records)
    write_metadata_csv(all_records, metadata_path)
    print(f"Saved metadata to {metadata_path} ({len(metadata_records)} rows for {page_filename}, total dataset rows: {len(all_records)}).")

    # 7. Generate annotated preview image
    suffix = "lines_annotated" if args.mode == "line" else "annotated"
    annotated_filename = f"{os.path.splitext(page_filename)[0]}_{suffix}.png"
    annotated_path = os.path.join(annotated_dir, annotated_filename)
    create_annotated_page(source_img, metadata_records, annotated_path)
    print(f"Saved annotated preview to {annotated_path}")

    # 8. Run Quality Control checks
    qc_summary = run_quality_control(metadata_records, output_dir, target_source_path)
    print("\n" + "=" * 50)
    print("QUALITY CONTROL SUMMARY")
    print("=" * 50)
    print(format_qc_report(qc_summary))
    print("=" * 50)

if __name__ == "__main__":
    main()

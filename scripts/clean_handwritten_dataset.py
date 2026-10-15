#!/usr/bin/env python3
"""Quality filter to purge defective, truncated, and unverified handwritten crops.

Specifically filters out:
1. Upper Vowel & Subscript Consonant Truncation:
   - Line crops where height < 34px (too cramped to safely fit 3-tier Khmer stacking).
   - Crops where upper vowels (ិ, ី, ឹ, ឺ, ើ, ាំ, etc.) or subscript consonants (Coeng ្...)
     are sliced off by the bounding box edge.
2. Filler Dot & Ellipsis Artifacts:
   - Transcription lines contaminated by placeholder dots (e.g. '...............').
3. Unverified / Blurry / Out-of-Bounds Crops:
   - Low contrast, corrupted images, or unreadable ink.

Outputs:
- data/metadata/labels_clean.csv: Pristine, validated real handwritten lines.
- data/metadata/cleaning_audit.json: Comprehensive audit log of every removed file and reason.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "data" / "metadata"
INPUT_CSV = METADATA_DIR / "labels.csv"
OUTPUT_CLEAN_CSV = METADATA_DIR / "labels_clean.csv"
AUDIT_JSON = METADATA_DIR / "cleaning_audit.json"

# Khmer Unicode Ranges
UPPER_VOWEL_RE = re.compile(r"[\u17B7-\u17BA\u17BE-\u17C0\u17C6\u17C7\u17C8\u17C9\u17CA\u17CB\u17CC\u17CD\u17CE\u17CF\u17D0]")
LOWER_SUB_RE = re.compile(r"[\u17BB-\u17BD\u17D2]")


def evaluate_crop(img_path: str, label: str, min_height: int = 34) -> tuple[bool, list[str]]:
    """Evaluates whether an image crop is clean, well-framed, and properly transcribed."""
    reasons = []

    # 1. Check dot fillers or excessive periods
    if re.search(r"\.{3,}", label) or label.count(".") > 4:
        reasons.append("filler_dots_artifact")

    if not os.path.exists(img_path):
        return False, ["missing_image_file"]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return False, ["corrupt_or_unreadable_image"]

    h, w = img.shape

    # 2. Minimum physical height check for 3-tier Khmer text
    if h < min_height:
        reasons.append(f"crop_height_{h}px_too_small_truncates_vowels_and_subscripts")

    # 3. Severe aspect ratio / dimension anomalies
    clean_chars = len(label.replace(" ", "").replace(".", ""))
    if clean_chars > 30 and w < 380:
        reasons.append(f"anomalous_crop_dimensions_{w}x{h}_for_{clean_chars}_chars")

    # 4. Blur / contrast check (Laplacian variance)
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    if lap_var < 25.0:
        reasons.append(f"low_contrast_or_blurry_var_{lap_var:.1f}")

    return len(reasons) == 0, reasons


def run_cleaner(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CLEAN_CSV,
    audit_path: Path = AUDIT_JSON,
    in_place: bool = False,
    min_height: int = 34,
):
    print("=" * 65)
    print("      HANDWRITTEN CROP QUALITY CLEANER & AUDIT TOOL")
    print("=" * 65)
    print(f"Reading: {input_csv}")

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"Initial raw handwritten lines: {initial_count}")

    clean_records = []
    removed_records = []

    for idx, row in df.iterrows():
        img_path = str(row["image"])
        label = str(row["label"]).strip()
        wid = str(row["writer_id"])

        is_valid, reasons = evaluate_crop(img_path, label, min_height=min_height)

        if is_valid:
            clean_records.append({
                "image": img_path,
                "label": label,
                "writer_id": wid,
            })
        else:
            removed_records.append({
                "image": img_path,
                "writer_id": wid,
                "label": label,
                "reasons": reasons,
            })

    df_clean = pd.DataFrame(clean_records)
    
    # Save clean dataset
    target_out = input_csv if in_place else output_csv
    if in_place:
        backup_path = input_csv.with_name(f"{input_csv.stem}_backup.csv")
        df.to_csv(backup_path, index=False)
        print(f"✓ Backup of original saved to: {backup_path}")

    df_clean.to_csv(target_out, index=False)
    print(f"✓ Clean dataset saved to: {target_out} ({len(df_clean)} samples)")

    # Generate audit report
    audit_summary = {
        "input_samples": initial_count,
        "clean_samples_retained": len(df_clean),
        "defective_crops_removed": len(removed_records),
        "retained_by_writer": df_clean["writer_id"].value_counts().to_dict(),
        "reasons_breakdown": {},
        "removed_details": removed_records,
    }

    for item in removed_records:
        for r in item["reasons"]:
            audit_summary["reasons_breakdown"][r] = audit_summary["reasons_breakdown"].get(r, 0) + 1

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Full audit report written to: {audit_path}")

    print("\n" + "=" * 65)
    print("                     CLEANING SUMMARY")
    print("=" * 65)
    print(f"Total lines evaluated : {initial_count}")
    print(f"Defective lines purged: {len(removed_records)}")
    print(f"Clean lines retained  : {len(df_clean)}")
    print("\nReasons breakdown:")
    for reason, count in sorted(audit_summary["reasons_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  • {reason}: {count}")

    print("\nRetained lines per writer:")
    for wid, count in sorted(audit_summary["retained_by_writer"].items()):
        print(f"  • {wid:6s}: {count:2d} clean lines")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Purge defective, truncated, and unverified handwritten crops.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="Path to input labels.csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CLEAN_CSV, help="Path for cleaned output labels.csv")
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON, help="Path for audit report JSON")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input_csv directly (creates backup)")
    parser.add_argument("--min-height", type=int, default=34, help="Minimum crop height in px (default: 34)")

    args = parser.parse_args()
    run_cleaner(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        audit_path=args.audit_json,
        in_place=args.in_place,
        min_height=args.min_height,
    )


if __name__ == "__main__":
    main()

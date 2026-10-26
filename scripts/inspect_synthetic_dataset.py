#!/usr/bin/env python3
"""Synthetic Dataset Inspection & Quality Audit Tool.

Inspects generated synthetic line datasets (e.g. data/synthetic_100k/):
1. Verifies dataset integrity and labels.csv formatting.
2. Checks 100% Unicode token coverage against char_to_idx.json.
3. Generates a composite visual audit sheet showing random line crops with
   overlaid ground truth inscriptions to verify text shaping and pen physics.

Usage:
  python scripts/inspect_synthetic_dataset.py \
    --dataset-dir data/synthetic_100k \
    --num-samples 10 \
    --output-viz results/synthetic_inspection_preview.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_khmer_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/Library/Fonts/NotoSansKhmer-Bold.ttf",
        "/Library/Fonts/KhmerOS.ttf",
        str(REPO_ROOT / "assets" / "fonts" / "NotoSansKhmer-Regular.ttf"),
        str(REPO_ROOT / "assets" / "fonts" / "KhmerOS.ttf"),
    ]
    for p in font_candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def inspect_dataset(
    dataset_dir: Path,
    vocab_path: Path = REPO_ROOT / "data" / "metadata" / "char_to_idx.json",
    num_samples: int = 10,
    output_viz: Path = REPO_ROOT / "results" / "synthetic_inspection_preview.png",
) -> None:
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.is_file():
        # Fallback to synthetic_labels.csv if present
        if (dataset_dir / "synthetic_labels.csv").is_file():
            labels_csv = dataset_dir / "synthetic_labels.csv"
        else:
            raise FileNotFoundError(f"No labels.csv found in {dataset_dir}")

    print("=" * 65)
    print(f"      INSPECTING SYNTHETIC DATASET: {dataset_dir.name}")
    print("=" * 65)

    # 1. Read CSV records
    records: list[tuple[Path, str, str]] = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                img_path = REPO_ROOT / row[0].strip() if not Path(row[0].strip()).is_absolute() else Path(row[0].strip())
                records.append((img_path, row[1].strip(), row[2].strip() if len(row) > 2 else ""))

    total_records = len(records)
    print(f"✓ Total samples cataloged in CSV: {total_records:,}")

    # 2. Check token coverage against vocabulary
    if vocab_path.is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            char_to_idx = json.load(f)
        required_chars = set(c for c in char_to_idx.keys() if c != "<blank>")

        all_text = "".join(r[1] for r in records)
        char_counts = Counter(all_text)
        missing_chars = [c for c in required_chars if char_counts[c] == 0]

        coverage_pct = ((len(required_chars) - len(missing_chars)) / len(required_chars)) * 100.0
        print(f"✓ Vocabulary Coverage: {coverage_pct:.1f}% ({len(required_chars) - len(missing_chars)}/{len(required_chars)} tokens)")
        if missing_chars:
            print(f"  Missing tokens: {missing_chars}")
        else:
            print("  🌟 100% PERFECT VOCABULARY COVERAGE ACROSS ALL 86 CHARACTERS!")

    # 3. Create visual inspection sheet
    print(f"\nSampling {num_samples} random line crops for visual inspection...")
    sampled = random.sample(records, min(num_samples, total_records))

    card_w = 1200
    row_h = 100
    canvas_h = len(sampled) * row_h + 80

    canvas = Image.new("RGB", (card_w, canvas_h), color=(250, 252, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = get_khmer_font(size=24)
    font_text = get_khmer_font(size=20)
    font_mono = ImageFont.load_default()

    draw.text((20, 20), f"Synthetic Dataset Preview ({dataset_dir.name}) — {num_samples} Random Samples", fill=(15, 23, 42), font=font_title)
    draw.line([(20, 60), (card_w - 20, 60)], fill=(203, 213, 225), width=2)

    cur_y = 80
    for idx, (img_path, text, writer_id) in enumerate(sampled, start=1):
        if not img_path.is_file():
            continue

        # Open and paste line crop
        crop_img = Image.open(img_path).convert("RGB")
        cw, ch = crop_img.size

        # Paste image on left
        target_cw = min(cw, 580)
        resized_crop = crop_img.resize((target_cw, 48), Image.Resampling.BILINEAR)
        canvas.paste(resized_crop, (25, cur_y + 10))

        # Inscribe Ground Truth & Metadata on right
        draw.text((625, cur_y + 8), f"Line {idx} [{writer_id}]:", fill=(100, 116, 139), font=font_text)
        draw.text((625, cur_y + 35), text, fill=(15, 23, 42), font=font_text)

        draw.line([(25, cur_y + 85), (card_w - 25, cur_y + 85)], fill=(241, 245, 249), width=1)
        cur_y += row_h

    output_viz.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_viz)
    print(f"🖼️ Saved visual inspection sheet to: {output_viz}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Inspector.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "synthetic", help="Path to synthetic dataset")
    parser.add_argument("--vocab-path", type=Path, default=REPO_ROOT / "data" / "metadata" / "char_to_idx.json", help="Path to char_to_idx.json")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of random lines to preview")
    parser.add_argument("--output-viz", type=Path, default=REPO_ROOT / "results" / "synthetic_inspection_preview.png", help="Output PNG preview")
    args = parser.parse_args()

    inspect_dataset(
        dataset_dir=args.dataset_dir,
        vocab_path=args.vocab_path,
        num_samples=args.num_samples,
        output_viz=args.output_viz,
    )


if __name__ == "__main__":
    main()

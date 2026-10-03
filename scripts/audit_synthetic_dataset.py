#!/usr/bin/env python3
"""Audit script for verifying Synthetic 2.0 dataset quality, length distribution,
and 100% vocabulary coverage against the real project metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import pandas as pd

SYNTH_CSV = Path("data/synthetic/synthetic_labels.csv")
REAL_CSV = Path("data/metadata/labels.csv")
VOCAB_JSON = Path("data/metadata/char_to_idx.json")


def audit():
    print("=" * 65)
    print("             SYNTHETIC 2.0 DATASET AUDIT REPORT")
    print("=" * 65)

    if not SYNTH_CSV.exists():
        print(f"Error: {SYNTH_CSV} does not exist!")
        return

    df_synth = pd.read_csv(SYNTH_CSV)
    df_real = pd.read_csv(REAL_CSV)

    with open(VOCAB_JSON, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_chars = set(c for c in vocab.keys() if c != "<blank>")

    # 1. Total Counts
    total_samples = len(df_synth)
    print(f"Total Synthetic Samples : {total_samples:,}")

    # 2. Writer ID Disjoint Verification
    real_writers = set(df_real["writer_id"].unique())
    synth_writers = set(df_synth["writer_id"].unique())
    overlap = real_writers.intersection(synth_writers)
    print(f"Synthetic Writer Clusters: {len(synth_writers)}")
    print(f"Real Writer IDs          : {sorted(list(real_writers))}")
    print(f"Writer ID Overlap        : {len(overlap)} (Zero leakage: {'PASS ✓' if len(overlap) == 0 else 'FAIL ✗'})")

    # 3. Text Length Distribution
    lengths = df_synth["label"].str.len()
    real_lengths = df_real["label"].str.len()
    print("-" * 65)
    print("Text Length Distribution (Characters):")
    print(f"  Synthetic : Min={lengths.min()}, Mean={lengths.mean():.1f}, Median={lengths.median():.0f}, Max={lengths.max()}")
    print(f"  Real Lines: Min={real_lengths.min()}, Mean={real_lengths.mean():.1f}, Median={real_lengths.median():.0f}, Max={real_lengths.max()}")

    # 4. Vocabulary Coverage
    synth_chars = set("".join(df_synth["label"].tolist()))
    missing_chars = vocab_chars - synth_chars
    coverage_pct = ((len(vocab_chars) - len(missing_chars)) / len(vocab_chars)) * 100.0
    print("-" * 65)
    print(f"Vocabulary Coverage      : {coverage_pct:.1f}% ({len(vocab_chars) - len(missing_chars)}/{len(vocab_chars)} chars)")
    if missing_chars:
        print(f"  Missing Characters     : {sorted(list(missing_chars))}")
    else:
        print("  All 86 Khmer Unicode tokens fully covered: PASS ✓")

    # 5. Image Integrity Check (First 100 samples)
    valid_dims = True
    widths = []
    for idx, row in df_synth.head(200).iterrows():
        img_path = Path(row["image"])
        if not img_path.exists():
            print(f"  Missing file: {img_path}")
            valid_dims = False
            break
        img = cv2.imread(str(img_path))
        if img is None:
            valid_dims = False
            break
        h, w, _ = img.shape
        widths.append(w)
        if h != 48:
            print(f"  Invalid height: {h} for {img_path}")
            valid_dims = False
            break

    print("-" * 65)
    print(f"Image Dimensional Check  : {'PASS ✓ (Height=48px fixed)' if valid_dims else 'FAIL ✗'}")
    print(f"Sample Width Distribution: Min={min(widths)}px, Mean={sum(widths)/len(widths):.0f}px, Max={max(widths)}px")

    # 6. Sample Examples
    print("-" * 65)
    print("Random Synthetic 2.0 Samples:")
    samples = df_synth.sample(5, random_state=42)
    for idx, (_, r) in enumerate(samples.iterrows(), 1):
        print(f"  [{idx}] ({len(r['label'])} chars): {r['label']}")

    print("=" * 65)


if __name__ == "__main__":
    audit()

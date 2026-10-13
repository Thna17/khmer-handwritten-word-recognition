#!/usr/bin/env python3
"""Downloads and prepares the SoyVitou Khmer Handwritten Dataset (4.2k).

Filters out non-Khmer samples (English notes), normalizes Unicode,
extracts images to PNG, and creates a standardized labels.csv.

Usage:
    python scripts/prepare_soyvitou_dataset.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "soyvitou"
IMAGES_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "soyvitou_labels.csv"
VOCAB_PATH = REPO_ROOT / "data" / "metadata" / "char_to_idx.json"

PARQUET_URLS = [
    ("train", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/trainset.parquet"),
    ("val", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/valset.parquet"),
    ("test", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/testset.parquet"),
]


def load_vocab() -> set[str]:
    """Loads approved character vocabulary."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return set(c for c in vocab.keys() if c != "<blank>")
    return set()


def download_file(url: str, dest_path: Path):
    """Downloads a file with basic progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  Using cached {dest_path.name}")
        return

    print(f"  Downloading {dest_path.name} from Hugging Face...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  ✓ Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")


def is_clean_khmer(text: str, approved_vocab: set[str]) -> bool:
    """Checks if text contains Khmer and has no English/Latin letters."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) == 0:
        return False

    # Must contain at least one Khmer character
    if not re.search(r"[\u1780-\u17FF\u19E0-\u19FF]", text):
        return False

    # Must NOT contain English/Latin alphabet
    if re.search(r"[a-zA-Z]", text):
        return False

    # Check against project vocabulary if available
    if approved_vocab:
        # Check if characters are mostly in vocab
        unknown = [c for c in text if c not in approved_vocab and c != " "]
        if unknown:
            return False

    return True


def main():
    print("=" * 65)
    print("     PREPARING SOYVITOU KHMER HANDWRITTEN DATASET (4.2K)")
    print("=" * 65)

    approved_vocab = load_vocab()
    print(f"Loaded project vocabulary ({len(approved_vocab)} allowed tokens).")

    cache_dir = REPO_ROOT / "data" / "cache_soyvitou"
    cache_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    global_idx = 0
    total_raw_rows = 0
    skipped_english = 0
    skipped_vocab = 0

    for split_name, url in PARQUET_URLS:
        parquet_file = cache_dir / f"{split_name}.parquet"
        download_file(url, parquet_file)

        df = pd.read_parquet(parquet_file)
        total_raw_rows += len(df)
        print(f"\nProcessing {split_name} split ({len(df)} rows)...")

        for _, row in df.iterrows():
            raw_text = row["text"]
            if not isinstance(raw_text, str):
                continue

            clean_text = unicodedata.normalize("NFC", raw_text.strip())

            # Check if it has English letters
            if re.search(r"[a-zA-Z]", clean_text):
                skipped_english += 1
                continue

            # Check if it contains Khmer
            if not re.search(r"[\u1780-\u17FF\u19E0-\u19FF]", clean_text):
                skipped_english += 1
                continue

            # Check against project vocabulary
            if approved_vocab:
                unknown_chars = [c for c in clean_text if c not in approved_vocab and c != " "]
                if unknown_chars:
                    skipped_vocab += 1
                    continue

            # Extract image bytes
            img_data = row["image"]
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            try:
                img = Image.open(BytesIO(img_bytes)).convert("L")
            except Exception:
                continue

            if img.width < 10 or img.height < 10:
                continue

            # Save clean image
            img_filename = f"sv_{global_idx:05d}.png"
            img_path = IMAGES_DIR / img_filename
            img.save(img_path)

            rel_img_path = str(Path("images") / img_filename)
            records.append({
                "image": rel_img_path,
                "label": clean_text,
                "writer_id": f"SV_{split_name.upper()}_{(global_idx % 40):02d}",
            })
            global_idx += 1

            if global_idx % 500 == 0:
                print(f"  Extracted {global_idx} clean Khmer handwritten samples...")

    # Save metadata CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(CSV_PATH, index=False)

    print("\n" + "=" * 65)
    print("                     DATASET SUMMARY")
    print("=" * 65)
    print(f"Raw rows processed       : {total_raw_rows}")
    print(f"Skipped English rows     : {skipped_english}")
    print(f"Skipped out-of-vocab rows: {skipped_vocab}")
    print(f"✓ Total Clean Khmer Crops: {len(df_out):,}")
    print(f"✓ Images directory       : {IMAGES_DIR}")
    print(f"✓ Metadata CSV saved to  : {CSV_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()

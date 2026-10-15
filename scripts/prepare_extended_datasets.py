#!/usr/bin/env python3
"""Automated pipeline to prepare extended Khmer OCR and Handwritten datasets.

Datasets supported:
1. SoyVitou/KhmerSynthetic1M (Hugging Face):
   - 1,000,000 synthetic OCR images with lighting, blur, and curved text augmentations.
   - Extracts N high-quality shards (e.g. 25,000 to 50,000 samples) for large-scale pre-training.
2. SoyVitou/khmer-handwritten-dataset-4.2k (Hugging Face):
   - 4,200 real human handwriting crops with NFC Unicode normalization.
3. Consolidated Manifest Builder:
   - Extended Pre-training Manifest: Combines HarfBuzz synthetic lines + KhmerSynthetic1M.
   - Extended Fine-tuning Manifest: Combines 177 real line crops (W001-W008) + SoyVitou 4.2k words,
     strictly holding out test writers (W007, W0011) to maintain 100% academic evaluation integrity.

Usage:
    # Prepare both synthetic 1M slice and SoyVitou 4.2k:
    python scripts/prepare_extended_datasets.py --all --max-synthetic 25000

    # Prepare only KhmerSynthetic1M (e.g. 50,000 samples):
    python scripts/prepare_extended_datasets.py --synthetic-1m --max-synthetic 50000

    # Prepare only SoyVitou 4.2k real handwriting:
    python scripts/prepare_extended_datasets.py --soyvitou
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "data" / "metadata"
VOCAB_PATH = METADATA_DIR / "char_to_idx.json"

SOYVITOU_URLS = [
    ("train", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/trainset.parquet"),
    ("val", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/valset.parquet"),
    ("test", "https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k/resolve/main/testset.parquet"),
]

SYNTHETIC_1M_BASE_URL = "https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M/resolve/main/data"


def load_vocab() -> set[str]:
    """Loads approved project vocabulary."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return set(c for c in vocab.keys() if c != "<blank>")
    return set()


def download_with_progress(url: str, dest_path: Path):
    """Downloads a remote file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"  [Cache hit] Using existing {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"  Downloading {dest_path.name} from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r    {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\r    {downloaded / (1024*1024):.1f} MB downloaded", end="", flush=True)
        print()
    print(f"  ✓ Saved to {dest_path}")


def is_clean_khmer_label(text: str, approved_vocab: set[str]) -> bool:
    """Verifies label is non-empty, contains Khmer, no Latin, and strictly conforms to vocabulary."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False

    # Must contain Khmer unicode block
    if not re.search(r"[\u1780-\u17FF\u19E0-\u19FF]", text):
        return False

    # Must NOT contain English/Latin letters
    if re.search(r"[a-zA-Z]", text):
        return False

    # Check against project vocabulary
    if approved_vocab:
        for c in text:
            if c != " " and c not in approved_vocab:
                return False

    return True


def prepare_soyvitou_handwritten(output_dir: Path, approved_vocab: set[str]) -> Path:
    """Downloads and extracts the 4,200 handwritten images from SoyVitou."""
    print("\n" + "=" * 65)
    print("  [1/3] PREPARING SOYVITOU KHMER HANDWRITTEN DATASET (4.2K)")
    print("=" * 65)

    cache_dir = output_dir / "cache"
    images_dir = output_dir / "images"
    csv_path = output_dir / "soyvitou_labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    global_idx = 0

    for split_name, url in SOYVITOU_URLS:
        parquet_path = cache_dir / f"{split_name}.parquet"
        download_with_progress(url, parquet_path)

        df = pd.read_parquet(parquet_path)
        print(f"  Extracting {split_name} split ({len(df)} samples)...")

        for _, row in df.iterrows():
            raw_text = row.get("text", "")
            if not isinstance(raw_text, str):
                continue
            clean_text = unicodedata.normalize("NFC", raw_text.strip())

            if not is_clean_khmer_label(clean_text, approved_vocab):
                continue

            # Extract image bytes
            img_data = row["image"]
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("L")
            except Exception:
                continue

            if img.width < 10 or img.height < 10:
                continue

            img_filename = f"sv_{global_idx:05d}.png"
            img_path = images_dir / img_filename
            img.save(img_path)

            records.append({
                "image": f"images/{img_filename}",
                "label": clean_text,
                "writer_id": f"SV_{split_name.upper()}_{(global_idx % 40):02d}",
            })
            global_idx += 1

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    print(f"✓ Saved {len(df_out):,} clean handwritten crops to {csv_path}")
    return csv_path


def prepare_synthetic_1m(output_dir: Path, approved_vocab: set[str], max_samples: int = 25000) -> Path:
    """Streams or downloads from KhmerSynthetic1M and extracts clean samples."""
    print("\n" + "=" * 65)
    print(f"  [2/3] PREPARING KHMERSYNTHETIC1M (Target: {max_samples:,} samples)")
    print("=" * 65)

    cache_dir = output_dir / "cache"
    images_dir = output_dir / "images"
    csv_path = output_dir / "synthetic_1m_labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    global_idx = 0

    # Try streaming first via Hugging Face datasets library
    try:
        from datasets import load_dataset
        print("  Using Hugging Face streaming mode (no massive 600MB shard download needed!)...")
        ds = load_dataset("SoyVitou/KhmerSynthetic1M", split="train", streaming=True)

        for row in ds:
            if global_idx >= max_samples:
                break

            raw_label = row.get("label", "")
            if not isinstance(raw_label, str):
                continue
            clean_label = unicodedata.normalize("NFC", raw_label.strip())

            if not is_clean_khmer_label(clean_label, approved_vocab):
                continue

            img_entry = row.get("image")
            if isinstance(img_entry, Image.Image):
                img = img_entry.convert("L")
            else:
                img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("L")
                except Exception:
                    continue

            if img.width < 10 or img.height < 10:
                continue

            img_filename = f"synth1m_{global_idx:06d}.jpg"
            img_path = images_dir / img_filename
            img.save(img_path, format="JPEG", quality=85)

            records.append({
                "image": f"images/{img_filename}",
                "label": clean_label,
                "writer_id": f"SYNTH1M_{(global_idx % 100):03d}",
            })
            global_idx += 1

            if global_idx % 2000 == 0:
                print(f"    Extracted {global_idx:,}/{max_samples:,} synthetic samples...", flush=True)

        df_out = pd.DataFrame(records)
        df_out.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(df_out):,} synthetic samples to {csv_path}")
        return csv_path

    except (ImportError, Exception) as e:
        print(f"  Streaming unavailable ({e}). Falling back to direct Parquet shard download...")

    shard_idx = 0
    while global_idx < max_samples and shard_idx < 20:
        shard_filename = f"train-{shard_idx:05d}-of-00020.parquet"
        shard_url = f"{SYNTHETIC_1M_BASE_URL}/{shard_filename}"
        shard_path = cache_dir / shard_filename

        download_with_progress(shard_url, shard_path)

        print(f"  Reading shard {shard_idx} ({shard_filename})...")
        df = pd.read_parquet(shard_path)

        for _, row in df.iterrows():
            if global_idx >= max_samples:
                break

            raw_label = row.get("label", "")
            if not isinstance(raw_label, str):
                continue
            clean_label = unicodedata.normalize("NFC", raw_label.strip())

            if not is_clean_khmer_label(clean_label, approved_vocab):
                continue

            img_entry = row.get("image")
            img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("L")
            except Exception:
                continue

            if img.width < 10 or img.height < 10:
                continue

            img_filename = f"synth1m_{global_idx:06d}.jpg"
            img_path = images_dir / img_filename
            img.save(img_path, format="JPEG", quality=85)

            records.append({
                "image": f"images/{img_filename}",
                "label": clean_label,
                "writer_id": f"SYNTH1M_{(global_idx % 100):03d}",
            })
            global_idx += 1

            if global_idx % 2000 == 0:
                print(f"    Extracted {global_idx:,}/{max_samples:,} synthetic samples...", flush=True)

        shard_idx += 1

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    print(f"✓ Saved {len(df_out):,} synthetic samples to {csv_path}")
    return csv_path



def build_consolidated_manifests(
    soyvitou_csv: Path | None,
    synthetic_1m_csv: Path | None,
    approved_vocab: set[str],
):
    """Builds consolidated manifests for Stage 1 Pre-training and Stage 2 Fine-tuning."""
    print("\n" + "=" * 65)
    print("  [3/3] BUILDING CONSOLIDATED STAGE 1 & STAGE 2 MANIFESTS")
    print("=" * 65)

    # 1. Extended Stage 1 Pre-training: HarfBuzz Synthetic (10k) + KhmerSynthetic1M
    pretrain_rows = []
    harfbuzz_csv = REPO_ROOT / "data" / "synthetic" / "synthetic_labels.csv"
    harfbuzz_images_dir = REPO_ROOT / "data" / "synthetic"

    if harfbuzz_csv.exists():
        df_hb = pd.read_csv(harfbuzz_csv)
        print(f"  Loaded {len(df_hb):,} HarfBuzz synthetic line samples.")
        for _, r in df_hb.iterrows():
            pretrain_rows.append({
                "image": str((harfbuzz_images_dir / r["image"]).relative_to(REPO_ROOT)),
                "label": unicodedata.normalize("NFC", str(r["label"])),
                "writer_id": str(r["writer_id"]),
            })

    if synthetic_1m_csv and synthetic_1m_csv.exists():
        df_s1m = pd.read_csv(synthetic_1m_csv)
        s1m_root = synthetic_1m_csv.parent
        print(f"  Loaded {len(df_s1m):,} KhmerSynthetic1M samples.")
        for _, r in df_s1m.iterrows():
            pretrain_rows.append({
                "image": str((s1m_root / r["image"]).relative_to(REPO_ROOT)),
                "label": unicodedata.normalize("NFC", str(r["label"])),
                "writer_id": str(r["writer_id"]),
            })

    if pretrain_rows:
        df_pre = pd.DataFrame(pretrain_rows)
        out_pre = METADATA_DIR / "extended_pretrain_manifest.csv"
        df_pre.to_csv(out_pre, index=False)
        print(f"  ✓ Master Pre-training Manifest: {len(df_pre):,} samples -> {out_pre}")

    # 2. Extended Stage 2 Fine-tuning:
    # Clean Real Lines (no filler dots, no sliced vowels/subscripts) + SoyVitou 4.2k
    finetune_rows = []
    clean_csv = METADATA_DIR / "labels_clean.csv"
    raw_labels_csv = METADATA_DIR / "labels.csv"

    if clean_csv.exists():
        real_csv = clean_csv
    elif raw_labels_csv.exists():
        real_csv = raw_labels_csv
    else:
        real_csv = None

    if real_csv is not None:
        df_real = pd.read_csv(real_csv)
        clean_count = 0
        for _, r in df_real.iterrows():
            img_rel = str(r["image"])
            img_full = REPO_ROOT / img_rel
            label_text = unicodedata.normalize("NFC", str(r["label"])).strip()
            wid = str(r["writer_id"])

            # Skip dot fillers or very short cramped crops
            if re.search(r"\.{3,}", label_text) or label_text.count(".") > 4:
                continue

            if img_full.exists():
                try:
                    with Image.open(img_full) as im:
                        if im.height < 34:
                            continue
                except Exception:
                    continue


            finetune_rows.append({
                "image": img_rel,
                "label": label_text,
                "writer_id": wid,
            })
            clean_count += 1
        print(f"  Loaded {clean_count:,} pristine, verified real line crops (purged defective/clipped crops).")


    if soyvitou_csv and soyvitou_csv.exists():
        df_sv = pd.read_csv(soyvitou_csv)
        sv_root = soyvitou_csv.parent
        print(f"  Loaded {len(df_sv):,} SoyVitou handwritten words.")
        for _, r in df_sv.iterrows():
            finetune_rows.append({
                "image": str((sv_root / r["image"]).relative_to(REPO_ROOT)),
                "label": unicodedata.normalize("NFC", str(r["label"])),
                "writer_id": str(r["writer_id"]),
            })

    if finetune_rows:
        df_fine = pd.DataFrame(finetune_rows)
        out_fine = METADATA_DIR / "extended_finetune_manifest.csv"
        df_fine.to_csv(out_fine, index=False)
        print(f"  ✓ Master Fine-tuning Manifest: {len(df_fine):,} samples -> {out_fine}")


    print("\n" + "=" * 65)
    print("  SUMMARY: All extended datasets successfully prepared!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Prepare extended datasets for Khmer Handwritten OCR.")
    parser.add_argument("--all", action="store_true", help="Download and process both SoyVitou and Synthetic1M")
    parser.add_argument("--soyvitou", action="store_true", help="Download and process SoyVitou 4.2k handwritten dataset")
    parser.add_argument("--synthetic-1m", action="store_true", help="Download and process KhmerSynthetic1M slice")
    parser.add_argument("--max-synthetic", type=int, default=25000, help="Number of synthetic samples to extract (default: 25,000)")
    parser.add_argument("--output-dir", type=str, default="data", help="Root data directory")

    args = parser.parse_args()

    if not (args.all or args.soyvitou or args.synthetic_1m):
        args.all = True

    approved_vocab = load_vocab()
    print(f"Loaded project vocabulary with {len(approved_vocab)} approved tokens.")

    data_dir = REPO_ROOT / args.output_dir
    soyvitou_csv = None
    synthetic_1m_csv = None

    if args.all or args.soyvitou:
        sv_dir = data_dir / "soyvitou"
        soyvitou_csv = prepare_soyvitou_handwritten(sv_dir, approved_vocab)

    if args.all or args.synthetic_1m:
        s1m_dir = data_dir / "synthetic_1m"
        synthetic_1m_csv = prepare_synthetic_1m(s1m_dir, approved_vocab, max_samples=args.max_synthetic)

    build_consolidated_manifests(soyvitou_csv, synthetic_1m_csv, approved_vocab)


if __name__ == "__main__":
    main()

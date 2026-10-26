#!/usr/bin/env python3
"""Interactive & Automated Khmer Line Transcription Helper.

Allows researchers and students to rapidly transcribe newly segmented handwriting
lines, perform AI-assisted hypothesis suggestion, validate Khmer Unicode NFC
orthography, and register the new writer safely into training splits.

Features:
1. Scans `data/handwritten_external/{writer_id}/lines_raw/` for uncataloged lines.
2. AI-Assisted Transcription: Optionally suggests an initial hypothesis using
   a trained KHTR checkpoint to minimize typing overhead.
3. Strict Unicode Validation:
   - Canonical Unicode NFC normalization.
   - Rejects or flags out-of-vocabulary characters not in `char_to_idx.json`.
4. Updates `data/metadata/labels.csv` atomically with backups.
5. Auto-Registers Writer in `data/metadata/writer_split.json`:
   - Adds new writer to `train_writers`.
   - Guarantees `test_writers` (W007, W0011) remain permanently untouched (zero-leakage).

Usage:
  # Interactive mode with AI hypothesis suggestion:
  python scripts/transcribe_new_lines.py --writer-id W012 --suggest-ai

  # Batch transcription from text file (one line per crop):
  python scripts/transcribe_new_lines.py --writer-id W012 --from-file transcripts.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from src.decoder import collapse_ctc_sequence
from src.model_baseline import BaselineCRNN
from src.postprocessor import clean_khmer_orthography
from src.tokenizer import KhmerTokenizer
from src.utils import get_device, load_checkpoint


def load_vocabulary(vocab_path: Path) -> tuple[dict[str, int], KhmerTokenizer]:
    """Loads vocabulary map and initializes tokenizer."""
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    tokenizer = KhmerTokenizer(char_to_idx)
    return char_to_idx, tokenizer


def load_ai_model(
    checkpoint_path: Path | None,
    vocab_size: int,
    device: torch.device,
) -> BaselineCRNN | None:
    """Loads trained BaselineCRNN model for AI-assisted transcription."""
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        model = BaselineCRNN(num_classes=vocab_size)
        load_checkpoint(checkpoint_path, model, map_location=device)
        model.to(device)
        model.eval()
        print(f"🤖 Loaded AI Assistant from {checkpoint_path.name}")
        return model
    except Exception as e:
        print(f"⚠️ Could not load AI model ({e}); proceeding with manual transcription.")
        return None


def run_ai_prediction(
    model: BaselineCRNN,
    image_path: Path,
    tokenizer: KhmerTokenizer,
    device: torch.device,
    target_height: int = 48,
    max_width: int = 1600,
) -> str:
    """Runs quick forward pass to generate draft hypothesis."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    scale = target_height / float(h)
    new_w = min(int(round(w * scale)), max_width)

    resized = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (max_width, target_height), 255)
    canvas.paste(resized, (0, 0))

    arr = (np.array(canvas, dtype=np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        seq_len = max(1, new_w // 4)
        pred_tokens = collapse_ctc_sequence(logits[0, :seq_len, :])
        raw_text = tokenizer.decode(pred_tokens)
        return clean_khmer_orthography(raw_text)


def validate_khmer_transcription(text: str, valid_chars: set[str]) -> tuple[bool, str, list[str]]:
    """Validates Khmer text against canonical Unicode NFC and known vocabulary."""
    # Canonical Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text).strip()
    if not normalized:
        return False, "", ["Empty transcription"]

    invalid_chars: list[str] = []
    for ch in normalized:
        if ch not in valid_chars:
            invalid_chars.append(f"'{ch}' (U+{ord(ch):04X})")

    is_valid = len(invalid_chars) == 0
    return is_valid, normalized, invalid_chars


def update_writer_split(
    writer_id: str,
    split_path: Path = REPO_ROOT / "data" / "metadata" / "writer_split.json",
) -> None:
    """Safely adds a new writer to train_writers, preserving test set isolation."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Check for test-set leakage
    if writer_id in splits.get("test_writers", []):
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: {writer_id} is marked as a HELD-OUT TEST WRITER. "
            "Never append test writers into training data!"
        )

    all_known = (
        set(splits.get("train_writers", []))
        | set(splits.get("val_writers", []))
        | set(splits.get("test_writers", []))
    )

    if writer_id not in all_known:
        splits["train_writers"].append(writer_id)
        splits["train_writers"] = sorted(list(set(splits["train_writers"])))

        # Atomic write
        temp_path = split_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)
        temp_path.replace(split_path)
        print(f"🔒 Added writer {writer_id} to 'train_writers' in {split_path.name}")
    else:
        print(f"ℹ️ Writer {writer_id} already registered in splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive & Automated Khmer Line Transcription.")
    parser.add_argument("--writer-id", required=True, help="Writer identifier (e.g. W012)")
    parser.add_argument("--suggest-ai", action="store_true", help="Suggest initial transcription with AI model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "a1_baseline_finetuned_best.pt",
        help="Path to trained model weights for AI suggestion",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=REPO_ROOT / "data" / "metadata" / "char_to_idx.json",
        help="Path to vocabulary JSON",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=REPO_ROOT / "data" / "metadata" / "labels.csv",
        help="Path to labels.csv",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        help="Optional text file containing one transcription line per crop (batch mode)",
    )
    args = parser.parse_args()

    # Load vocab & tokenizer
    char_to_idx, tokenizer = load_vocabulary(args.vocab_path)
    valid_chars = set(char_to_idx.keys())
    device = get_device()

    # Model for AI suggestions
    ai_model = None
    if args.suggest_ai:
        ai_model = load_ai_model(args.checkpoint, tokenizer.vocab_size, device)

    # Locate writer line crops
    writer_lines_dir = REPO_ROOT / "data" / "handwritten_external" / args.writer_id / "lines_raw"
    if not writer_lines_dir.is_dir():
        print(f"❌ No line crops found at: {writer_lines_dir}")
        print(f"   Please run scripts/add_new_writer.py --writer-id {args.writer_id} first.")
        sys.exit(1)

    crop_files = sorted(list(writer_lines_dir.glob("*.png")))
    if not crop_files:
        print(f"❌ No PNG files found in {writer_lines_dir}")
        sys.exit(1)

    # Read existing entries to skip already labeled lines
    existing_images = set()
    if args.labels_csv.is_file():
        with open(args.labels_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    existing_images.add(row[0].strip())

    pending_crops = [p for p in crop_files if str(p.relative_to(REPO_ROOT)) not in existing_images]
    print(f"📄 Writer {args.writer_id}: Found {len(crop_files)} total crops ({len(pending_crops)} pending transcription).")

    if not pending_crops:
        print("✓ All lines for this writer are already cataloged in labels.csv!")
        update_writer_split(args.writer_id)
        return

    # Batch mode from text file
    transcriptions: list[tuple[str, str, str]] = []
    if args.from_file and args.from_file.is_file():
        with open(args.from_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) != len(pending_crops):
            print(f"⚠️ Warning: File has {len(lines)} lines but there are {len(pending_crops)} pending image crops.")

        for crop_path, text in zip(pending_crops, lines):
            is_valid, norm_text, invalid = validate_khmer_transcription(text, valid_chars)
            if not is_valid:
                print(f"❌ Error in line '{text}': Invalid tokens {invalid}")
                continue
            rel_path = str(crop_path.relative_to(REPO_ROOT))
            transcriptions.append((rel_path, norm_text, args.writer_id))
    else:
        # Interactive Mode
        print("\n=======================================================")
        print("   Interactive Khmer Transcription Console")
        print("   Commands: [Enter] Accept suggestion | [q] Quit & save")
        print("=======================================================\n")

        for idx, crop_path in enumerate(pending_crops, start=1):
            rel_path = str(crop_path.relative_to(REPO_ROOT))
            print(f"\n[{idx}/{len(pending_crops)}] Image: {crop_path.name}")

            suggestion = ""
            if ai_model is not None:
                suggestion = run_ai_prediction(ai_model, crop_path, tokenizer, device)
                print(f"🤖 AI Draft : {suggestion}")

            while True:
                prompt_str = f"✍️ Enter text [{'Accept: Enter' if suggestion else ''}]: "
                user_input = input(prompt_str).strip()

                if user_input.lower() == "q":
                    print("Exiting transcription session...")
                    break

                target_text = user_input if user_input else suggestion
                if not target_text:
                    print("⚠️ Text cannot be empty. Please enter transcription or 'q' to abort.")
                    continue

                is_valid, norm_text, invalid = validate_khmer_transcription(target_text, valid_chars)
                if not is_valid:
                    print(f"❌ Invalid Khmer tokens detected: {', '.join(invalid)}")
                    print("   Please re-type or correct orthography.")
                    continue

                transcriptions.append((rel_path, norm_text, args.writer_id))
                print(f"✓ Saved: {norm_text}")
                break

            if user_input.lower() == "q":
                break

    # Save to labels.csv
    if transcriptions:
        # Create backup first
        backup_csv = args.labels_csv.with_suffix(f".bak_{args.writer_id}")
        shutil.copyfile(args.labels_csv, backup_csv)
        print(f"📦 Created backup of labels.csv at: {backup_csv.name}")

        with open(args.labels_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row in transcriptions:
                writer.writerow(row)

        print(f"🎉 Appended {len(transcriptions)} verified lines to {args.labels_csv}")

        # Register writer in training split
        update_writer_split(args.writer_id)


if __name__ == "__main__":
    main()

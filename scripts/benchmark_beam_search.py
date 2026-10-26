#!/usr/bin/env python3
"""Benchmark Greedy Decoding vs CTC Prefix Beam Search + Language Model.

Evaluates performance on permanently held-out test writers (W007 & W0011):
- Greedy CTC decoding (argmax)
- CTC Beam Search (acoustic-only, alpha=0)
- CTC Beam Search + 4-Gram Khmer Language Model (alpha=0.6, beta=1.0)

Measures:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact Line Match Rate
- Average Latency per Line
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from src.decoder import collapse_ctc_sequence
from src.decoder_beam_search import build_default_lm, ctc_prefix_beam_search_single
from src.metrics import character_error_rate, word_error_rate
from src.model_baseline import BaselineCRNN
from src.postprocessor import clean_khmer_orthography
from src.tokenizer import KhmerTokenizer
from src.utils import get_device, load_checkpoint


def preprocess_image(image_path: Path, target_height: int = 48, max_width: int = 1600) -> tuple[torch.Tensor, int]:
    img = Image.open(image_path).convert("L")
    w, h = img.size
    scale = target_height / float(h)
    new_w = min(int(round(w * scale)), max_width)

    resized = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (max_width, target_height), 255)
    canvas.paste(resized, (0, 0))

    arr = (np.array(canvas, dtype=np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, new_w


def run_benchmark(
    checkpoint_path: Path,
    vocab_path: Path,
    labels_csv: Path,
    split_path: Path,
    beam_width: int = 20,
    alpha: float = 0.6,
    beta: float = 1.0,
    max_samples: int | None = None,
) -> None:
    device = get_device()
    print(f"⚡ Benchmark Device: {device}")

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    tokenizer = KhmerTokenizer(char_to_idx)

    # Load model
    model = BaselineCRNN(num_classes=tokenizer.vocab_size)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded: {checkpoint_path.name}")

    # Load 4-Gram LM
    lm = build_default_lm()
    print(f"✓ Language Model loaded: 4-Gram with {lm.total_chars:,} training characters")

    # Load held-out test split
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_writers = set(splits.get("test_writers", ["W007", "W0011"]))
    print(f"🔒 Evaluating strictly on held-out test writers: {sorted(list(test_writers))}")

    # Collect test samples
    test_samples: list[tuple[Path, str]] = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[2].strip() in test_writers:
                img_path = REPO_ROOT / row[0].strip()
                if img_path.is_file():
                    test_samples.append((img_path, row[1].strip()))

    if max_samples:
        test_samples = test_samples[:max_samples]

    print(f"📊 Collected {len(test_samples)} test line samples for decoding comparison.")

    # Tracking metrics
    greedy_refs, greedy_preds = [], []
    beam_refs, beam_preds = [], []
    beam_lm_refs, beam_lm_preds = [], []

    greedy_times, beam_times, beam_lm_times = [], [], []

    print("\nRunning side-by-side evaluation across all test lines...")
    for idx, (img_path, ground_truth) in enumerate(test_samples, start=1):
        tensor, valid_w = preprocess_image(img_path)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            seq_len = max(1, valid_w // 4)
            logits_valid = logits[0, :seq_len, :]

        # 1. Greedy decoding
        t0 = time.perf_counter()
        greedy_token_ids = collapse_ctc_sequence(logits_valid.argmax(dim=-1).tolist())
        greedy_text = clean_khmer_orthography(tokenizer.decode(greedy_token_ids))
        greedy_times.append(time.perf_counter() - t0)

        # 2. Beam Search without LM (alpha = 0)
        t0 = time.perf_counter()
        beam_raw, _ = ctc_prefix_beam_search_single(
            logits_valid, tokenizer=tokenizer, lm=None, beam_width=beam_width, alpha=0.0, beta=beta
        )
        beam_text = clean_khmer_orthography(beam_raw)
        beam_times.append(time.perf_counter() - t0)

        # 3. Beam Search WITH 4-Gram LM (alpha = alpha)
        t0 = time.perf_counter()
        beam_lm_raw, _ = ctc_prefix_beam_search_single(
            logits_valid, tokenizer=tokenizer, lm=lm, beam_width=beam_width, alpha=alpha, beta=beta
        )
        beam_lm_text = clean_khmer_orthography(beam_lm_raw)
        beam_lm_times.append(time.perf_counter() - t0)

        greedy_refs.append(ground_truth)
        greedy_preds.append(greedy_text)

        beam_refs.append(ground_truth)
        beam_preds.append(beam_text)

        beam_lm_refs.append(ground_truth)
        beam_lm_preds.append(beam_lm_text)

        if idx % 10 == 0 or idx == len(test_samples):
            print(f"  Processed [{idx}/{len(test_samples)}] lines...")

    # Calculate aggregate metrics
    cer_greedy = character_error_rate(greedy_refs, greedy_preds)
    wer_greedy = word_error_rate(greedy_refs, greedy_preds)
    exact_greedy = sum(1 for r, p in zip(greedy_refs, greedy_preds) if r == p) / len(greedy_refs)

    cer_beam = character_error_rate(beam_refs, beam_preds)
    wer_beam = word_error_rate(beam_refs, beam_preds)
    exact_beam = sum(1 for r, p in zip(beam_refs, beam_preds) if r == p) / len(beam_refs)

    cer_beam_lm = character_error_rate(beam_lm_refs, beam_lm_preds)
    wer_beam_lm = word_error_rate(beam_lm_refs, beam_lm_preds)
    exact_beam_lm = sum(1 for r, p in zip(beam_lm_refs, beam_lm_preds) if r == p) / len(beam_lm_refs)

    print("\n" + "=" * 76)
    print("      DECODING BENCHMARK: GREEDY VS CTC BEAM SEARCH + 4-GRAM LM")
    print("=" * 76)
    print(f"{'Decoding Strategy':<30} | {'CER ↓':<8} | {'WER ↓':<8} | {'Exact Match ↑':<14} | {'Latency':<10}")
    print("-" * 76)
    print(f"{'Greedy Decoding (argmax)':<30} | {cer_greedy*100:>6.2f}% | {wer_greedy*100:>6.2f}% | {exact_greedy*100:>12.2f}% | {np.mean(greedy_times)*1000:>6.1f} ms")
    print(f"{'Beam Search (acoustic only)':<30} | {cer_beam*100:>6.2f}% | {wer_beam*100:>6.2f}% | {exact_beam*100:>12.2f}% | {np.mean(beam_times)*1000:>6.1f} ms")
    print(f"{'Beam Search + 4-Gram LM':<30} | {cer_beam_lm*100:>6.2f}% | {wer_beam_lm*100:>6.2f}% | {exact_beam_lm*100:>12.2f}% | {np.mean(beam_lm_times)*1000:>6.1f} ms")
    print("=" * 76)

    # Show qualitative twin corrections
    print("\n🔍 Qualitative Error Mode Analysis on Held-Out Samples:")
    corrections_found = 0
    for ref, g_pred, lm_pred in zip(greedy_refs, greedy_preds, beam_lm_preds):
        if g_pred != lm_pred:
            print(f"• Ground Truth : {ref}")
            print(f"  Greedy       : {g_pred}")
            print(f"  Beam + LM    : {lm_pred}")
            corrections_found += 1
            if corrections_found >= 3:
                break
    if corrections_found == 0:
        print("  Greedy and Beam search agreed on top candidates across sampled lines.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CTC Beam Search vs Greedy Decoding.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "a1_baseline_finetuned_best.pt",
        help="Path to model checkpoint",
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
        "--split-path",
        type=Path,
        default=REPO_ROOT / "data" / "metadata" / "writer_split.json",
        help="Path to writer_split.json",
    )
    parser.add_argument("--beam-width", type=int, default=20, help="Beam width (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.6, help="LM weight alpha (default: 0.6)")
    parser.add_argument("--beta", type=float, default=1.0, help="Length bonus beta (default: 1.0)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample limit")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        labels_csv=args.labels_csv,
        split_path=args.split_path,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

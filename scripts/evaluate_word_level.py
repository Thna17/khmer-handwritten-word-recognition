#!/usr/bin/env python3
"""Word-Level Tokenized Accuracy & Precision/Recall Evaluation.

Breaks down full 50-80 character lines into individual Khmer word tokens
to evaluate actual word recognition accuracy, precision, and recall on held-out test writers.

Usage:
    python scripts/evaluate_word_level.py
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.postprocessor import clean_khmer_orthography

RESULTS_DIR = Path("results")
MODELS = [
    ("A1 Baseline CRNN", "a1_baseline_finetuned"),
    ("A2 Transfer ResNet-18", "a2_transfer_finetuned"),
    ("A3 Transformer Encoder", "a3_transformer_finetuned"),
]


def tokenize_khmer_words(text: str) -> list[str]:
    """Extracts individual Khmer word tokens, stripping punctuation and whitespace."""
    if not text:
        return []
    # Match Khmer Unicode word blocks or alphanumeric tokens
    tokens = re.findall(r'[\u1780-\u17FF\u19E0-\u19FF\u200B]+|[a-zA-Z0-9]+', text)
    # Filter out single-character punctuation/accents that might be isolated
    return [t.strip() for t in tokens if len(t.strip()) > 0]


def evaluate_word_accuracy(targets: list[str], predictions: list[str]) -> dict[str, float]:
    """Computes:
    1. Exact Token Match (Positional Alignment Accuracy).
    2. Word Precision, Recall, and F1-Score (Bag-of-Words multiset overlap).
    3. Total words evaluated.
    """
    total_target_words = 0
    total_pred_words = 0
    exact_positional_matches = 0
    multiset_matches = 0

    correctly_predicted_words = []

    for t_line, p_line in zip(targets, predictions):
        t_words = tokenize_khmer_words(t_line)
        p_words = tokenize_khmer_words(p_line)

        total_target_words += len(t_words)
        total_pred_words += len(p_words)

        # 1. Multiset (Bag-of-Words) overlap
        t_counts = Counter(t_words)
        p_counts = Counter(p_words)
        for w, c in t_counts.items():
            common = min(c, p_counts.get(w, 0))
            multiset_matches += common
            if common > 0:
                correctly_predicted_words.extend([w] * common)

        # 2. Positional alignment match using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, t_words, p_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                exact_positional_matches += (i2 - i1)

    precision = (multiset_matches / total_pred_words * 100.0) if total_pred_words > 0 else 0.0
    recall = (multiset_matches / total_target_words * 100.0) if total_target_words > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    positional_acc = (exact_positional_matches / total_target_words * 100.0) if total_target_words > 0 else 0.0

    return {
        "total_target_words": total_target_words,
        "total_pred_words": total_pred_words,
        "positional_accuracy": positional_acc,
        "word_recall": recall,
        "word_precision": precision,
        "word_f1": f1,
        "top_correct_words": Counter(correctly_predicted_words).most_common(10),
    }


def main():
    print("=" * 70)
    print("       TOKEN-LEVEL KHMER WORD RECOGNITION EVALUATION REPORT")
    print("=" * 70)

    summary_rows = []

    for label, prefix in MODELS:
        json_path = RESULTS_DIR / f"{prefix}_results.json"
        if not json_path.exists():
            print(f"Skipping {prefix} (file not found)")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data.get("example_predictions", [])
        if not examples:
            continue

        targets = [ex["target"] for ex in examples]
        preds = [clean_khmer_orthography(ex["prediction"]) for ex in examples]

        metrics = evaluate_word_accuracy(targets, preds)

        summary_rows.append({
            "Architecture": label,
            "Target Words": metrics["total_target_words"],
            "Word Recall %": round(metrics["word_recall"], 2),
            "Word Precision %": round(metrics["word_precision"], 2),
            "Word F1 %": round(metrics["word_f1"], 2),
            "Positional Acc %": round(metrics["positional_accuracy"], 2),
        })

        print(f"\n[{label}] — Unseen Test Writers (W007, W0011):")
        print(f"  • Total Words in Test Lines : {metrics['total_target_words']}")
        print(f"  • Word Recall (Found Words) : {metrics['word_recall']:.2f}%")
        print(f"  • Word Precision            : {metrics['word_precision']:.2f}%")
        print(f"  • Word F1-Score             : {metrics['word_f1']:.2f}%")
        print(f"  • Positional Match Accuracy : {metrics['positional_accuracy']:.2f}%")
        print("  • Most Common Correct Words :")
        for word, count in metrics["top_correct_words"][:6]:
            print(f"      - '{word}': {count} times")

    print("\n" + "=" * 70)
    print("                  SUMMARY COMPARISON TABLE")
    print("=" * 70)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    print("=" * 70)

    # Save to CSV
    csv_out = RESULTS_DIR / "word_level_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n✓ Saved word-level metrics to: {csv_out}")


if __name__ == "__main__":
    main()

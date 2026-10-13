#!/usr/bin/env python3
"""Word-level evaluation report across the trained models.

The whole-line exact-match number that `run_experiment` prints as
"word_accuracy" reads 0.00% on this dataset even for a model that reads
most words correctly, because a real label is a ~58-character line and
exact match demands every one of those characters. This script reports the
word-level view instead: what share of word tokens the model actually
recovers, with precision/recall/F1 over the token multiset.

The tokenization and scoring live in `src.metrics` so that this report and
the numbers printed during training can never drift apart. Punctuation is
excluded from scoring — 8.5% of the tokens in the real labels are bare
punctuation (mostly ។) and counting those as recognized words inflates the
result without any reading ability behind it.

Usage:
    python scripts/evaluate_word_level.py
    python scripts/evaluate_word_level.py --results-dir results --postprocess
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import evaluate_predictions, tokenize_words  # noqa: E402
from src.postprocessor import clean_khmer_orthography  # noqa: E402

DEFAULT_MODELS = [
    ("A1 Baseline CRNN", "a1_baseline_finetuned"),
    ("A2 Transfer ResNet-18", "a2_transfer_finetuned"),
    ("A3 Transformer Encoder", "a3_transformer_finetuned"),
]


def load_predictions(path: Path, postprocess: bool) -> tuple[list[str], list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("example_predictions", [])
    targets = [example["target"] for example in examples]
    predictions = [example["prediction"] for example in examples]
    if postprocess:
        predictions = [clean_khmer_orthography(prediction) for prediction in predictions]
    writers = [example.get("writer_id", "?") for example in examples]
    return targets, predictions, writers


def most_recovered_words(targets: list[str], predictions: list[str], limit: int = 6) -> list[tuple[str, int]]:
    recovered: Counter[str] = Counter()
    for target, prediction in zip(targets, predictions):
        predicted_counts = Counter(tokenize_words(prediction))
        for token, count in Counter(tokenize_words(target)).items():
            overlap = min(count, predicted_counts.get(token, 0))
            if overlap:
                recovered[token] += overlap
    return recovered.most_common(limit)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--suffix", default="_finetuned",
                        help="which run to report on, e.g. '_finetuned' or '_synthetic'")
    parser.add_argument("--postprocess", action="store_true",
                        help="apply the Khmer orthography cleaner before scoring")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    models = [(label, prefix.replace("_finetuned", args.suffix)) for label, prefix in DEFAULT_MODELS]

    print("=" * 78)
    print("            WORD-LEVEL KHMER RECOGNITION REPORT — unseen test writers")
    print("=" * 78)

    summary_rows = []
    for label, prefix in models:
        path = results_dir / f"{prefix}_results.json"
        if not path.exists():
            print(f"\n[{label}] no results at {path} — skipped.")
            continue

        targets, predictions, _ = load_predictions(path, args.postprocess)
        if not targets:
            print(f"\n[{label}] results file holds no predictions — skipped.")
            continue

        metrics = evaluate_predictions(targets, predictions)
        summary_rows.append({
            "Architecture": label,
            "Lines": len(targets),
            "CER %": round(metrics["cer"] * 100, 2),
            "Word-level Acc %": round(metrics["word_level_accuracy"] * 100, 2),
            "Word Recall %": round(metrics["word_recall"] * 100, 2),
            "Word Precision %": round(metrics["word_precision"] * 100, 2),
            "Word F1 %": round(metrics["word_f1"] * 100, 2),
            "Exact-line Acc %": round(metrics["word_accuracy"] * 100, 2),
        })

        print(f"\n[{label}] — {len(targets)} test lines, {metrics['reference_word_count']:.0f} scoreable words")
        print(f"  • Character error rate      : {metrics['cer'] * 100:.2f}%")
        print(f"  • Word-level accuracy       : {metrics['word_level_accuracy'] * 100:.2f}%")
        print(f"  • Word recall / precision   : {metrics['word_recall'] * 100:.2f}% / "
              f"{metrics['word_precision'] * 100:.2f}%")
        print(f"  • Word F1                   : {metrics['word_f1'] * 100:.2f}%")
        print(f"  • Exact whole-line accuracy : {metrics['word_accuracy'] * 100:.2f}%")
        recovered = most_recovered_words(targets, predictions)
        if recovered:
            print("  • Most recovered words      : " + ", ".join(f"{w} (x{n})" for w, n in recovered))

    if not summary_rows:
        print("\nNo results files found — run the fine-tuning stage first.")
        return 1

    print("\n" + "=" * 78)
    print("                            SUMMARY COMPARISON")
    print("=" * 78)
    table = pd.DataFrame(summary_rows)
    print(table.to_string(index=False))
    print("=" * 78)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out = results_dir / "word_level_metrics.csv"
    table.to_csv(csv_out, index=False)
    print(f"\n✓ Saved word-level metrics to: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

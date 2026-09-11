"""Unified CLI entrypoint: ties dataset + tokenizer + writer split + a
chosen model (A1/A2/A3) + train.fit() + test-set evaluation together.

Every path is a CLI argument (metadata CSV, images dir, checkpoint dir,
results dir, vocab file, split file) — nothing is hardcoded to a local
directory — so the EXACT SAME command works whether you're running a
quick local smoke test on synthetic data or a real training run in
Google Colab against Google Drive paths (see notebooks/train_colab.ipynb).

The tokenizer vocabulary and the writer-disjoint split are each built
ONCE (from the first approach you train) and then reused automatically
for every subsequent approach, via get_or_create_split() / a saved
char_to_idx.json — this is what guarantees A1, A2, and A3 are compared
on the identical test set, as the course rubric requires.

Example:
    python -m src.run_experiment \\
        --approach baseline \\
        --metadata-csv data/metadata/labels.csv \\
        --images-dir data/raw \\
        --checkpoint-dir checkpoints \\
        --results-dir results \\
        --vocab-path data/metadata/char_to_idx.json \\
        --split-path data/metadata/writer_split.json \\
        --max-epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.decoder import greedy_decode
from src.metrics import evaluate_predictions
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.split import apply_split, get_or_create_split
from src.tokenizer import KhmerTokenizer
from src.train import TrainConfig, fit
from src.utils import get_device, load_checkpoint, set_seed

APPROACH_CHECKPOINT_PREFIX = {
    "baseline": "a1_baseline",
    "transfer": "a2_transfer",
    "transformer": "a3_transformer",
}


def build_model(approach: str, num_classes: int, freeze_backbone: bool) -> tuple[torch.nn.Module, dict]:
    if approach == "baseline":
        model = BaselineCRNN(num_classes=num_classes)
        config = {"model_class": "BaselineCRNN", "num_classes": num_classes}
    elif approach == "transfer":
        model = TransferCRNN(num_classes=num_classes, freeze_backbone=freeze_backbone)
        config = {"model_class": "TransferCRNN", "num_classes": num_classes, "freeze_backbone": freeze_backbone}
    elif approach == "transformer":
        model = TransformerCRNN(num_classes=num_classes)
        config = {"model_class": "TransformerCRNN", "num_classes": num_classes}
    else:
        raise ValueError(f"unknown approach {approach!r}; expected one of {list(APPROACH_CHECKPOINT_PREFIX)}")
    return model, config


def get_or_create_tokenizer(vocab_path: str | Path, all_labels: list[str]) -> KhmerTokenizer:
    vocab_path = Path(vocab_path)
    if vocab_path.exists():
        return KhmerTokenizer.load(vocab_path.parent)
    tokenizer = KhmerTokenizer.build_from_labels(all_labels)
    tokenizer.save(vocab_path.parent)
    return tokenizer


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()

    samples = load_metadata_csv(args.metadata_csv, args.images_dir)
    if not samples:
        raise ValueError(f"no samples loaded from {args.metadata_csv}")

    tokenizer = get_or_create_tokenizer(args.vocab_path, [s.label for s in samples])

    writer_ids = [s.writer_id for s in samples]
    split = get_or_create_split(
        writer_ids, args.split_path,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    train_samples, val_samples, test_samples = apply_split(samples, split)
    print(f"Writers -> train:{len(split.train_writers)} val:{len(split.val_writers)} test:{len(split.test_writers)}")
    print(f"Samples -> train:{len(train_samples)} val:{len(val_samples)} test:{len(test_samples)}")

    train_dataset = KhmerWordDataset(train_samples, tokenizer, image_height=args.image_height, max_width=args.max_width)
    val_dataset = KhmerWordDataset(val_samples, tokenizer, image_height=args.image_height, max_width=args.max_width)
    test_dataset = KhmerWordDataset(test_samples, tokenizer, image_height=args.image_height, max_width=args.max_width)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model, model_config = build_model(args.approach, tokenizer.vocab_size, args.freeze_backbone)
    checkpoint_prefix = args.checkpoint_prefix or APPROACH_CHECKPOINT_PREFIX[args.approach]

    train_config = TrainConfig(
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
    history = fit(model, train_loader, val_loader, tokenizer, model_config, config=train_config, device=device)

    # Evaluate the BEST checkpoint (lowest val CER during training) on the
    # held-out TEST set -- never touched until this single, final pass.
    best_checkpoint_path = Path(args.checkpoint_dir) / f"{checkpoint_prefix}_best.pt"
    load_checkpoint(best_checkpoint_path, model, map_location=device)
    model.to(device)

    test_metrics, example_predictions = evaluate_on_test_set(model, test_loader, tokenizer, device)
    print(f"TEST -> CER={test_metrics['cer']*100:.2f}% WER={test_metrics['wer']*100:.2f}% WordAcc={test_metrics['word_accuracy']*100:.2f}%")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{checkpoint_prefix}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "approach": args.approach,
            "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "history": history,
            "test_metrics": test_metrics,
            "example_predictions": example_predictions[:20],
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {results_path}")

    return {"history": history, "test_metrics": test_metrics}


@torch.no_grad()
def evaluate_on_test_set(model, loader, tokenizer, device) -> tuple[dict, list[dict]]:
    model.eval()
    all_targets, all_predictions, all_writer_ids = [], [], []

    for batch in loader:
        images = batch["images"].to(device)
        log_probs = model(images)
        input_lengths = torch.tensor(
            [model.compute_sequence_length(w.item()) for w in batch["image_widths"]], dtype=torch.long,
        )
        predictions = greedy_decode(log_probs, tokenizer, input_lengths=input_lengths)
        all_predictions.extend(predictions)
        all_targets.extend(batch["labels"])
        all_writer_ids.extend(batch["writer_ids"])

    metrics = evaluate_predictions(all_targets, all_predictions)
    examples = [
        {"target": t, "prediction": p, "writer_id": w, "correct": t == p}
        for t, p, w in zip(all_targets, all_predictions, all_writer_ids)
    ]
    return metrics, examples


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", required=True, choices=list(APPROACH_CHECKPOINT_PREFIX))
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--vocab-path", required=True, help="path to char_to_idx.json (created on first run, reused after)")
    parser.add_argument("--split-path", required=True, help="path to writer_split.json (created on first run, reused after)")
    parser.add_argument("--checkpoint-prefix", default=None, help="override the default a1_baseline/a2_transfer/a3_transformer prefix")

    parser.add_argument("--freeze-backbone", action="store_true", help="A2 only: freeze the pretrained backbone (linear probe)")
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

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
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.augment import build_augmentor
from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.decoder import greedy_decode
from src.metrics import evaluate_predictions
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.postprocessor import clean_khmer_orthography
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


def seed_augmentation_per_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation random stream.

    Workers are forked, so without this they inherit one identical
    `random.Random` state and draw the SAME sequence of transforms —
    verified: two generators restored from one state produce identical
    draws indefinitely. With `--num-workers 2` that halves the effective
    variety of augmented images, which is the opposite of the point.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    transform = getattr(worker_info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "rng"):
        transform.rng = random.Random(worker_info.seed + worker_id)


def build_training_pool(train_samples: list, args: argparse.Namespace, tokenizer: KhmerTokenizer) -> list:
    """Assemble the list of samples one training epoch iterates over.

    Two knobs, both aimed at the same failure: fine-tuning on 177 real lines
    drove training loss to 0.14 while validation loss stayed at 1.42, i.e.
    the model memorized seven writers rather than learning the script, and
    it also forgot most of what 10,000 synthetic samples had taught it.

      --mix-csv / --mix-size   keep a slice of the synthetic data in the
                               fine-tuning mixture, so the gradient still
                               contains Khmer orthography that the real
                               images alone cannot cover.
      --real-oversample        repeat the real samples within one epoch, so
                               mixing in synthetic data does not reduce how
                               often the model actually sees real ink.

    The mixed-in samples join the TRAINING pool only. They are never added
    to validation or test, and they bypass the writer split entirely —
    their synthetic writer ids are not in it, and routing them through
    apply_split() would (correctly) refuse them.
    """
    pool = list(train_samples) * max(1, args.real_oversample)
    if args.real_oversample > 1:
        print(f"Real samples oversampled x{args.real_oversample} -> {len(pool)} per epoch.")

    if not args.mix_csv:
        return pool

    if not args.mix_images_dir:
        raise ValueError("--mix-csv requires --mix-images-dir (the root the mix CSV's image paths are relative to).")

    mix_samples = load_metadata_csv(args.mix_csv, args.mix_images_dir)
    if not mix_samples:
        raise ValueError(f"--mix-csv {args.mix_csv!r} produced no samples.")

    # Fail loudly here rather than thousands of steps later: an unknown
    # character means the mix corpus does not match the vocabulary this run
    # is training against.
    unknown = {ch for sample in mix_samples for ch in tokenizer.normalize(sample.label)
               if ch not in tokenizer.char_to_idx}
    if unknown:
        raise ValueError(
            f"--mix-csv contains {len(unknown)} character(s) absent from the vocabulary "
            f"(e.g. {sorted(unknown)[:5]}). Rebuild the vocabulary over both corpora, or use a mix corpus "
            "that shares this one's character set."
        )

    if 0 < args.mix_size < len(mix_samples):
        mix_samples = random.Random(args.seed).sample(mix_samples, args.mix_size)

    print(f"Mixing in {len(mix_samples)} samples from {args.mix_csv}")
    return pool + mix_samples


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

    training_pool = build_training_pool(train_samples, args, tokenizer)

    # Augmentation is applied to the TRAINING pool only. Validation and test
    # must measure the model on the images as they actually are.
    augmentor = build_augmentor(args.augment_strength, seed=args.seed)
    if augmentor is not None:
        print(f"Augmentation ON (strength={args.augment_strength}) for {len(training_pool)} training images.")

    train_dataset = KhmerWordDataset(training_pool, tokenizer, image_height=args.image_height,
                                     max_width=args.max_width, transform=augmentor)
    val_dataset = KhmerWordDataset(val_samples, tokenizer, image_height=args.image_height, max_width=args.max_width)
    test_dataset = KhmerWordDataset(test_samples, tokenizer, image_height=args.image_height, max_width=args.max_width)

    loader_kwargs = {"num_workers": args.num_workers, "collate_fn": collate_fn}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        worker_init_fn=seed_augmentation_per_worker if augmentor is not None else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model, model_config = build_model(args.approach, tokenizer.vocab_size, args.freeze_backbone)
    checkpoint_prefix = args.checkpoint_prefix or APPROACH_CHECKPOINT_PREFIX[args.approach]

    if args.init_weights:
        print(f"Loading pretrained weights from {args.init_weights} ...")
        load_checkpoint(args.init_weights, model, map_location=device)
        print("✓ Pretrained weights successfully loaded!")

    train_config = TrainConfig(
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        backbone_lr_mult=args.backbone_lr_mult,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        ema_decay=args.ema_decay,
    )
    history = fit(model, train_loader, val_loader, tokenizer, model_config, config=train_config, device=device)

    # Evaluate the BEST checkpoint (lowest val CER during training) on the
    # held-out TEST set -- never touched until this single, final pass.
    best_checkpoint_path = Path(args.checkpoint_dir) / f"{checkpoint_prefix}_best.pt"
    load_checkpoint(best_checkpoint_path, model, map_location=device)
    model.to(device)

    test_metrics, example_predictions = evaluate_on_test_set(model, test_loader, tokenizer, device)
    print(
        f"TEST -> Character Accuracy = {(1.0 - test_metrics['cer'])*100:.2f}% (CER={test_metrics['cer']*100:.2f}%) | WER={test_metrics['wer']*100:.2f}%\n"
        f"        Word-Level Accuracy = {test_metrics['word_level_accuracy']*100:.2f}% "
        f"(Word F1={test_metrics['word_f1']*100:.2f}%, over {test_metrics['reference_word_count']:.0f} words)\n"
        f"        Exact Whole-Line Match = {test_metrics['word_accuracy']*100:.2f}%"
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{checkpoint_prefix}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "approach": args.approach,
            "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "history": history,
            "test_metrics": test_metrics,
            # Every test prediction, not a 20-row sample: word-level scoring
            # and per-writer error analysis both need the full set, and 50
            # short strings cost nothing to store.
            "example_predictions": example_predictions,
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
        predictions = [clean_khmer_orthography(p) for p in predictions]
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
    parser.add_argument("--init-weights", default=None, help="path to a pretrained .pt checkpoint to initialize model weights before training")
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    augmentation = parser.add_argument_group("augmentation and data mixing")
    augmentation.add_argument("--augment-strength", type=float, default=0.0,
                              help="handwriting augmentation magnitude; 0 disables it (default), 1.0 is the tested setting")
    augmentation.add_argument("--mix-csv", default=None,
                              help="extra metadata CSV (e.g. synthetic) to blend into TRAINING only")
    augmentation.add_argument("--mix-images-dir", default=None, help="images root for --mix-csv")
    augmentation.add_argument("--mix-size", type=int, default=0,
                              help="how many rows to draw from --mix-csv; 0 uses all of them")
    augmentation.add_argument("--real-oversample", type=int, default=1,
                              help="repeat the real training samples this many times per epoch")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--weight-decay", type=float, default=1e-4)
    optimization.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    optimization.add_argument("--warmup-epochs", type=int, default=0, help="cosine schedule only")
    optimization.add_argument("--backbone-lr-mult", type=float, default=1.0,
                              help="CNN learning rate as a multiple of --lr (e.g. 0.1 to fine-tune it gently)")
    optimization.add_argument("--freeze-backbone-epochs", type=int, default=0,
                              help="train the sequence model and head only for this many epochs first")
    optimization.add_argument("--ema-decay", type=float, default=0.0,
                              help="exponential moving average of weights; 0 disables, 0.999 suits small datasets")
    optimization.add_argument("--num-workers", type=int, default=0,
                              help="DataLoader worker processes; use 2-4 when augmentation is on")

    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

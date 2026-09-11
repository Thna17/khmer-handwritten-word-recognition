"""Writer-disjoint train/validation/test split.

The single most important integrity rule in this project: a writer who
appears in training must NEVER appear in validation or test. This is
what lets the reported CER/WER/Word Accuracy honestly claim the model
generalizes to unseen handwriting styles, not just memorized these
particular writers' strokes.

Splitting happens over WHOLE WRITERS, never individual images — an image
is assigned to whichever split its writer_id landed in.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WriterSplit:
    train_writers: set[str]
    val_writers: set[str]
    test_writers: set[str]

    def assert_disjoint(self) -> None:
        assert not (self.train_writers & self.val_writers), "train/val writer overlap!"
        assert not (self.train_writers & self.test_writers), "train/test writer overlap!"
        assert not (self.val_writers & self.test_writers), "val/test writer overlap!"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "train_writers": sorted(self.train_writers),
                "val_writers": sorted(self.val_writers),
                "test_writers": sorted(self.test_writers),
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WriterSplit":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        split = cls(set(data["train_writers"]), set(data["val_writers"]), set(data["test_writers"]))
        split.assert_disjoint()
        return split


def split_writers(
    writer_ids: list[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> WriterSplit:
    """Randomly assign WHOLE WRITERS to train/val/test. Deterministic
    given the same seed and writer list, regardless of input order."""
    unique_writers = sorted(set(writer_ids))  # sorted first so shuffling is order-independent
    if len(unique_writers) < 3:
        raise ValueError(f"need at least 3 distinct writers to split three ways, got {len(unique_writers)}")

    rng = random.Random(seed)
    shuffled = unique_writers[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction))
    if n_val + n_test >= n:
        raise ValueError(
            f"val_fraction+test_fraction leaves no writers for training "
            f"({n} writers total, {n_val} would go to val + {n_test} to test)"
        )

    val_writers = set(shuffled[:n_val])
    test_writers = set(shuffled[n_val:n_val + n_test])
    train_writers = set(shuffled[n_val + n_test:])

    split = WriterSplit(train_writers, val_writers, test_writers)
    split.assert_disjoint()
    return split


def get_or_create_split(
    writer_ids: list[str],
    split_path: str | Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> WriterSplit:
    """Load a previously saved split if one exists at `split_path`,
    otherwise create and save a new one. This is what lets A1, A2, and A3
    automatically reuse the EXACT SAME split — required by the course
    rubric ("same test set for every approach") — without having to
    remember to pass identical arguments by hand every time."""
    split_path = Path(split_path)
    if split_path.exists():
        return WriterSplit.load(split_path)
    split = split_writers(writer_ids, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    split.save(split_path)
    return split


def apply_split(samples: list, split: WriterSplit) -> tuple[list, list, list]:
    """samples: objects with a `.writer_id` attribute (e.g. dataset.Sample).
    Returns (train_samples, val_samples, test_samples)."""
    train = [s for s in samples if s.writer_id in split.train_writers]
    val = [s for s in samples if s.writer_id in split.val_writers]
    test = [s for s in samples if s.writer_id in split.test_writers]

    known_writers = split.train_writers | split.val_writers | split.test_writers
    unassigned = [s for s in samples if s.writer_id not in known_writers]
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} sample(s) have a writer_id not present in this split "
            f"(e.g. {unassigned[0].writer_id!r}) — rebuild the split from the FULL current "
            "writer list instead of silently dropping them."
        )
    return train, val, test

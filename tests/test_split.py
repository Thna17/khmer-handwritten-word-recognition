"""
Writer-disjoint split verification.

Run:
    .venv/bin/pytest tests/test_split.py -v
"""

import dataclasses

import pytest

from src.split import apply_split, get_or_create_split, split_writers, WriterSplit

WRITERS = [f"W{i:03d}" for i in range(1, 21)]  # 20 writers


@dataclasses.dataclass
class FakeSample:
    writer_id: str


def test_split_is_fully_disjoint():
    split = split_writers(WRITERS, val_fraction=0.15, test_fraction=0.15, seed=0)
    assert not (split.train_writers & split.val_writers)
    assert not (split.train_writers & split.test_writers)
    assert not (split.val_writers & split.test_writers)


def test_split_covers_every_writer_exactly_once():
    split = split_writers(WRITERS, seed=0)
    all_assigned = split.train_writers | split.val_writers | split.test_writers
    assert all_assigned == set(WRITERS)
    assert len(split.train_writers) + len(split.val_writers) + len(split.test_writers) == len(WRITERS)


def test_split_is_deterministic_given_same_seed_and_order_independent():
    split_a = split_writers(WRITERS, seed=42)
    split_b = split_writers(list(reversed(WRITERS)), seed=42)
    assert split_a.train_writers == split_b.train_writers
    assert split_a.val_writers == split_b.val_writers
    assert split_a.test_writers == split_b.test_writers


def test_different_seeds_can_produce_different_splits():
    split_a = split_writers(WRITERS, seed=1)
    split_b = split_writers(WRITERS, seed=2)
    assert split_a.val_writers != split_b.val_writers or split_a.test_writers != split_b.test_writers


def test_rejects_fewer_than_three_writers():
    with pytest.raises(ValueError):
        split_writers(["W001", "W002"])


def test_rejects_fractions_that_leave_no_training_writers():
    with pytest.raises(ValueError):
        split_writers(WRITERS[:3], val_fraction=0.5, test_fraction=0.5)


def test_apply_split_partitions_samples_correctly():
    split = WriterSplit(train_writers={"W001", "W002"}, val_writers={"W003"}, test_writers={"W004"})
    samples = [
        FakeSample("W001"), FakeSample("W001"), FakeSample("W002"),
        FakeSample("W003"), FakeSample("W004"), FakeSample("W004"),
    ]
    train, val, test = apply_split(samples, split)
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 2
    assert all(s.writer_id in {"W001", "W002"} for s in train)
    assert all(s.writer_id == "W003" for s in val)
    assert all(s.writer_id == "W004" for s in test)


def test_apply_split_raises_on_sample_with_unknown_writer():
    split = WriterSplit(train_writers={"W001"}, val_writers={"W002"}, test_writers={"W003"})
    samples = [FakeSample("W001"), FakeSample("W999")]
    with pytest.raises(ValueError):
        apply_split(samples, split)


def test_save_and_load_round_trip(tmp_path):
    split = split_writers(WRITERS, seed=7)
    path = tmp_path / "split.json"
    split.save(path)
    loaded = WriterSplit.load(path)
    assert loaded.train_writers == split.train_writers
    assert loaded.val_writers == split.val_writers
    assert loaded.test_writers == split.test_writers


def test_get_or_create_split_reuses_existing_file_instead_of_recomputing(tmp_path):
    path = tmp_path / "split.json"
    first = get_or_create_split(WRITERS, path, seed=123)

    # Even with a DIFFERENT seed, an existing split file must be reused
    # unchanged -- this is what guarantees A1/A2/A3 share the same split.
    second = get_or_create_split(WRITERS, path, seed=999)
    assert first.train_writers == second.train_writers
    assert first.val_writers == second.val_writers
    assert first.test_writers == second.test_writers


def test_get_or_create_split_creates_file_on_first_call(tmp_path):
    path = tmp_path / "nested" / "split.json"
    assert not path.exists()
    get_or_create_split(WRITERS, path, seed=0)
    assert path.exists()

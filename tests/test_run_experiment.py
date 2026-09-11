"""
End-to-end CLI mechanics check for run_experiment.py, using SYNTHETIC
placeholder images and a tiny epoch budget. This does NOT test model
quality (that's what test_overfit.py is for) -- it proves the CLI wires
dataset + tokenizer + split + model + train + test-evaluation together
correctly, and that vocab/split files are created once and reused across
approaches, which is what guarantees a fair comparison later.

Run:
    .venv/bin/pytest tests/test_run_experiment.py -v
"""

import csv

import pytest

from src.run_experiment import parse_args, run
from tests.test_dataset import WORDS, _render_word_image

WRITER_IDS = ["W001", "W002", "W003"]  # minimum needed to split 3 ways


@pytest.fixture
def synthetic_project(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    for i, word in enumerate(WORDS):
        font_size = 28 + (i % 5) * 6
        filename = f"{i:04d}.png"
        _render_word_image(word, font_size, images_dir / filename)
        rows.append({"image": filename, "label": word, "writer_id": WRITER_IDS[i % len(WRITER_IDS)]})

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path, csv_path, images_dir


def _base_args(tmp_path, csv_path, images_dir, approach):
    return parse_args([
        "--approach", approach,
        "--metadata-csv", str(csv_path),
        "--images-dir", str(images_dir),
        "--checkpoint-dir", str(tmp_path / "checkpoints"),
        "--results-dir", str(tmp_path / "results"),
        "--vocab-path", str(tmp_path / "metadata" / "char_to_idx.json"),
        "--split-path", str(tmp_path / "metadata" / "writer_split.json"),
        "--max-epochs", "2",
        "--batch-size", "8",
        "--val-fraction", "0.34",  # with 3 writers: 1 val, 1 test, 1 train
        "--test-fraction", "0.34",
    ])


def test_run_baseline_end_to_end_creates_all_expected_artifacts(synthetic_project):
    tmp_path, csv_path, images_dir = synthetic_project
    args = _base_args(tmp_path, csv_path, images_dir, "baseline")

    result = run(args)

    assert len(result["history"]) == 2
    assert "cer" in result["test_metrics"]
    assert (tmp_path / "checkpoints" / "a1_baseline_best.pt").exists()
    assert (tmp_path / "checkpoints" / "a1_baseline_latest.pt").exists()
    assert (tmp_path / "results" / "a1_baseline_results.json").exists()
    assert (tmp_path / "metadata" / "char_to_idx.json").exists()
    assert (tmp_path / "metadata" / "writer_split.json").exists()


def test_second_approach_reuses_the_same_vocab_and_split(synthetic_project):
    tmp_path, csv_path, images_dir = synthetic_project

    run(_base_args(tmp_path, csv_path, images_dir, "baseline"))

    import json
    split_before = json.loads((tmp_path / "metadata" / "writer_split.json").read_text())
    vocab_before = json.loads((tmp_path / "metadata" / "char_to_idx.json").read_text())

    # Run a DIFFERENT approach with a DIFFERENT seed -- if vocab/split were
    # being recomputed instead of reused, this would likely change them.
    args = _base_args(tmp_path, csv_path, images_dir, "transformer")
    args.seed = 999
    run(args)

    split_after = json.loads((tmp_path / "metadata" / "writer_split.json").read_text())
    vocab_after = json.loads((tmp_path / "metadata" / "char_to_idx.json").read_text())

    assert split_before == split_after, "the writer split must be identical across approaches for a fair comparison"
    assert vocab_before == vocab_after, "the vocabulary must be identical across approaches"

    assert (tmp_path / "checkpoints" / "a3_transformer_best.pt").exists()
    assert (tmp_path / "results" / "a3_transformer_results.json").exists()


def test_run_transfer_approach_with_frozen_backbone(synthetic_project):
    tmp_path, csv_path, images_dir = synthetic_project
    args = _base_args(tmp_path, csv_path, images_dir, "transfer")
    args.freeze_backbone = True

    result = run(args)
    assert "cer" in result["test_metrics"]
    assert (tmp_path / "checkpoints" / "a2_transfer_best.pt").exists()


def test_results_json_has_expected_structure(synthetic_project):
    tmp_path, csv_path, images_dir = synthetic_project
    run(_base_args(tmp_path, csv_path, images_dir, "baseline"))

    import json
    results = json.loads((tmp_path / "results" / "a1_baseline_results.json").read_text())
    assert results["approach"] == "baseline"
    assert results["num_trainable_params"] > 0
    assert "test_metrics" in results
    for key in ("cer", "wer", "word_accuracy"):
        assert key in results["test_metrics"]
    assert isinstance(results["example_predictions"], list)
    if results["example_predictions"]:
        example = results["example_predictions"][0]
        assert set(example.keys()) == {"target", "prediction", "writer_id", "correct"}

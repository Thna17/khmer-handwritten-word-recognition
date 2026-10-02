from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from scripts.audit_line_dataset import audit_dataset, write_snapshot


def _image(path: Path, mark: int) -> None:
    image = Image.new("L", (80, 30), 255)
    draw = ImageDraw.Draw(image)
    draw.line((5, 10 + mark, 70, 10 + mark), fill=0, width=3)
    image.save(path)


def _row(crop_id: str, status: str, page: str = "page_001.png") -> dict[str, object]:
    return {
        "crop_id": crop_id,
        "image_path": f"lines_raw/{crop_id}.png",
        "source_page": page,
        "writer_id": "W001",
        "x_min": 0,
        "y_min": 0,
        "x_max": 80,
        "y_max": 30,
        "label_raw": "",
        "label_normalized": "",
        "label_confidence": "0.00",
        "segmentation_confidence": "0.95",
        "review_required": "true",
        "review_status": status,
        "notes": "human transcription required",
    }


def test_audit_excludes_rejected_and_writes_traceable_snapshot(tmp_path):
    root = tmp_path / "data" / "handwritten_external"
    writer = root / "W001"
    (writer / "source").mkdir(parents=True)
    (writer / "lines_raw").mkdir()
    (writer / "lines_processed").mkdir()
    _image(writer / "source" / "page_001.png", 0)
    for crop_id, mark in (("W001_P001_L001", 0), ("W001_P001_L002", 2)):
        _image(writer / "lines_raw" / f"{crop_id}.png", mark)
        _image(writer / "lines_processed" / f"{crop_id}.png", mark)
    pd.DataFrame([_row("W001_P001_L001", "pending"), _row("W001_P001_L002", "rejected")]).to_csv(
        writer / "line_metadata.csv", index=False
    )

    manifest, sources, issues, summary = audit_dataset(root, tmp_path)
    assert summary["total_records"] == 2
    assert summary["usable_records"] == 1
    assert summary["rejected_records"] == 1
    assert summary["safe_to_begin_transcription"] is True
    assert not issues

    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot, manifest, sources, issues, summary)
    assert (snapshot / "dataset_manifest.csv").exists()
    assert (snapshot / "writer_summary.csv").exists()
    assert (snapshot / "page_summary.csv").exists()
    assert (snapshot / "SHA256SUMS").exists()


def test_audit_detects_duplicate_images_and_missing_processed_crop(tmp_path):
    root = tmp_path / "data" / "handwritten_external"
    writer = root / "W001"
    (writer / "source").mkdir(parents=True)
    (writer / "lines_raw").mkdir()
    (writer / "lines_processed").mkdir()
    _image(writer / "source" / "page_001.png", 0)
    for crop_id in ("W001_P001_L001", "W001_P001_L002"):
        _image(writer / "lines_raw" / f"{crop_id}.png", 0)
    _image(writer / "lines_processed" / "W001_P001_L001.png", 0)
    pd.DataFrame([_row("W001_P001_L001", "pending"), _row("W001_P001_L002", "pending")]).to_csv(
        writer / "line_metadata.csv", index=False
    )

    _, _, issues, summary = audit_dataset(root, tmp_path)
    codes = {issue["code"] for issue in issues}
    assert "duplicate_raw_image" in codes
    assert "missing_processed_crop" in codes
    assert summary["safe_to_begin_transcription"] is False


def test_audit_hashes_orphans_and_detects_orphan_duplicates(tmp_path):
    root = tmp_path / "data" / "handwritten_external"
    writer = root / "W001"
    (writer / "source").mkdir(parents=True)
    (writer / "lines_raw").mkdir()
    (writer / "lines_processed").mkdir()
    _image(writer / "source" / "page_001.png", 0)
    _image(writer / "lines_raw" / "W001_P001_L001.png", 0)
    _image(writer / "lines_processed" / "W001_P001_L001.png", 0)
    _image(writer / "lines_raw" / "orphan.png", 0)
    _image(writer / "lines_processed" / "orphan.png", 0)
    pd.DataFrame([_row("W001_P001_L001", "pending")]).to_csv(writer / "line_metadata.csv", index=False)

    _, _, issues, _ = audit_dataset(root, tmp_path)
    codes = {issue["code"] for issue in issues}
    assert "orphan_raw_crop" in codes
    assert "orphan_processed_crop" in codes
    assert "orphan_duplicate_raw_image" in codes
    assert "orphan_duplicate_processed_image" in codes
    orphan_issue = next(issue for issue in issues if issue["code"] == "orphan_raw_crop")
    assert "sha256=" in orphan_issue["detail"]

import csv
import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.transcribe_lines import DatasetStore, make_handler, validate_entry


FIELDS = [
    "crop_id", "image_path", "source_page", "writer_id", "x_min", "y_min", "x_max", "y_max",
    "label_raw", "label_normalized", "label_confidence", "segmentation_confidence", "review_required",
    "review_status", "notes",
]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_dataset(tmp_path: Path) -> tuple[DatasetStore, Path, Path]:
    writer = tmp_path / "external" / "W001"
    raw = writer / "lines_raw"
    processed = writer / "lines_processed"
    raw.mkdir(parents=True)
    processed.mkdir()
    image = Image.new("L", (100, 30), 255)
    ImageDraw.Draw(image).line((5, 15, 95, 15), fill=0, width=3)
    for directory in (raw, processed):
        image.save(directory / "W001_P001_L001.png")
        image.save(directory / "W001_P001_L002.png")
    metadata = writer / "line_metadata.csv"
    rows = [
        ["W001_P001_L001", "lines_raw/W001_P001_L001.png", "page_001.png", "W001", 0, 0, 100, 30, "", "", "0.00", "0.95", "true", "pending", "synthetic test crop"],
        ["W001_P001_L002", "lines_raw/W001_P001_L002.png", "page_001.png", "W001", 0, 0, 100, 30, "", "", "0.00", "0.95", "true", "rejected", "synthetic rejected crop"],
    ]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(FIELDS)
        writer_csv.writerows(rows)
    return DatasetStore(tmp_path / "external"), metadata, raw / "W001_P001_L001.png"


def test_filters_exclude_rejected_by_default_and_report_progress(tmp_path):
    store, _, _ = _make_dataset(tmp_path)
    assert [item["crop_id"] for item in store.list_items(status="usable")] == ["W001_P001_L001"]
    assert [item["crop_id"] for item in store.list_items(status="rejected")] == ["W001_P001_L002"]
    progress = store.progress()[0]
    assert progress["untranscribed"] == 1
    assert progress["rejected"] == 1
    assert progress["usable"] == 1


def test_atomic_save_preserves_identity_image_and_unicode_exactly(tmp_path):
    store, metadata, image_path = _make_dataset(tmp_path)
    before_hash = _hash(image_path)
    synthetic_unicode = "  ក្មេរ  "
    saved = store.save_entry(
        "W001_P001_L001",
        label_raw=synthetic_unicode,
        review_status="transcribed",
        reviewer_notes="temporary test only",
        confidence="0.87",
    )
    assert saved["label_raw"] == synthetic_unicode
    assert saved["confidence"] == "0.87"
    assert _hash(image_path) == before_hash
    assert not list(metadata.parent.glob(".line_metadata.csv.*.tmp"))

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["crop_id"] == "W001_P001_L001"
    assert row["image_path"] == "lines_raw/W001_P001_L001.png"
    assert row["label_raw"] == synthetic_unicode
    assert row["label_normalized"] == ""
    assert row["reviewer_notes"] == "temporary test only"


@pytest.mark.parametrize("status", ["verified", "transcribed"])
def test_nonempty_statuses_reject_empty_labels(status):
    with pytest.raises(ValueError, match="non-empty"):
        validate_entry("", status, "0.5")


def test_unclear_requires_blank_label_and_markers_are_forbidden():
    assert validate_entry("", "unclear", "")[1] == "unclear"
    with pytest.raises(ValueError, match="blank"):
        validate_entry("synthetic", "unclear", "0.2")
    with pytest.raises(ValueError, match="markers"):
        validate_entry("[UNCLEAR]", "transcribed", "0.2")


def test_confidence_range_is_validated():
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_entry("synthetic", "transcribed", "1.1")


def test_http_workflow_uses_temporary_data_and_rejects_empty_verification(tmp_path):
    store, _, _ = _make_dataset(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/api/list?writer_id=W001&status=untranscribed") as response:
            listing = json.load(response)
        assert listing["items"][0]["crop_id"] == "W001_P001_L001"

        payload = json.dumps(
            {
                "crop_id": "W001_P001_L001",
                "label_raw": "",
                "review_status": "verified",
                "reviewer_notes": "temporary HTTP test",
                "confidence": "",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/save", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 400

        payload = json.dumps(
            {
                "crop_id": "W001_P001_L001",
                "label_raw": "",
                "review_status": "unclear",
                "reviewer_notes": "temporary HTTP test",
                "confidence": "0.20",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/save", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        assert result["item"]["review_status"] == "unclear"
        assert result["item"]["label_raw"] == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

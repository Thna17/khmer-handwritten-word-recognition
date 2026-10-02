from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from scripts.prepare_public_dataset import build_manifest, write_outputs


def _ink_image(path: Path) -> None:
    image = Image.new("L", (80, 30), 255)
    draw = ImageDraw.Draw(image)
    draw.line((5, 15, 75, 15), fill=0, width=4)
    image.save(path)


def test_non_destructive_manifest_and_decisions(tmp_path):
    writer = tmp_path / "external" / "W001"
    crops = writer / "crops_processed"
    crops.mkdir(parents=True)
    for name in ("ok.png", "pending.png", "empty.png", "long.png"):
        _ink_image(crops / name)

    rows = [
        {"crop_id": "ok", "image_path": "crops_raw/ok.png", "source_page": "p1.png", "writer_id": "W001", "label_raw": "ទឹក", "label_normalized": "ទឹក", "review_status": "confirmed"},
        {"crop_id": "pending", "image_path": "crops_raw/pending.png", "source_page": "p1.png", "writer_id": "W001", "label_raw": "សាលា", "label_normalized": "សាលា", "review_status": "pending"},
        {"crop_id": "empty", "image_path": "crops_raw/empty.png", "source_page": "p1.png", "writer_id": "W001", "label_raw": "", "label_normalized": "", "review_status": "pending"},
        {"crop_id": "long", "image_path": "crops_raw/long.png", "source_page": "p1.png", "writer_id": "W001", "label_raw": "ក" * 30, "label_normalized": "ក" * 30, "review_status": "confirmed"},
    ]
    pd.DataFrame(rows).to_csv(writer / "line_metadata.csv", index=False)

    manifest = build_manifest(tmp_path / "external", max_label_length=25)
    decisions = dict(zip(manifest.crop_id, manifest.auto_decision))
    assert decisions == {"ok": "candidate", "pending": "review", "empty": "reject", "long": "review"}

    output = tmp_path / "output"
    summary = write_outputs(manifest, output, max_label_length=25)
    assert summary["total_records"] == 4
    assert summary["human_verified_candidates"] == 1
    assert (output / "public_manifest.csv").exists()
    assert (output / "public_sources.csv").exists()
    assert len(pd.read_csv(writer / "line_metadata.csv", keep_default_na=False)) == 4

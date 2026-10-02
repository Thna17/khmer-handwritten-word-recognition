"""Metadata manager and CSV writer/validator for handwritten crops."""

import csv
import os
from typing import List, Dict, Any

METADATA_COLUMNS = [
    "crop_id",
    "image_path",
    "source_page",
    "writer_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "label_raw",
    "label_normalized",
    "label_confidence",
    "segmentation_confidence",
    "review_required",
    "review_status",
    "notes"
]


def _as_bool(value: Any) -> bool:
    """Parse booleans safely when records came from a CSV as strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def write_metadata_csv(records: List[Dict[str, Any]], output_csv_path: str) -> None:
    """Writes metadata records to CSV with exact schema and UTF-8 encoding."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for rec in records:
            row = {
                "crop_id": rec.get("crop_id", ""),
                "image_path": rec.get("image_path", ""),
                "source_page": rec.get("source_page", ""),
                "writer_id": rec.get("writer_id", ""),
                "x_min": int(rec.get("x_min", 0)),
                "y_min": int(rec.get("y_min", 0)),
                "x_max": int(rec.get("x_max", 0)),
                "y_max": int(rec.get("y_max", 0)),
                "label_raw": rec.get("label_raw", ""),
                "label_normalized": rec.get("label_normalized", ""),
                "label_confidence": f"{float(rec.get('label_confidence', 0.0)):.2f}",
                "segmentation_confidence": f"{float(rec.get('segmentation_confidence', 0.0)):.2f}",
                "review_required": "true" if _as_bool(rec.get("review_required", False)) else "false",
                "review_status": rec.get("review_status", "pending"),
                "notes": rec.get("notes", "")
            }
            writer.writerow(row)

def read_metadata_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Reads metadata records from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records

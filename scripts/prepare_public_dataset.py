#!/usr/bin/env python3
"""Build a non-destructive audit and candidate manifest for public handwriting.

The source ``metadata.csv`` files are never modified.  This script discovers
``data/handwritten_external/W*/metadata.csv`` and writes derived artifacts to
``data/metadata/public_dataset``:

* ``public_manifest.csv`` - every discovered record plus audit columns
* ``training_candidates.csv`` - structurally usable word/short-segment crops
* ``review_queue.csv`` - records requiring visual or transcription review
* ``rejected_records.csv`` - records that cannot be used without correction
* ``public_sources.csv`` - provenance template, preserved between runs
* ``audit_summary.json`` - reproducible dataset counts and thresholds

This is an automated structural pass, not a substitute for verifying that a
handwritten crop actually matches its Khmer transcription.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


KHMER_START = 0x1780
KHMER_END = 0x17FF


def contains_khmer(text: str) -> bool:
    return any(KHMER_START <= ord(char) <= KHMER_END for char in text)


def normalized_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return unicodedata.normalize("NFC", str(value).strip())


def resolve_image(writer_dir: Path, image_path: str) -> Path:
    """Prefer the processed crop while retaining the source metadata path."""
    source = writer_dir / image_path
    processed = writer_dir / "crops_processed" / Path(image_path).name
    return processed if processed.exists() else source


def image_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "image_exists": False,
            "image_width": 0,
            "image_height": 0,
            "pixel_std": 0.0,
            "ink_ratio": 0.0,
        }
    try:
        image = Image.open(path).convert("L")
        pixels = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError):
        return {
            "image_exists": False,
            "image_width": 0,
            "image_height": 0,
            "pixel_std": 0.0,
            "ink_ratio": 0.0,
        }
    return {
        "image_exists": True,
        "image_width": image.width,
        "image_height": image.height,
        "pixel_std": round(float(pixels.std()), 4),
        "ink_ratio": round(float((pixels < 220).mean()), 6),
    }


def audit_reason(row: pd.Series, max_label_length: int) -> tuple[str, str]:
    label = row["label"]
    status = str(row.get("review_status", "pending")).strip().lower()

    if status == "rejected":
        return "reject", "source_marked_rejected"
    if not row["image_exists"]:
        return "reject", "missing_or_unreadable_image"
    if not label:
        notes = str(row.get("notes", "")).lower()
        if "human transcription required" in notes:
            return "review", "transcription_required"
        return "reject", "empty_label"
    if not contains_khmer(label):
        return "reject", "no_khmer_character"
    if row["image_width"] < 8 or row["image_height"] < 8:
        return "reject", "image_too_small"
    if row["pixel_std"] < 3.0 or row["ink_ratio"] < 0.002:
        return "reject", "blank_or_near_blank"
    if len(label) > max_label_length:
        return "review", "long_text_segment"
    if any(char.isspace() for char in label):
        return "review", "contains_whitespace"
    if status not in {"confirmed", "corrected"}:
        return "review", "transcription_not_human_verified"
    return "candidate", "verified_structurally_usable"


def ensure_source_registry(path: Path, writer_ids: list[str]) -> None:
    columns = [
        "writer_id",
        "document_id",
        "source_name",
        "source_url",
        "license_or_permission",
        "writer_identity_known",
        "notes",
    ]
    if path.exists():
        registry = pd.read_csv(path, keep_default_na=False)
    else:
        registry = pd.DataFrame(columns=columns)
    known = set(registry.get("writer_id", pd.Series(dtype=str)).astype(str))
    additions = []
    for writer_id in writer_ids:
        if writer_id not in known:
            additions.append(
                {
                    "writer_id": writer_id,
                    "document_id": f"{writer_id}_DOC001",
                    "source_name": "NEEDS_USER_INPUT",
                    "source_url": "NEEDS_USER_INPUT",
                    "license_or_permission": "TEACHER_APPROVED_SOURCE; REDISTRIBUTION_STATUS_UNKNOWN",
                    "writer_identity_known": "unknown",
                    "notes": "Fill source URL and confirm whether pages share one writer.",
                }
            )
    if additions:
        registry = pd.concat([registry, pd.DataFrame(additions)], ignore_index=True)
    registry = registry.reindex(columns=columns).sort_values("writer_id")
    registry.to_csv(path, index=False, encoding="utf-8")


def build_manifest(
    metadata_root: Path,
    max_label_length: int,
    metadata_filename: str = "line_metadata.csv",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metadata_path in sorted(metadata_root.glob(f"W*/{metadata_filename}")):
        writer_dir = metadata_path.parent
        table = pd.read_csv(metadata_path, keep_default_na=False)
        for source_row, (_, row) in enumerate(table.iterrows(), start=2):
            image_path = resolve_image(writer_dir, str(row.get("image_path", "")))
            label = normalized_label(row.get("label_normalized") or row.get("label_raw"))
            record = row.to_dict()
            record.update(
                {
                    "metadata_file": str(metadata_path),
                    "metadata_row": source_row,
                    "resolved_image_path": str(image_path.resolve()),
                    "label": label,
                    "label_codepoints": len(label),
                    **image_metrics(image_path),
                }
            )
            records.append(record)

    if not records:
        raise FileNotFoundError(f"no W*/{metadata_filename} files found under {metadata_root}")

    manifest = pd.DataFrame(records)
    decisions = manifest.apply(lambda row: audit_reason(row, max_label_length), axis=1)
    manifest["auto_decision"] = [item[0] for item in decisions]
    manifest["auto_reason"] = [item[1] for item in decisions]
    manifest["group_id"] = manifest["writer_id"].astype(str)
    return manifest


def write_outputs(manifest: pd.DataFrame, output_dir: Path, max_label_length: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "public_manifest.csv", index=False, encoding="utf-8")

    candidates = manifest[manifest["auto_decision"] == "candidate"].copy()
    review = manifest[manifest["auto_decision"] == "review"].copy()
    rejected = manifest[manifest["auto_decision"] == "reject"].copy()
    candidates.to_csv(output_dir / "training_candidates.csv", index=False, encoding="utf-8")
    review.sort_values(["auto_reason", "writer_id", "crop_id"]).to_csv(
        output_dir / "review_queue.csv", index=False, encoding="utf-8"
    )
    rejected.sort_values(["auto_reason", "writer_id", "crop_id"]).to_csv(
        output_dir / "rejected_records.csv", index=False, encoding="utf-8"
    )

    valid_labels = manifest.loc[manifest["label"] != "", "label"]
    label_counts = valid_labels.value_counts()
    group_count = int(manifest["group_id"].nunique())
    summary: dict[str, object] = {
        "automated_pass_only": True,
        "max_label_length": max_label_length,
        "total_records": int(len(manifest)),
        "writer_or_source_groups": group_count,
        "writers": sorted(manifest["writer_id"].astype(str).unique().tolist()),
        "source_pages": int(manifest[["writer_id", "source_page"]].drop_duplicates().shape[0]),
        "non_empty_labels": int((manifest["label"] != "").sum()),
        "unique_labels": int(valid_labels.nunique()),
        "singleton_labels": int((label_counts == 1).sum()),
        "unique_codepoints": len(set("".join(valid_labels.tolist()))),
        "decision_counts": {str(k): int(v) for k, v in Counter(manifest["auto_decision"]).items()},
        "reason_counts": {str(k): int(v) for k, v in Counter(manifest["auto_reason"]).items()},
        "human_verified_candidates": int(len(candidates)),
        "minimum_group_count_met": group_count >= 3,
        "ready_for_final_split": group_count >= 3 and len(candidates) > 0,
        "warning": (
            "Minimum group count is met, but verified transcriptions are still required before splitting."
            if group_count >= 3 and len(candidates) == 0
            else "At least three independent groups and verified transcriptions are required; ten or more groups is recommended."
        ),
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    ensure_source_registry(output_dir / "public_sources.csv", summary["writers"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/handwritten_external"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata/public_line_dataset"))
    parser.add_argument("--metadata-filename", default="line_metadata.csv")
    parser.add_argument("--max-label-length", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.metadata_root, args.max_label_length, args.metadata_filename)
    summary = write_outputs(manifest, args.output_dir, args.max_label_length)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

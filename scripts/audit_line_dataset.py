#!/usr/bin/env python3
"""Audit and snapshot the line-recognition dataset without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = (
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
    "notes",
)
REQUIRED_VALUES = (
    "crop_id",
    "image_path",
    "source_page",
    "writer_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "review_status",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(value: object) -> tuple[Any, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def image_metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "readable": False,
        "bytes": 0,
        "sha256": "",
        "pixel_sha256": "",
        "width": 0,
        "height": 0,
        "mode": "",
        "pixel_std": 0.0,
        "ink_ratio": 0.0,
    }
    if not result["exists"]:
        return result
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    try:
        with Image.open(path) as opened:
            opened.load()
            result["mode"] = opened.mode
            gray = np.asarray(opened.convert("L"), dtype=np.uint8)
            result["width"], result["height"] = opened.size
    except (OSError, ValueError, UnidentifiedImageError):
        return result
    result["readable"] = gray.size > 0
    if gray.size:
        result["pixel_sha256"] = hashlib.sha256(gray.tobytes()).hexdigest()
        result["pixel_std"] = round(float(gray.std()), 6)
        result["ink_ratio"] = round(float((gray < 220).mean()), 8)
    return result


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    detail: str,
    *,
    crop_id: str = "",
    writer_id: str = "",
    source_page: str = "",
    path: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "writer_id": writer_id,
            "source_page": source_page,
            "crop_id": crop_id,
            "path": path,
            "detail": detail,
        }
    )


def parse_int(value: object) -> int | None:
    try:
        number = float(str(value).strip())
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def _processed_path(writer_dir: Path, image_path: str) -> Path:
    raw = Path(image_path)
    directory = "lines_processed" if raw.parent.name == "lines_raw" else "crops_processed"
    return writer_dir / directory / raw.name


def _duplicate_issues(
    manifest: pd.DataFrame,
    issues: list[dict[str, str]],
    column: str,
    code: str,
    path_column: str,
) -> None:
    valid = manifest[(manifest[column] != "") & manifest[column].notna()]
    for digest, group in valid.groupby(column, sort=True):
        ids = sorted(group["crop_id"].astype(str).unique(), key=natural_key)
        if len(ids) <= 1:
            continue
        paths = sorted(group[path_column].astype(str).unique())
        add_issue(
            issues,
            "error",
            code,
            f"{len(ids)} crop IDs share image digest {digest}: {', '.join(ids)}",
            crop_id="|".join(ids),
            path="|".join(paths),
        )


def audit_dataset(metadata_root: Path, workspace_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]], dict[str, Any]]:
    metadata_paths = sorted(metadata_root.glob("W*/line_metadata.csv"), key=lambda p: natural_key(p.parent.name))
    if not metadata_paths:
        raise FileNotFoundError(f"No W*/line_metadata.csv files found under {metadata_root}")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    sources: dict[str, dict[str, Any]] = {}
    expected_raw: set[Path] = set()
    expected_processed: set[Path] = set()
    metadata_checksums: dict[str, str] = {}

    for metadata_path in metadata_paths:
        writer_dir = metadata_path.parent
        metadata_checksums[relative(metadata_path, workspace_root)] = sha256_file(metadata_path)
        table = pd.read_csv(metadata_path, keep_default_na=False, dtype=str)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
        for column in missing_columns:
            add_issue(
                issues,
                "error",
                "missing_metadata_column",
                f"Required column {column!r} is absent",
                writer_id=writer_dir.name,
                path=relative(metadata_path, workspace_root),
            )

        for metadata_row, (_, series) in enumerate(table.iterrows(), start=2):
            row = {column: str(series.get(column, "")).strip() for column in table.columns}
            crop_id = row.get("crop_id", "")
            writer_id = row.get("writer_id", "")
            source_page = row.get("source_page", "")
            review_status = row.get("review_status", "").lower()
            usable = review_status != "rejected"

            for field in REQUIRED_VALUES:
                if not row.get(field, ""):
                    add_issue(
                        issues,
                        "error",
                        "missing_metadata_value",
                        f"Required field {field!r} is empty at CSV row {metadata_row}",
                        crop_id=crop_id,
                        writer_id=writer_id or writer_dir.name,
                        source_page=source_page,
                        path=relative(metadata_path, workspace_root),
                    )
            if writer_id and writer_id != writer_dir.name:
                add_issue(
                    issues,
                    "error",
                    "writer_directory_mismatch",
                    f"Metadata writer {writer_id!r} does not match directory {writer_dir.name!r}",
                    crop_id=crop_id,
                    writer_id=writer_id,
                    source_page=source_page,
                )

            raw_path = writer_dir / row.get("image_path", "")
            processed_path = _processed_path(writer_dir, row.get("image_path", ""))
            source_path = writer_dir / "source" / source_page
            expected_raw.add(raw_path.resolve())
            expected_processed.add(processed_path.resolve())
            raw = image_metrics(raw_path)
            processed = image_metrics(processed_path)

            source_key = relative(source_path, workspace_root)
            if source_key not in sources:
                source_metrics = image_metrics(source_path)
                sources[source_key] = {
                    "writer_id": writer_id or writer_dir.name,
                    "source_page": source_page,
                    "source_path": source_key,
                    **{f"source_{key}": value for key, value in source_metrics.items()},
                }
            source = sources[source_key]

            for kind, path, metrics in (("raw", raw_path, raw), ("processed", processed_path, processed)):
                if not metrics["exists"]:
                    add_issue(
                        issues,
                        "error" if usable else "warning",
                        f"missing_{kind}_crop",
                        f"{kind.title()} crop file is missing",
                        crop_id=crop_id,
                        writer_id=writer_id,
                        source_page=source_page,
                        path=relative(path, workspace_root),
                    )
                elif metrics["bytes"] == 0 or not metrics["readable"]:
                    add_issue(
                        issues,
                        "error" if usable else "warning",
                        f"empty_or_unreadable_{kind}_crop",
                        f"{kind.title()} crop is empty or unreadable",
                        crop_id=crop_id,
                        writer_id=writer_id,
                        source_page=source_page,
                        path=relative(path, workspace_root),
                    )
            if not source["source_exists"] or not source["source_readable"]:
                add_issue(
                    issues,
                    "error" if usable else "warning",
                    "missing_or_unreadable_source",
                    "Source page is missing or unreadable",
                    crop_id=crop_id,
                    writer_id=writer_id,
                    source_page=source_page,
                    path=source_key,
                )

            bbox = {key: parse_int(row.get(key, "")) for key in ("x_min", "y_min", "x_max", "y_max")}
            bbox_valid = all(value is not None for value in bbox.values())
            if bbox_valid:
                bbox_valid = bool(
                    bbox["x_min"] >= 0
                    and bbox["y_min"] >= 0
                    and bbox["x_min"] < bbox["x_max"]
                    and bbox["y_min"] < bbox["y_max"]
                    and (
                        not source["source_readable"]
                        or (
                            bbox["x_max"] <= source["source_width"]
                            and bbox["y_max"] <= source["source_height"]
                        )
                    )
                )
            if not bbox_valid:
                add_issue(
                    issues,
                    "error" if usable else "warning",
                    "invalid_bounding_box",
                    f"Invalid or out-of-source bounding box: {bbox}",
                    crop_id=crop_id,
                    writer_id=writer_id,
                    source_page=source_page,
                )

            if raw["readable"]:
                expected_width = (bbox["x_max"] - bbox["x_min"]) if bbox_valid else None
                expected_height = (bbox["y_max"] - bbox["y_min"]) if bbox_valid else None
                if bbox_valid and (raw["width"] != expected_width or raw["height"] != expected_height):
                    add_issue(
                        issues,
                        "error" if usable else "warning",
                        "crop_bbox_dimension_mismatch",
                        f"Raw crop {raw['width']}x{raw['height']} != bbox {expected_width}x{expected_height}",
                        crop_id=crop_id,
                        writer_id=writer_id,
                        source_page=source_page,
                    )
                width, height = raw["width"], raw["height"]
                aspect = width / height if height else 0.0
                if width < 16 or height < 12 or width > 2500 or height > 300 or aspect < 0.5 or aspect > 100:
                    add_issue(
                        issues,
                        "warning",
                        "extreme_crop_dimensions",
                        f"Raw crop dimensions are {width}x{height} (aspect {aspect:.2f})",
                        crop_id=crop_id,
                        writer_id=writer_id,
                        source_page=source_page,
                        path=relative(raw_path, workspace_root),
                    )
                if raw["pixel_std"] < 3.0 or raw["ink_ratio"] < 0.002:
                    add_issue(
                        issues,
                        "error" if usable else "warning",
                        "blank_or_near_blank_crop",
                        f"Raw crop std={raw['pixel_std']:.3f}, ink_ratio={raw['ink_ratio']:.6f}",
                        crop_id=crop_id,
                        writer_id=writer_id,
                        source_page=source_page,
                        path=relative(raw_path, workspace_root),
                    )
            if raw["readable"] and processed["readable"] and (
                raw["width"] != processed["width"] or raw["height"] != processed["height"]
            ):
                add_issue(
                    issues,
                    "warning",
                    "raw_processed_dimension_mismatch",
                    f"Raw {raw['width']}x{raw['height']} vs processed {processed['width']}x{processed['height']}",
                    crop_id=crop_id,
                    writer_id=writer_id,
                    source_page=source_page,
                )

            rows.append(
                {
                    **{column: row.get(column, "") for column in REQUIRED_COLUMNS},
                    "metadata_file": relative(metadata_path, workspace_root),
                    "metadata_row": metadata_row,
                    "audit_class": "rejected" if not usable else "usable",
                    "raw_path": relative(raw_path, workspace_root),
                    "processed_path": relative(processed_path, workspace_root),
                    "source_path": source_key,
                    **{f"raw_{key}": value for key, value in raw.items()},
                    **{f"processed_{key}": value for key, value in processed.items()},
                    "bbox_valid": bbox_valid,
                }
            )

    manifest = pd.DataFrame(rows)
    source_manifest = pd.DataFrame(sources.values())

    for crop_id, group in manifest.groupby("crop_id", dropna=False, sort=True):
        if not crop_id or len(group) > 1:
            add_issue(
                issues,
                "error",
                "duplicate_crop_id" if crop_id else "empty_crop_id",
                f"Crop ID {crop_id!r} appears {len(group)} times",
                crop_id=str(crop_id),
                writer_id="|".join(sorted(group["writer_id"].astype(str).unique(), key=natural_key)),
            )
    _duplicate_issues(manifest, issues, "raw_sha256", "duplicate_raw_image", "raw_path")
    _duplicate_issues(manifest, issues, "processed_pixel_sha256", "duplicate_processed_pixels", "processed_path")

    if not source_manifest.empty:
        valid_sources = source_manifest[source_manifest["source_sha256"] != ""]
        for digest, group in valid_sources.groupby("source_sha256", sort=True):
            paths = sorted(group["source_path"].astype(str).unique())
            if len(paths) > 1:
                add_issue(
                    issues,
                    "error",
                    "duplicate_source_image",
                    f"{len(paths)} source pages share digest {digest}",
                    path="|".join(paths),
                )

    actual_raw: set[Path] = set()
    actual_processed: set[Path] = set()
    for writer_dir in metadata_root.glob("W*"):
        for directory, target in ((writer_dir / "lines_raw", actual_raw), (writer_dir / "lines_processed", actual_processed)):
            if directory.is_dir():
                target.update(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    orphan_raw = actual_raw - expected_raw
    orphan_processed = actual_processed - expected_processed
    for kind, paths, expected, orphans in (
        ("raw", actual_raw, expected_raw, orphan_raw),
        ("processed", actual_processed, expected_processed, orphan_processed),
    ):
        digest_paths: defaultdict[str, list[Path]] = defaultdict(list)
        for path in sorted(paths):
            digest_paths[sha256_file(path)].append(path)
        for digest, duplicate_paths in sorted(digest_paths.items()):
            if len(duplicate_paths) > 1 and any(path in orphans for path in duplicate_paths):
                add_issue(
                    issues,
                    "warning",
                    f"orphan_duplicate_{kind}_image",
                    f"An unreferenced {kind} crop shares SHA-256 {digest} with another file",
                    path="|".join(relative(path, workspace_root) for path in duplicate_paths),
                )

    for kind, paths in (("raw", orphan_raw), ("processed", orphan_processed)):
        for path in sorted(paths):
            metrics = image_metrics(path)
            add_issue(
                issues,
                "warning",
                f"orphan_{kind}_crop",
                (
                    f"{kind.title()} crop is not referenced by metadata; "
                    f"bytes={metrics['bytes']}, dimensions={metrics['width']}x{metrics['height']}, "
                    f"sha256={metrics['sha256']}"
                ),
                path=relative(path, workspace_root),
            )

    issue_counts = Counter(issue["code"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    blocking_codes = sorted({issue["code"] for issue in issues if issue["severity"] == "error"})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_root": relative(metadata_root, workspace_root),
        "total_records": int(len(manifest)),
        "usable_records": int((manifest["audit_class"] == "usable").sum()),
        "rejected_records": int((manifest["audit_class"] == "rejected").sum()),
        "writer_count": int(manifest["writer_id"].nunique()),
        "source_page_count": int(manifest[["writer_id", "source_page"]].drop_duplicates().shape[0]),
        "non_empty_labels": int(((manifest["label_raw"] != "") | (manifest["label_normalized"] != "")).sum()),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "blocking_issue_codes": blocking_codes,
        "safe_to_begin_transcription": not blocking_codes,
        "metadata_sha256": metadata_checksums,
        "thresholds": {
            "blank_pixel_std_lt": 3.0,
            "blank_ink_ratio_lt": 0.002,
            "extreme_width": "<16 or >2500",
            "extreme_height": "<12 or >300",
            "extreme_aspect_ratio": "<0.5 or >100",
        },
    }
    return manifest, source_manifest, issues, summary


def _status_summary(manifest: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = manifest.groupby(group_columns + ["audit_class"], dropna=False).size().unstack(fill_value=0)
    for column in ("usable", "rejected"):
        if column not in grouped:
            grouped[column] = 0
    result = grouped.reset_index()
    result["total"] = result["usable"] + result["rejected"]
    return result[group_columns + ["usable", "rejected", "total"]].sort_values(group_columns, key=lambda column: column.map(natural_key))


def write_snapshot(
    snapshot_dir: Path,
    manifest: pd.DataFrame,
    source_manifest: pd.DataFrame,
    issues: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise FileExistsError(f"Snapshot directory already exists and is not empty: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest.sort_values(["writer_id", "source_page", "crop_id"], key=lambda column: column.map(natural_key)).to_csv(
        snapshot_dir / "dataset_manifest.csv", index=False, encoding="utf-8"
    )
    source_manifest.sort_values(["writer_id", "source_page"], key=lambda column: column.map(natural_key)).to_csv(
        snapshot_dir / "source_manifest.csv", index=False, encoding="utf-8"
    )
    _status_summary(manifest, ["writer_id"]).to_csv(snapshot_dir / "writer_summary.csv", index=False, encoding="utf-8")
    _status_summary(manifest, ["writer_id", "source_page"]).to_csv(snapshot_dir / "page_summary.csv", index=False, encoding="utf-8")
    issue_columns = ["severity", "code", "writer_id", "source_page", "crop_id", "path", "detail"]
    pd.DataFrame(issues, columns=issue_columns).sort_values(["severity", "code", "writer_id", "crop_id"]).to_csv(
        snapshot_dir / "issues.csv", index=False, encoding="utf-8"
    )
    with (snapshot_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    output_files = sorted(path for path in snapshot_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (snapshot_dir / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for path in output_files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/handwritten_external"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, sources, issues, summary = audit_dataset(args.metadata_root, args.workspace_root)
    write_snapshot(args.snapshot_dir, manifest, sources, issues, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

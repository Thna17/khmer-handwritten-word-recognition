"""
Tests for src/data_preparation module and external handwritten dataset integration.
"""

import os
import numpy as np
import pytest

from src.data_preparation.crop_processor import preprocess_training_crop
from src.data_preparation.metadata import read_metadata_csv, write_metadata_csv
from src.data_preparation.quality_control import run_quality_control
from src.data_preparation.line_segmentation import HandwrittenLine, line_bounding_box, line_bounding_boxes
from src.data_preparation.page_preprocessing import extract_handwriting_roi
from src.data_preparation.transcription import LINE_GROUND_TRUTH_P1, assign_transcription
from src.tokenizer import KhmerTokenizer


def test_imports_and_structure():
    from src.data_preparation import (
        annotated_visualizer,
        crop_processor,
        line_segmentation,
        metadata,
        page_preprocessing,
        quality_control,
        transcription,
        unicode_diagnostics,
        word_segmentation,
    )
    assert crop_processor is not None
    assert transcription is not None


def test_crop_processor():
    dummy_crop = np.full((50, 100, 3), 255, dtype=np.uint8)
    processed = preprocess_training_crop(dummy_crop)
    assert processed.shape == (50, 100)
    assert processed.dtype == np.uint8


def test_new_writer_uses_generic_page_roi_not_w001_form_crop():
    page = np.full((960, 720, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W003")
    assert roi == {"x_min": 29, "y_min": 29, "x_max": 691, "y_max": 912}


def test_w005_ruled_form_roi_excludes_printed_header_and_page_number():
    page = np.full((1527, 1284, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W005")
    assert roi == {"x_min": 51, "y_min": 214, "x_max": 1233, "y_max": 1435}


def test_w006_notebook_roi_excludes_date_box():
    page = np.full((960, 720, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W006")
    assert roi == {"x_min": 29, "y_min": 211, "x_max": 698, "y_max": 922}


def test_w007_notebook_roi_preserves_opening_strokes_and_excludes_footer():
    page = np.full((1647, 1170, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W007")
    assert roi == {"x_min": 23, "y_min": 25, "x_max": 1147, "y_max": 1573}


def test_w008_compact_notebook_roi_excludes_form_header_and_footer():
    page = np.full((1283, 860, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W008")
    assert roi == {"x_min": 34, "y_min": 80, "x_max": 826, "y_max": 1219}


def test_w009_full_page_roi_preserves_edge_handwriting():
    page = np.full((1796, 1284, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W009")
    assert roi == {"x_min": 13, "y_min": 18, "x_max": 1271, "y_max": 1778}


def test_w0010_form_roi_excludes_personal_information_header():
    page = np.full((1800, 1273, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W0010")
    assert roi == {"x_min": 32, "y_min": 585, "x_max": 1235, "y_max": 1746}


def test_w0011_square_notebook_roi_preserves_prompt_and_final_line():
    page = np.full((1277, 1284, 3), 255, dtype=np.uint8)
    _, roi = extract_handwriting_roi(page, writer_id="W0011")
    assert roi == {"x_min": 26, "y_min": 19, "x_max": 1271, "y_max": 1251}


def test_line_bounding_box_contains_components_and_clamps():
    line = HandwrittenLine(
        line_index=0,
        center_y=15,
        y_min=2,
        y_max=28,
        components=[(3, 4, 10, 8, 50), (70, 3, 25, 20, 100)],
    )
    assert line_bounding_box(line, (30, 100), padding_x=8, padding_y=5) == {
        "x_min": 0,
        "y_min": 0,
        "x_max": 100,
        "y_max": 30,
    }


def test_line_bounding_boxes_use_midpoints_instead_of_crossing_components():
    lines = [
        HandwrittenLine(0, 20, 5, 38, [(10, 5, 50, 33, 100)]),
        HandwrittenLine(1, 50, 30, 70, [(12, 30, 60, 40, 100)]),
    ]
    boxes = line_bounding_boxes(lines, (80, 100), 0, 80, boundary_overlap=2)
    assert boxes[0]["y_min"] == 3
    assert boxes[0]["y_max"] == 37
    assert boxes[1]["y_min"] == 33
    assert boxes[1]["y_max"] == 72


def test_assign_transcription():
    res = assign_transcription("page_001.png", 1, 0, 14, {"review_required": False})
    assert isinstance(res, dict)
    assert res["label_normalized"] == LINE_GROUND_TRUTH_P1[1][0]["token"]
    assert res["label_confidence"] > 0.8
    assert res["review_status"] in ("accepted", "pending", "auto_aligned")


def test_metadata_writer_preserves_string_false(tmp_path):
    output = tmp_path / "metadata.csv"
    write_metadata_csv(
        [{"crop_id": "x", "review_required": "false"}],
        str(output),
    )
    assert read_metadata_csv(str(output))[0]["review_required"] == "false"


def test_unknown_page_never_reuses_another_pages_transcription():
    res = assign_transcription(
        "page_002.png", 0, 0, 5, {"review_required": False}, writer_id="W002"
    )
    assert res["label_normalized"] == ""
    assert res["review_required"] is True
    assert res["review_status"] == "pending"
    assert "human transcription required" in res["notes"]


def test_handwritten_metadata_qc_if_present():
    metadata_path = "data/handwritten_external/W001/metadata.csv"
    if not os.path.exists(metadata_path):
        pytest.skip(f"{metadata_path} not found")

    records = read_metadata_csv(metadata_path)
    assert len(records) == 1477

    # Run QC for Page 1 records against Page 1 source image
    p1_records = [r for r in records if r.get("source_page") == "page_001.png"]
    assert len(p1_records) > 0
    source_p1 = "data/handwritten_external/W001/source/page_001.png"
    report = run_quality_control(p1_records, "data/handwritten_external/W001", source_p1)
    assert len(report["errors"]) == 0, f"QC errors found: {report['errors']}"


def test_transcriptions_compatible_with_tokenizer():
    metadata_path = "data/handwritten_external/W001/metadata.csv"
    if not os.path.exists(metadata_path):
        pytest.skip(f"{metadata_path} not found")

    records = read_metadata_csv(metadata_path)
    unique_labels = list({r["label_normalized"] for r in records if r.get("label_normalized")})
    assert len(unique_labels) > 0

    tokenizer = KhmerTokenizer.build_from_labels(unique_labels)
    for label in unique_labels:
        token_ids = tokenizer.encode(label)
        decoded = tokenizer.decode(token_ids)
        assert decoded == label, f"Tokenizer roundtrip failed for: {label}"


def test_writer_2_metadata_qc_if_present():
    metadata_path = "data/handwritten_external/W002/metadata.csv"
    if not os.path.exists(metadata_path):
        pytest.skip(f"{metadata_path} not found")

    records = read_metadata_csv(metadata_path)
    page_1_records = [r for r in records if r.get("source_page") == "page_001.png"]
    assert len(page_1_records) == 118

    source_p1 = "data/handwritten_external/W002/source/page_001.png"
    report = run_quality_control(page_1_records, "data/handwritten_external/W002", source_p1)
    assert len(report["errors"]) == 0, f"QC errors found: {report['errors']}"

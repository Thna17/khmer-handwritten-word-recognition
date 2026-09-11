"""
Scan-to-crop pipeline verification, using SYNTHETIC "fake scanned sheet"
images (solid-color boxes for exact geometric checks, rendered Khmer
words for the end-to-end file/metadata check) — no real scans exist yet.

Run:
    .venv/bin/pytest tests/test_scan_pipeline.py -v
"""

import csv

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.scan_pipeline import (
    GridLayout,
    crop_word_boxes,
    deskew_page,
    load_word_list,
    process_sheet,
    visualize_calibration,
)

FONT_PATH = "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"


# ---------------------------------------------------------------------
# GridLayout calibration + box_region arithmetic
# ---------------------------------------------------------------------

def test_calibrate_derives_expected_box_geometry():
    layout = GridLayout.calibrate(
        box_top_left=(10, 10),
        box_bottom_right=(60, 40),      # box_width=50, box_height=30
        next_column_top_left=(80, 10),  # column_pitch=70
        next_row_top_left=(10, 60),     # row_pitch=50
        num_columns=3,
        num_rows=2,
    )
    assert layout.box_width == 50
    assert layout.box_height == 30
    assert layout.column_pitch == 70
    assert layout.row_pitch == 50


@pytest.mark.parametrize("index,expected", [
    (0, (10, 10, 60, 40)),     # row 0, col 0
    (1, (80, 10, 130, 40)),    # row 0, col 1
    (2, (150, 10, 200, 40)),   # row 0, col 2
    (3, (10, 60, 60, 90)),     # row 1, col 0 (wraps to next row)
    (5, (150, 60, 200, 90)),   # row 1, col 2 (last box)
])
def test_box_region_matches_hand_computed_positions(index, expected):
    layout = GridLayout.calibrate(
        box_top_left=(10, 10), box_bottom_right=(60, 40),
        next_column_top_left=(80, 10), next_row_top_left=(10, 60),
        num_columns=3, num_rows=2,
    )
    assert layout.box_region(index) == expected


def test_box_region_out_of_range_raises():
    layout = GridLayout.calibrate((0, 0), (10, 10), (20, 0), (0, 20), num_columns=2, num_rows=2)
    with pytest.raises(ValueError):
        layout.box_region(4)  # only indices 0-3 exist in a 2x2 grid


@pytest.mark.parametrize("kwargs", [
    dict(box_top_left=(10, 10), box_bottom_right=(5, 40), next_column_top_left=(80, 10), next_row_top_left=(10, 60)),   # bad width
    dict(box_top_left=(10, 10), box_bottom_right=(60, 5), next_column_top_left=(80, 10), next_row_top_left=(10, 60)),   # bad height
    dict(box_top_left=(10, 10), box_bottom_right=(60, 40), next_column_top_left=(5, 10), next_row_top_left=(10, 60)),   # bad column pitch
    dict(box_top_left=(10, 10), box_bottom_right=(60, 40), next_column_top_left=(80, 10), next_row_top_left=(10, 5)),   # bad row pitch
])
def test_calibrate_rejects_degenerate_calibration_points(kwargs):
    with pytest.raises(ValueError):
        GridLayout.calibrate(num_columns=3, num_rows=2, **kwargs)


# ---------------------------------------------------------------------
# crop_word_boxes: exact geometric correctness via solid-color fills
# ---------------------------------------------------------------------

def test_crop_word_boxes_extracts_geometrically_correct_regions():
    layout = GridLayout.calibrate(
        box_top_left=(10, 10), box_bottom_right=(60, 40),
        next_column_top_left=(80, 10), next_row_top_left=(10, 60),
        num_columns=3, num_rows=2,
    )
    image = Image.new("RGB", (250, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for index in range(6):
        draw.rectangle(layout.box_region(index), fill=colors[index])

    word_ids = [101, 102, 103, 104, 105, 106]
    crops = crop_word_boxes(image, layout, word_ids)

    assert [wid for wid, _ in crops] == word_ids
    for (word_id, crop), expected_color in zip(crops, colors):
        assert crop.size == (50, 30)
        center_pixel = crop.getpixel((25, 15))
        assert center_pixel == expected_color, f"word_id {word_id}: expected {expected_color}, got {center_pixel}"


def test_crop_word_boxes_rejects_more_word_ids_than_grid_has_boxes():
    layout = GridLayout.calibrate((0, 0), (10, 10), (20, 0), (0, 20), num_columns=2, num_rows=2)  # 4 boxes
    image = Image.new("L", (100, 100), color=255)
    with pytest.raises(ValueError):
        crop_word_boxes(image, layout, word_ids_in_order=[1, 2, 3, 4, 5])


# ---------------------------------------------------------------------
# process_sheet: end-to-end file + metadata CSV correctness
# ---------------------------------------------------------------------

def _draw_khmer_word(draw: ImageDraw.ImageDraw, word: str, center: tuple[float, float], font_size: int = 26):
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((center[0] - w / 2 - bbox[0], center[1] - h / 2 - bbox[1]), word, font=font, fill=0)


@pytest.fixture
def word_list() -> dict[int, str]:
    return {1: "ទឹក", 2: "ភ្លើង", 3: "ខ្យល់", 4: "ថ្ងៃ", 5: "ខែ", 6: "ឆ្នាំ"}


@pytest.fixture
def synthetic_sheet_image(tmp_path, word_list):
    layout = GridLayout.calibrate(
        box_top_left=(20, 20), box_bottom_right=(120, 70),
        next_column_top_left=(140, 20), next_row_top_left=(20, 90),
        num_columns=3, num_rows=2,
    )
    image = Image.new("L", (280, 180), color=255)
    draw = ImageDraw.Draw(image)
    word_ids_in_order = [1, 2, 3, 4, 5, 6]
    for index, word_id in enumerate(word_ids_in_order):
        box = layout.box_region(index)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        _draw_khmer_word(draw, word_list[word_id], center)

    path = tmp_path / "fake_sheet_W001.png"
    image.save(path)
    return path, layout, word_ids_in_order


def test_process_sheet_writes_correctly_sized_images_and_metadata(tmp_path, synthetic_sheet_image, word_list):
    image_path, layout, word_ids_in_order = synthetic_sheet_image
    output_dir = tmp_path / "images"
    csv_path = tmp_path / "metadata.csv"

    rows = process_sheet(image_path, layout, word_ids_in_order, word_list, "W001", output_dir, csv_path)

    assert len(rows) == 6
    for row in rows:
        assert (output_dir / row["image"]).exists()
        with Image.open(output_dir / row["image"]) as crop:
            assert crop.size == (100, 50)
        assert row["writer_id"] == "W001"

    with open(csv_path, encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 6
    assert {r["label"] for r in csv_rows} == set(word_list.values())


def test_process_sheet_appends_across_multiple_writers_without_duplicate_header(tmp_path, synthetic_sheet_image, word_list):
    image_path, layout, word_ids_in_order = synthetic_sheet_image
    output_dir = tmp_path / "images"
    csv_path = tmp_path / "metadata.csv"

    process_sheet(image_path, layout, word_ids_in_order, word_list, "W001", output_dir, csv_path)
    process_sheet(image_path, layout, word_ids_in_order, word_list, "W002", output_dir, csv_path)

    with open(csv_path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    assert lines[0].strip() == "image,label,writer_id"
    assert sum(1 for line in lines if line.strip() == "image,label,writer_id") == 1  # header appears exactly once

    with open(csv_path, encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 12
    assert {r["writer_id"] for r in csv_rows} == {"W001", "W002"}


def test_process_sheet_rejects_unknown_word_id_instead_of_mislabeling(tmp_path, synthetic_sheet_image, word_list):
    image_path, layout, _ = synthetic_sheet_image
    with pytest.raises(ValueError):
        process_sheet(image_path, layout, [999], word_list, "W001", tmp_path / "images", tmp_path / "metadata.csv")


def test_load_word_list_matches_project_master_list():
    mapping = load_word_list("data/metadata/word_list.csv")
    assert len(mapping) == 30
    assert set(mapping.keys()) == set(range(1, 31))
    assert mapping[1] == "ទឹក"
    assert mapping[30] == "តូច"


# ---------------------------------------------------------------------
# visualize_calibration: sanity check it runs and draws something
# ---------------------------------------------------------------------

def test_visualize_calibration_draws_a_visible_rectangle():
    layout = GridLayout.calibrate((10, 10), (60, 40), (80, 10), (10, 60), num_columns=1, num_rows=1)
    image = Image.new("L", (100, 100), color=255)
    preview = visualize_calibration(image, layout, num_boxes=1)

    assert preview.size == image.size
    assert preview.mode == "RGB"
    # A pixel ON the drawn rectangle's top edge should no longer be white.
    assert preview.getpixel((30, 10)) != (255, 255, 255)


# ---------------------------------------------------------------------
# deskew_page: exact geometric proof using a SHEARED PARALLELOGRAM quad.
# A parallelogram-to-rectangle mapping is a pure affine transform, so any
# point at fractional position (s, t) within the source parallelogram
# (TL + s*width_vector + t*height_vector) must land EXACTLY at
# (s*output_W, t*output_H) in the output -- this is hand-computable
# without needing rotation/homography math.
# ---------------------------------------------------------------------

def test_deskew_page_recovers_content_from_a_sheared_quad():
    top_left = (50.0, 50.0)
    width_vector = (200.0, 20.0)   # sheared: moving "right" also drifts down slightly
    height_vector = (20.0, 200.0)  # sheared: moving "down" also drifts right slightly

    def source_point(s: float, t: float) -> tuple[float, float]:
        return (
            top_left[0] + s * width_vector[0] + t * height_vector[0],
            top_left[1] + s * width_vector[1] + t * height_vector[1],
        )

    top_right = source_point(1, 0)
    bottom_left = source_point(0, 1)
    bottom_right = source_point(1, 1)

    output_size = (200, 200)
    marker_fractions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0)]

    source = Image.new("RGB", (350, 350), color=(255, 255, 255))
    draw = ImageDraw.Draw(source)
    marker_radius = 6
    for (s, t), color in zip(marker_fractions, colors):
        x, y = source_point(s, t)
        draw.ellipse((x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius), fill=color)

    corners = (top_left, top_right, bottom_right, bottom_left)
    output = deskew_page(source, corners, output_size)

    assert output.size == output_size
    for (s, t), expected_color in zip(marker_fractions, colors):
        expected_x = round(s * output_size[0])
        expected_y = round(t * output_size[1])
        actual_color = output.getpixel((expected_x, expected_y))
        assert actual_color == expected_color, (
            f"marker at fraction ({s},{t}) expected at output pixel "
            f"({expected_x},{expected_y}) to be {expected_color}, got {actual_color}"
        )


def test_deskew_page_identity_case_is_a_plain_crop_and_resize():
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    ImageDraw.Draw(image).rectangle((40, 40, 60, 60), fill=(0, 0, 255))

    corners = ((0, 0), (100, 0), (100, 100), (0, 100))  # the image's own corners: no real skew
    output = deskew_page(image, corners, output_size=(100, 100))

    assert output.getpixel((50, 50)) == (0, 0, 255)
    assert output.getpixel((5, 5)) == (255, 255, 255)

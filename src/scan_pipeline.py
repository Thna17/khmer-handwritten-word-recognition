"""Turn one scanned/photographed collection sheet into per-word images
plus metadata CSV rows.

The collection sheet (see the published "Khmer Handwriting Collection
Sheet") lays its word-boxes out as a REGULAR GRID: every box the same
size, evenly spaced, in row-major reading order (left-to-right, top-to-
bottom). Printing/scanning/photographing does NOT preserve the exact CSS
pixel layout — paper size, print margins, scanner DPI, and how many
pages the browser split the grid across are all unpredictable from here.
So this pipeline never hardcodes pixel coordinates from the HTML/CSS.
Instead:

  1. (phone photos only) `deskew_page()` perspective-corrects a photo
     into a clean rectangle, given the page's 4 corners. Flatbed scans
     are already rectangular and can skip this step entirely.
  2. `GridLayout.calibrate()` locates the whole grid from just 4 points
     marked ONCE per physical page layout (not once per sheet!): one
     box's two opposite corners (for box size), plus the top-left corner
     of the next box to the right and the next box below (for spacing).
     Uniform spacing fills in every other box position automatically.
  3. `crop_word_boxes()` slices out each box, paired with the word_id
     that belongs at that grid position (position -> word is fixed by
     the sheet's own layout, supplied explicitly as `word_ids_in_order`
     since a real print run may split the 30 words across pages
     unpredictably — never assumed here).
  4. `process_sheet()` saves the crops and appends metadata CSV rows.

Fully automatic page-corner / grid detection (no manual marking at all)
is a reasonable future improvement (see TASKS.md) once real scans exist
to tune it against. Manual calibration only needs doing once per
scanning setup (e.g. once per scanner, or once per phone-photo rig), not
once per sheet, and is far more reliable than computer vision tuned
blind, without any real photos to test against.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from PIL import Image, ImageDraw

Point = tuple[float, float]


@dataclasses.dataclass
class GridLayout:
    """A regular grid of equal-size, evenly-spaced boxes on a page image,
    described in pixel coordinates of that specific image."""

    box_top_left: Point
    box_width: float
    box_height: float
    column_pitch: float  # horizontal distance between adjacent box top-left corners
    row_pitch: float     # vertical distance between adjacent box top-left corners
    num_columns: int
    num_rows: int

    @classmethod
    def calibrate(
        cls,
        box_top_left: Point,
        box_bottom_right: Point,
        next_column_top_left: Point,
        next_row_top_left: Point,
        num_columns: int,
        num_rows: int,
    ) -> "GridLayout":
        """Derive the full grid from 4 points marked on ONE reference box
        and its two neighbors:
          - box_top_left / box_bottom_right: opposite corners of box #1
            (gives box_width, box_height).
          - next_column_top_left: top-left corner of the box immediately
            to the RIGHT of box #1, same row (gives column_pitch).
          - next_row_top_left: top-left corner of the box immediately
            BELOW box #1, same column (gives row_pitch).
        """
        box_width = box_bottom_right[0] - box_top_left[0]
        box_height = box_bottom_right[1] - box_top_left[1]
        column_pitch = next_column_top_left[0] - box_top_left[0]
        row_pitch = next_row_top_left[1] - box_top_left[1]

        if box_width <= 0 or box_height <= 0:
            raise ValueError("box_bottom_right must be below and to the right of box_top_left.")
        if column_pitch <= 0:
            raise ValueError("next_column_top_left must be to the right of box_top_left.")
        if row_pitch <= 0:
            raise ValueError("next_row_top_left must be below box_top_left.")

        return cls(box_top_left, box_width, box_height, column_pitch, row_pitch, num_columns, num_rows)

    def box_region(self, index: int) -> tuple[float, float, float, float]:
        """0-indexed position in row-major reading order -> pixel box
        (left, top, right, bottom), suitable for PIL's Image.crop()."""
        total_boxes = self.num_columns * self.num_rows
        if not (0 <= index < total_boxes):
            raise ValueError(f"index {index} is out of range for a {self.num_rows}x{self.num_columns} grid ({total_boxes} boxes).")

        row, col = divmod(index, self.num_columns)
        x0 = self.box_top_left[0] + col * self.column_pitch
        y0 = self.box_top_left[1] + row * self.row_pitch
        return (x0, y0, x0 + self.box_width, y0 + self.box_height)


def visualize_calibration(image: Image.Image, layout: GridLayout, num_boxes: int) -> Image.Image:
    """Draw the computed grid over a COPY of `image` in red, so you can
    visually confirm calibration lines up with the real boxes before
    trusting it to crop anything for real."""
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    for index in range(num_boxes):
        draw.rectangle(layout.box_region(index), outline=(220, 30, 30), width=3)
    return preview


def deskew_page(image: Image.Image, corners: tuple[Point, Point, Point, Point], output_size: tuple[int, int]) -> Image.Image:
    """Perspective-correct a photographed page given its 4 corners, in
    (top_left, top_right, bottom_right, bottom_left) order, mapping them
    onto a clean rectangle of `output_size`. Skip this for flatbed scans,
    which are already rectangular.

    Finding these 4 corners automatically (no manual marking) is a
    reasonable future improvement once real photos exist to tune
    detection against — see TASKS.md.
    """
    top_left, top_right, bottom_right, bottom_left = corners
    # PIL's QUAD data order is (upper-left, lower-left, lower-right, upper-right).
    quad = (*top_left, *bottom_left, *bottom_right, *top_right)
    return image.transform(output_size, Image.QUAD, quad, resample=Image.BILINEAR)


def load_word_list(csv_path: str | Path) -> dict[int, str]:
    """Read data/metadata/word_list.csv into a {word_id: khmer_text} map."""
    mapping: dict[int, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[int(row["word_id"])] = row["khmer"]
    return mapping


def crop_word_boxes(
    image: Image.Image,
    layout: GridLayout,
    word_ids_in_order: list[int],
) -> list[tuple[int, Image.Image]]:
    """Crop one box per entry in `word_ids_in_order` (row-major order,
    matching however the words were actually laid out on THIS page —
    never assumed to be all 30 words, since real print pagination may
    split them across pages). Returns (word_id, cropped_image) pairs."""
    total_boxes = layout.num_columns * layout.num_rows
    if len(word_ids_in_order) > total_boxes:
        raise ValueError(
            f"{len(word_ids_in_order)} word ids given but the calibrated grid only has {total_boxes} boxes."
        )
    return [(word_id, image.crop(layout.box_region(index))) for index, word_id in enumerate(word_ids_in_order)]


def process_sheet(
    image_path: str | Path,
    layout: GridLayout,
    word_ids_in_order: list[int],
    word_list: dict[int, str],
    writer_id: str,
    output_images_dir: str | Path,
    metadata_csv_path: str | Path,
) -> list[dict]:
    """Crop one scanned page into per-word images, save them, and append
    rows to the metadata CSV. Returns the rows that were written."""
    image = Image.open(image_path).convert("L")  # grayscale, matching the rest of the pipeline
    crops = crop_word_boxes(image, layout, word_ids_in_order)

    output_images_dir = Path(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for word_id, crop in crops:
        if word_id not in word_list:
            raise ValueError(f"word_id {word_id} is not in the word list — check word_ids_in_order, don't guess a label.")
        filename = f"{writer_id}_{word_id:02d}.png"
        crop.save(output_images_dir / filename)
        rows.append({"image": filename, "label": word_list[word_id], "writer_id": writer_id})

    metadata_csv_path = Path(metadata_csv_path)
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metadata_csv_path.exists() and metadata_csv_path.stat().st_size > 0
    with open(metadata_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return rows

"""
Dataset pipeline verification using SYNTHETIC PLACEHOLDER images.

No real handwriting exists yet, so this test renders our draft word list
with a system Khmer font (printed, not handwritten) purely to exercise
the shape/padding/tokenizer logic end-to-end before real photos exist.
These rendered images are NEVER used as real training data — only as a
stand-in to catch pipeline bugs early (same spirit as Day 1-4's synthetic
tensors and MNIST rehearsal).

Run:
    .venv/bin/pytest tests/test_dataset.py -v
"""

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.tokenizer import KhmerTokenizer

WORDS = [
    "ទឹក", "ភ្លើង", "ខ្យល់", "ថ្ងៃ", "ខែ", "ឆ្នាំ", "ផ្ទះ", "សាលា",
    "គ្រូ", "សៀវភៅ", "ខ្មៅដៃ", "តុ", "កៅអី", "ឆ្កែ", "ឆ្មា", "មាន់",
    "ត្រី", "ផ្កា", "ជ្រូក", "គោ", "ខ្មែរ", "កម្ពុជា", "គ្រួសារ", "មិត្ត",
]
WRITER_IDS = ["W001", "W002", "W003"]  # pretend these came from 3 different writers

# A standard macOS system font with full Khmer coverage. This is a PRINTED
# font, standing in only for pixels — not a claim about handwriting style.
FONT_PATH = "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"


def _render_word_image(word: str, font_size: int, out_path: Path) -> None:
    """Render one word at a given font size onto a white canvas, sized to
    fit — this naturally produces DIFFERENT pixel widths per word/size,
    which is exactly the variability the real handwriting will have."""
    font = ImageFont.truetype(FONT_PATH, font_size)
    dummy = Image.new("L", (10, 10), color=255)
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), word, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 6

    canvas = Image.new("L", (text_w + 2 * padding, text_h + 2 * padding), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding - bbox[0], padding - bbox[1]), word, font=font, fill=0)
    canvas.save(out_path)


@pytest.fixture(scope="module")
def synthetic_dataset_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_khmer_dataset")
    images_dir = root / "images"
    images_dir.mkdir()

    rows = []
    for i, word in enumerate(WORDS):
        font_size = 28 + (i % 5) * 6  # vary size -> vary width, like real handwriting would
        filename = f"{i:04d}.png"
        _render_word_image(word, font_size, images_dir / filename)
        rows.append({"image": filename, "label": word, "writer_id": WRITER_IDS[i % len(WRITER_IDS)]})

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)

    return root


@pytest.fixture(scope="module")
def tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(WORDS)


@pytest.fixture(scope="module")
def dataset(synthetic_dataset_dir, tokenizer) -> KhmerWordDataset:
    samples = load_metadata_csv(synthetic_dataset_dir / "metadata.csv", synthetic_dataset_dir / "images")
    return KhmerWordDataset(samples, tokenizer, image_height=48, max_width=256)


def test_metadata_loads_all_rows(dataset):
    assert len(dataset) == len(WORDS)


def test_every_image_has_fixed_height_and_preserved_aspect_ratio(dataset):
    widths_seen = set()
    for i in range(len(dataset)):
        item = dataset[i]
        c, h, w = item["image"].shape
        assert c == 1, "must be single-channel grayscale"
        assert h == 48, "height must always be exactly the fixed target"
        assert 1 <= w <= 256, "width must be positive and within the clamp"
        widths_seen.add(w)

    # Different words/font sizes must NOT all collapse to one identical
    # width — that would mean images are being stretched instead of
    # resized proportionally.
    assert len(widths_seen) > 1, "expected varying widths (no horizontal stretching to one fixed size)"


def test_pixel_values_are_normalized_to_unit_range(dataset):
    item = dataset[0]
    assert item["image"].min() >= 0.0
    assert item["image"].max() <= 1.0


def test_target_ids_match_tokenizer_roundtrip(dataset, tokenizer):
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["target_length"] == len(item["target_ids"])
        decoded = tokenizer.decode(item["target_ids"].tolist())
        assert decoded == tokenizer.normalize(item["label"])


def test_writer_id_is_never_dropped(dataset):
    seen_writers = {dataset[i]["writer_id"] for i in range(len(dataset))}
    assert seen_writers == set(WRITER_IDS)


def test_collate_fn_pads_to_batch_max_width_not_a_fixed_constant(dataset):
    batch_a = [dataset[i] for i in range(4)]   # shorter words
    batch_b = [dataset[i] for i in range(20, 24)]  # longer words (later in list)

    collated_a = collate_fn(batch_a)
    collated_b = collate_fn(batch_b)

    max_w_a = max(item["image"].shape[-1] for item in batch_a)
    max_w_b = max(item["image"].shape[-1] for item in batch_b)

    # Each batch pads to ITS OWN max width, not a hardcoded constant like 256 —
    # this is the actual proof, not just that shapes happen to look right.
    assert collated_a["images"].shape == (4, 1, 48, max_w_a)
    assert collated_b["images"].shape == (4, 1, 48, max_w_b)
    assert max_w_a != 256 or max_w_b != 256, "widths should reflect real content, not always the hard max clamp"


def test_collate_fn_padding_region_is_white(dataset):
    batch = [dataset[i] for i in range(4)]
    collated = collate_fn(batch)
    for i, item in enumerate(batch):
        true_w = item["image"].shape[-1]
        padded_region = collated["images"][i, :, :, true_w:]
        if padded_region.numel() > 0:
            assert torch.allclose(padded_region, torch.ones_like(padded_region))


def test_collate_fn_targets_are_concatenated_with_correct_lengths(dataset):
    batch = [dataset[i] for i in range(5)]
    collated = collate_fn(batch)
    assert collated["targets"].shape[0] == sum(item["target_length"] for item in batch)
    assert collated["target_lengths"].tolist() == [item["target_length"] for item in batch]


def test_dataloader_end_to_end_with_variable_width_images(dataset):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    total_seen = 0
    for batch in loader:
        b = batch["images"].shape[0]
        assert batch["images"].shape[1:3] == (1, 48)
        assert batch["target_lengths"].shape[0] == b
        assert len(batch["labels"]) == b
        assert len(batch["writer_ids"]) == b
        total_seen += b
    assert total_seen == len(dataset)


def test_empty_label_raises_instead_of_silently_skipping(dataset, tokenizer):
    from src.dataset import Sample

    bad_dataset = KhmerWordDataset(
        samples=[Sample(image_path=dataset.samples[0].image_path, label="", writer_id="W001")],
        tokenizer=tokenizer,
    )
    with pytest.raises(ValueError):
        bad_dataset[0]

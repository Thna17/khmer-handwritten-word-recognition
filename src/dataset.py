"""PyTorch Dataset for Khmer handwritten short-word images.

Pipeline per sample: load image -> grayscale -> resize to a fixed height
while PRESERVING aspect ratio (never stretched to a fixed width) -> clamp
width to a maximum -> tensor in [0, 1]. Batches of different-width images
are combined by `collate_fn`, which pads every image on the right, up to
that batch's own max width, with white (value 1.0) — not a fixed 256
every time, to avoid wasting compute on batches of short words.

Shape reference (H = image_height, W_i = sample i's own resized width,
W_batch = max width in one batch):
    single sample image : [1, H, W_i]
    collated batch       : [B, 1, H, W_batch]
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.tokenizer import KhmerTokenizer

PAD_VALUE = 1.0  # white background, matching ToTensor()'s [0, 1] scaling of a white pixel


@dataclass
class Sample:
    image_path: str
    label: str
    writer_id: str


def load_metadata_csv(csv_path: str | Path, images_dir: str | Path) -> list[Sample]:
    """Read the locked `image,label,writer_id` CSV format into Sample rows.

    `images_dir` is joined with each row's `image` field so the CSV itself
    only needs to store filenames, not full paths.
    """
    images_dir = Path(images_dir)
    samples = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            samples.append(Sample(
                image_path=str(images_dir / row["image"]),
                label=row["label"],
                writer_id=row["writer_id"],
            ))
    return samples


class KhmerWordDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        tokenizer: KhmerTokenizer,
        image_height: int = 48,
        max_width: int = 256,
        transform=None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_height = image_height
        self.max_width = max_width
        self.transform = transform  # optional PIL-image -> PIL-image augmentation hook

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_resize(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("L")  # grayscale, 1 channel

        orig_w, orig_h = image.size
        if orig_h == 0 or orig_w == 0:
            raise ValueError(f"Image {image_path!r} has a zero dimension: {image.size}")

        # Preserve aspect ratio: scale so height becomes exactly image_height.
        scale = self.image_height / orig_h
        new_w = max(1, round(orig_w * scale))
        new_w = min(new_w, self.max_width)  # clamp only pathologically long words

        return image.resize((new_w, self.image_height), Image.BILINEAR)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if not sample.label:
            raise ValueError(f"Empty label for image {sample.image_path!r} — fix the metadata, don't skip it silently.")

        image = self._load_and_resize(sample.image_path)
        if self.transform is not None:
            image = self.transform(image)

        # PIL 'L' image [H, W] in [0, 255] -> tensor [1, H, W] in [0.0, 1.0]
        image_array = np.array(image, dtype=np.uint8)  # [H, W]
        image_tensor = torch.from_numpy(image_array).float().div(255.0).unsqueeze(0)

        target_ids = self.tokenizer.encode(sample.label)

        return {
            "image": image_tensor,                              # [1, H, W_i]
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_length": len(target_ids),
            "label": sample.label,                               # original text, never modified
            "writer_id": sample.writer_id,
            "image_width": image_tensor.shape[-1],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad every image in the batch to this batch's own max width.

    Targets are concatenated into one 1D tensor (not padded per-sample),
    matching the format torch.nn.CTCLoss expects when given target_lengths
    alongside it.
    """
    batch_size = len(batch)
    height = batch[0]["image"].shape[1]
    max_width = max(item["image"].shape[-1] for item in batch)

    images = torch.full((batch_size, 1, height, max_width), fill_value=PAD_VALUE, dtype=torch.float32)
    for i, item in enumerate(batch):
        w = item["image"].shape[-1]
        images[i, :, :, :w] = item["image"]

    targets = torch.cat([item["target_ids"] for item in batch])
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    image_widths = torch.tensor([item["image_width"] for item in batch], dtype=torch.long)

    return {
        "images": images,                    # [B, 1, H, max_width]
        "targets": targets,                  # [sum(target_lengths)]
        "target_lengths": target_lengths,    # [B]
        "image_widths": image_widths,        # [B] — true (pre-padding) widths, for computing input_lengths later
        "labels": [item["label"] for item in batch],
        "writer_ids": [item["writer_id"] for item in batch],
    }

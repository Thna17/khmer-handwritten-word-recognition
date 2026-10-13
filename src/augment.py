"""Handwriting-specific augmentation for the real-data fine-tuning stage.

WHY THIS EXISTS
---------------
Stage 2 fine-tunes on 177 training line images with 8.5M parameters and no
augmentation at all, which drives train_loss to 0.14 while val_loss sits at
1.42 — the model memorizes 7 writers' strokes instead of learning Khmer
script. Every transform here is chosen from a measured property of THIS
dataset rather than copied from a generic vision recipe:

  * `scale_jitter` / `resolution_drop` — native line heights run 28-129px
    (median 30px for W001, 83px for test writer W007). After the fixed
    resize to `image_height`, the training writers are mostly UPscaled and
    the test writers DOWNscaled, so stroke width in pixels differs between
    train and test. Jittering scale and re-sampling resolution makes the
    model scale-invariant instead of tuned to ~30px lines.
  * `stretch` — measured pixels-per-character after resize ranges 11.2
    (W002) to 17.5 (W006). Horizontal stretching covers that spread; CTC is
    alignment-free so it tolerates time-axis rescaling.
  * `pad_crop` — line crops come from automatic segmentation with varying
    tightness, and test-set predictions contain fragments of the NEXT line
    (e.g. a line labelled "ប្រធាន: …" decodes with a trailing "របស់ពពះ…",
    which is the following line's text). Randomizing the margin teaches the
    model that the vertical band it must read is the central one.
  * `stroke_width` — pen type varies per writer (gel vs ballpoint).
    Grayscale min/max filtering is morphological erosion/dilation of the
    ink, which is exactly a pen-width change.
  * `shear` / `baseline_wave` — writer slant and the fact that handwritten
    baselines drift rather than staying ruler-straight.

Deliberately NOT included: vertical flips, 90-degree rotations, large
rotations, and cutout/random-erasing. Those either produce images that
cannot occur in handwriting or can delete a character while the CTC target
still claims it is there, which injects label noise into an already tiny
training set.

Implemented with PIL + NumPy only (no cv2/scipy), so the same code runs in
Colab and in the local test suite.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

WHITE = 255  # paper background in an 'L' mode image

# A transform that leaves less than this share of the original ink has removed
# characters, not just degraded them, and would train the model against a label
# describing text that is no longer in the image.
MIN_INK_RETENTION = 0.35


def _ink_fraction(image: Image.Image) -> float:
    """Share of pixels that are ink, measured against this image's OWN paper
    tone rather than a fixed cutoff — so brightness and gamma changes (which
    preserve every stroke) do not register as ink loss, while morphology and
    blur (which really do erase strokes) still do."""
    array = np.asarray(image, dtype=np.float32)
    paper = float(np.percentile(array, 95))
    darkest = float(np.percentile(array, 1))
    if paper - darkest < 1e-3:
        return 0.0
    return float((array < paper - 0.35 * (paper - darkest)).mean())


class HandwritingAugmentor:
    """Randomized pipeline of handwriting-plausible image distortions.

    Call it with a grayscale PIL image (mode 'L') at its NATIVE resolution —
    before the dataset's resize-to-fixed-height — and it returns a new
    grayscale image. The label is never changed, so every transform must
    preserve the reading order and the full set of characters.

    `strength` scales every magnitude at once: 0.0 is a no-op, 1.0 is the
    tested default, and values above ~1.5 start producing lines a human
    would struggle to read.
    """

    def __init__(self, strength: float = 1.0, rng: random.Random | None = None):
        if strength < 0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        self.strength = strength
        self.rng = rng or random.Random()

    # ---- individual transforms -------------------------------------------------

    def pad_crop(self, image: Image.Image) -> Image.Image:
        """Randomize the segmentation margin: pad with paper, or shave a sliver.

        Vertical range is deliberately wider than horizontal because the
        line-segmentation bleed this corrects is a vertical phenomenon.
        """
        w, h = image.size
        max_v = max(1, int(round(0.12 * h * self.strength)))
        max_hor = max(1, int(round(0.04 * w * self.strength)))

        top, bottom = self.rng.randint(-max_v, max_v), self.rng.randint(-max_v, max_v)
        left, right = self.rng.randint(-max_hor, max_hor), self.rng.randint(-max_hor, max_hor)

        # Never shave so much that the image collapses.
        top = max(top, -(h // 4))
        bottom = max(bottom, -(h // 4))
        left = max(left, -(w // 8))
        right = max(right, -(w // 8))

        new_w, new_h = w + left + right, h + top + bottom
        if new_w < 8 or new_h < 8:
            return image

        canvas = Image.new("L", (new_w, new_h), WHITE)
        canvas.paste(image, (left, top))
        return canvas

    def stretch(self, image: Image.Image) -> Image.Image:
        """Rescale the width only, changing pixels-per-character density."""
        w, h = image.size
        factor = 1.0 + self.rng.uniform(-0.18, 0.18) * self.strength
        new_w = max(8, int(round(w * factor)))
        return image.resize((new_w, h), Image.BILINEAR)

    def shear(self, image: Image.Image) -> Image.Image:
        """Horizontal slant, widening the canvas so nothing is clipped off."""
        w, h = image.size
        amount = self.rng.uniform(-0.18, 0.18) * self.strength
        if abs(amount) < 1e-3:
            return image
        pad = int(abs(amount) * h) + 1
        canvas = Image.new("L", (w + 2 * pad, h), WHITE)
        canvas.paste(image, (pad, 0))
        # x' = x + amount * y, expressed as PIL's inverse affine mapping.
        return canvas.transform(
            canvas.size, Image.AFFINE, (1, amount, -amount * h / 2, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=WHITE,
        )

    def rotate(self, image: Image.Image) -> Image.Image:
        """Small whole-line tilt. Kept tiny — text lines are near-horizontal."""
        angle = self.rng.uniform(-1.8, 1.8) * self.strength
        return image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=WHITE)

    def baseline_wave(self, image: Image.Image) -> Image.Image:
        """Smooth sinusoidal vertical drift, like a hand-written baseline.

        Done as a per-column integer roll rather than a mesh transform: it is
        O(width) in NumPy, exactly invertible in reading order, and cannot
        introduce resampling blur on top of the other geometric ops.
        """
        amplitude = self.rng.uniform(0.02, 0.07) * self.strength
        w, h = image.size
        pixels = max(1, int(round(amplitude * h)))
        if pixels < 1 or w < 4:
            return image

        periods = self.rng.uniform(0.5, 2.0)
        phase = self.rng.uniform(0, 2 * np.pi)
        columns = np.arange(w)
        offsets = np.round(pixels * np.sin(2 * np.pi * periods * columns / w + phase)).astype(int)

        padded = np.full((h + 2 * pixels, w), WHITE, dtype=np.uint8)
        padded[pixels:pixels + h, :] = np.array(image, dtype=np.uint8)
        output = np.empty_like(padded)
        for x in range(w):
            output[:, x] = np.roll(padded[:, x], int(offsets[x]))
        return Image.fromarray(output[pixels:pixels + h, :])

    def stroke_width(self, image: Image.Image) -> Image.Image:
        """Thicken or thin the ink (grayscale morphology = pen-width change).

        Thinning is rationed by the detail budget AND then verified: a 3x3
        max-filter removes a full pixel of ink from each side, and on W001's
        faint 30px lines that was measured to destroy 99% of the ink,
        producing a blank image still paired with its full CTC target. When
        the check fails the image is thickened instead of thinned, so the
        transform still fires rather than silently becoming a no-op.
        """
        thin_probability = 0.5 * self._detail_budget(image.size[1])
        if self.rng.random() < thin_probability:
            thinned = image.filter(ImageFilter.MaxFilter(3))
            if _ink_fraction(thinned) >= 0.35 * _ink_fraction(image):
                return thinned
        return image.filter(ImageFilter.MinFilter(3))      # dark ink grows -> thicker pen

    def _detail_budget(self, height: int) -> float:
        """How much detail destruction this image can absorb, in [0, 1].

        A 30px-tall W001 line has roughly 8px per Khmer vertical zone; a
        93px-tall W007 line has 3x that. Applying the same blur radius and
        the same downscale factor to both would leave the small one
        unreadable (verified by eye on a contact sheet) while barely marking
        the large one. Everything that removes high-frequency detail scales
        by this factor so the AMOUNT OF DETAIL LOST is comparable instead of
        the nominal parameter being equal.
        """
        return float(np.clip((height - 24.0) / 56.0, 0.15, 1.0))

    def resolution_drop(self, image: Image.Image) -> Image.Image:
        """Downscale then restore size, destroying fine detail.

        This is the transform that closes the train/test scale gap: a test
        writer's 83px-tall line downscaled to 48px has genuinely less detail
        than a training writer's 30px line upscaled to 48px, and without this
        the model never sees the low-detail regime during fine-tuning.
        """
        w, h = image.size
        budget = self._detail_budget(h)
        factor = 1.0 - self.rng.uniform(0.10, 0.40) * self.strength * budget
        small_w, small_h = max(8, int(w * factor)), max(8, int(h * factor))
        small = image.resize((small_w, small_h), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def photometric(self, image: Image.Image) -> Image.Image:
        """Paper tone, ink darkness and scanner gamma.

        Brightness is biased DOWNWARD (-0.22 .. +0.10): raising brightness on
        a faint pencil line pushes the ink toward the paper value and can
        erase characters outright, which would leave the CTC target claiming
        text that is no longer in the image.
        """
        image = ImageEnhance.Contrast(image).enhance(
            1.0 + self.rng.uniform(-0.20, 0.40) * self.strength)
        image = ImageEnhance.Brightness(image).enhance(
            1.0 + self.rng.uniform(-0.22, 0.10) * self.strength)

        gamma = 1.0 + self.rng.uniform(-0.25, 0.25) * self.strength
        if abs(gamma - 1.0) > 1e-3:
            table = [min(255, int(round(255.0 * ((i / 255.0) ** gamma)))) for i in range(256)]
            image = image.point(table)
        return image

    def blur(self, image: Image.Image) -> Image.Image:
        budget = self._detail_budget(image.size[1])
        return image.filter(ImageFilter.GaussianBlur(
            radius=self.rng.uniform(0.2, 0.8) * self.strength * budget))

    def noise(self, image: Image.Image) -> Image.Image:
        """Additive sensor noise, as seen in the phone-camera source pages."""
        sigma = self.rng.uniform(3.0, 11.0) * self.strength
        array = np.array(image, dtype=np.float32)
        array += np.random.normal(0.0, sigma, array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    # ---- pipeline --------------------------------------------------------------

    # Geometry first (while the image is still at native resolution), then ink
    # morphology, then resolution loss, then sensor-side effects — the same
    # order these distortions occur in the physical pipeline.
    PIPELINE = (
        (0.60, "pad_crop"),
        (0.50, "stretch"),
        (0.40, "shear"),
        (0.40, "rotate"),
        (0.35, "baseline_wave"),
        (0.40, "stroke_width"),
        (0.35, "resolution_drop"),
        (0.60, "photometric"),
        (0.25, "blur"),
        (0.30, "noise"),
    )

    def _apply_once(self, image: Image.Image) -> Image.Image:
        for probability, name in self.PIPELINE:
            if self.rng.random() < probability:
                image = getattr(self, name)(image)
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.strength == 0:
            return image
        if image.mode != "L":
            image = image.convert("L")

        original_ink = _ink_fraction(image)

        # Individually safe transforms can still stack into an unreadable
        # image (a downscale, then a blur, then a thinning filter). Rather
        # than weaken every transform to make the worst stack safe, draw
        # again when the stack destroyed the ink, and fall back to the
        # undistorted image if even the retry fails.
        for _ in range(3):
            candidate = self._apply_once(image)
            if _ink_fraction(candidate) >= MIN_INK_RETENTION * original_ink:
                return candidate

        return image


def build_augmentor(strength: float, seed: int | None = None) -> HandwritingAugmentor | None:
    """Returns None for strength <= 0 so callers can pass the result straight
    into `KhmerWordDataset(transform=...)` without branching."""
    if strength <= 0:
        return None
    return HandwritingAugmentor(strength=strength, rng=random.Random(seed))

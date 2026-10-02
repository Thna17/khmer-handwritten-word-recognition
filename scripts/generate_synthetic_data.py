#!/usr/bin/env python3
"""
High-speed synthetic Khmer text generator using HarfBuzz + FreeType with font caching.
Properly shapes all subscripts, vowels, and ligatures.
"""

import os
import random
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import uharfbuzz as hb
import freetype

OUTPUT_DIR = "data/synthetic"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

FONT_CANDIDATES = [
    "/Library/Fonts/KhmerOSbattambang.ttf",
    "/Library/Fonts/KhmerOSsiemreap.ttf",
    "/Library/Fonts/KhmerOS.ttf",
    "/Library/Fonts/KhmerOScontent.ttf",
    "/Library/Fonts/KhmerOSfreehand.ttf",
    "/Library/Fonts/KhmerOSfasthand.ttf",
    "/Library/Fonts/AKbalthom KhmerHand.ttf",
    "/Library/Fonts/Khmer Savuth Pen.ttf",
    "/Library/Fonts/NotoSansKhmer-Regular.ttf",
    "/Library/Fonts/NotoSansKhmer-Bold.ttf",
]

AVAILABLE_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]

# Pre-cache font blobs in RAM
FONT_CACHE = {}
for fp in AVAILABLE_FONTS:
    with open(fp, "rb") as f:
        fdata = f.read()
    blob = hb.Blob(fdata)
    hb_face = hb.Face(blob)
    ft_face = freetype.Face(fp)
    FONT_CACHE[fp] = (hb_face, ft_face)

print(f"Pre-cached {len(FONT_CACHE)} fonts in RAM.")

def render_khmer_fast(text: str, font_path: str, font_size: int = 32) -> Image.Image:
    hb_face, ft_face = FONT_CACHE[font_path]
    ft_face.set_char_size(font_size * 64)
    
    hb_font = hb.Font(hb_face)
    hb_font.scale = (font_size * 64, font_size * 64)
    
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)
    
    infos = buf.glyph_infos
    positions = buf.glyph_positions
    
    advances = [pos.x_advance for pos in positions]
    total_w = int(sum(advances) / 64) + 60
    total_h = int(font_size * 2.8)
    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)
    
    cur_x = 15 * 64
    base_y = int(font_size * 1.7)
    
    for info, pos in zip(infos, positions):
        gid = info.codepoint
        try:
            ft_face.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        except Exception:
            continue
        bitmap = ft_face.glyph.bitmap
        
        bx = int((cur_x + pos.x_offset) / 64) + ft_face.glyph.bitmap_left
        by = base_y - ft_face.glyph.bitmap_top - int(pos.y_offset / 64)
        
        bw, bh = bitmap.width, bitmap.rows
        if bw > 0 and bh > 0 and 0 <= bx < total_w - bw and 0 <= by < total_h - bh:
            bmp_arr = np.array(bitmap.buffer, dtype=np.uint8).reshape((bh, bw))
            canvas[by:by+bh, bx:bx+bw] = np.minimum(canvas[by:by+bh, bx:bx+bw], 255 - bmp_arr)
            
        cur_x += pos.x_advance
        
    ink_y, ink_x = np.where(canvas < 220)
    if len(ink_x) == 0 or len(ink_y) == 0:
        return Image.fromarray(canvas)
        
    x_min, x_max = max(0, ink_x.min() - 6), min(total_w, ink_x.max() + 6)
    y_min, y_max = max(0, ink_y.min() - 5), min(total_h, ink_y.max() + 5)
    return Image.fromarray(canvas[y_min:y_max, x_min:x_max])

# 2. Vocabulary
manifest_df = pd.read_csv("data/metadata/labels.csv")
base_words = set()
for text in manifest_df["label"]:
    tokens = re.findall(r'[\u1780-\u17FF\u19E0-\u19FF\u200B]+|[a-zA-Z0-9]+', str(text))
    for t in tokens:
        if 2 <= len(t) <= 22:
            base_words.add(t)

word_list = sorted(list(base_words))
random.seed(42)
np.random.seed(42)

text_samples = []
for _ in range(3000):
    mode = random.random()
    if mode < 0.65:
        text = random.choice(word_list)
    elif mode < 0.88:
        w1 = random.choice(word_list)
        w2 = random.choice(word_list)
        text = f"{w1} {w2}"
    else:
        w = random.choice(word_list)
        punct = random.choice([" ។", "...", " ?", "៖", ""])
        text = f"{w}{punct}"
    text_samples.append(text)

print("Generating 3,000 HarfBuzz-shaped Khmer images...")
records = []
TARGET_HEIGHT = 48

for idx, text in enumerate(text_samples):
    font_path = random.choice(AVAILABLE_FONTS)
    font_size = random.randint(28, 36)
    
    try:
        cropped = render_khmer_fast(text, font_path, font_size)
    except Exception:
        continue
        
    if cropped.height < 8 or cropped.width < 8:
        continue
        
    scale = TARGET_HEIGHT / cropped.height
    new_width = max(16, round(cropped.width * scale))
    resized = cropped.resize((new_width, TARGET_HEIGHT), Image.BILINEAR)
    
    # Subtle augmentations
    if random.random() < 0.30:
        shear = random.uniform(-0.12, 0.12)
        resized = resized.transform(resized.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=255)
    if random.random() < 0.20:
        resized = resized.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.6)))
    if random.random() < 0.20:
        arr = np.array(resized).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2, 4), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        resized = Image.fromarray(arr)

    img_filename = f"synth_{idx:05d}.png"
    img_path = os.path.join(IMAGES_DIR, img_filename)
    resized.save(img_path)
    
    rel_path = os.path.relpath(img_path, ".")
    records.append({
        "image": rel_path,
        "label": text,
        "writer_id": f"SYNTH_{idx % 20:02d}"
    })

df_synth = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, "synthetic_labels.csv")
df_synth.to_csv(csv_path, index=False)
print(f"✓ Successfully rendered and saved {len(df_synth)} correctly shaped synthetic Khmer images!")

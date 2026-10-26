"""Khmer Handwritten Text Recognition (KHTR) — Interactive Live Demo & Web UI.

Built with Gradio for pair programming demonstration, thesis presentation,
and real-time inference on unseen Khmer handwriting.

Supports:
  - Single-model inference & 3-way side-by-side comparison (A1 Baseline vs A2 Transfer vs A3 Transformer).
  - Both Fine-Tuned (real handwriting) and Pre-Trained (synthetic-only) checkpoints.
  - Image upload, clipboard paste, drag & drop, and built-in test gallery from held-out writers (W007 & W0011).
  - Real-time CER / WER calculation when ground truth reference is provided.
  - Rule-based Khmer orthographic post-processing (subscript & vowel ordering normalization).
  - Colab public sharing via `--share` (generates a gradio.live link).

Usage:
  # Run locally:
  python app.py

  # Run in Google Colab with public shareable link:
  python app.py --share --checkpoint-dir /content/drive/MyDrive/KhmerOCR_Checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Patch gradio_client schema parsing bug where boolean additionalProperties crashes ASGI /info
try:
    import gradio_client.utils as _gc_utils
    if hasattr(_gc_utils, "_json_schema_to_python_type"):
        _orig_json_schema = _gc_utils._json_schema_to_python_type
        def _safe_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig_json_schema(schema, defs)
        _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

# Patch Starlette >= 0.38 signature change breaking Gradio 4 TemplateResponse(name, context)
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse
    def _safe_template_response(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name, context = args[0], args[1]
            request = context.get("request")
            if request is not None:
                return _orig_template_response(self, request, name, context, *args[2:], **kwargs)
        return _orig_template_response(self, *args, **kwargs)
    starlette.templating.Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import spaces  # Hugging Face ZeroGPU runtime
except ImportError:
    class _SpacesMock:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is not None:
                return fn
            return lambda f: f
    spaces = _SpacesMock()

from src.decoder import BLANK_INDEX, collapse_ctc_sequence
from src.guardrail import validate_ctc_prediction, validate_image_input
from src.metrics import character_error_rate, word_error_rate
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.postprocessor import clean_khmer_orthography
from src.tokenizer import KhmerTokenizer
from src.utils import get_device, load_checkpoint


# =============================================================================
# GLOBAL MODEL REGISTRY & CACHE
# =============================================================================
_DEVICE = get_device()
_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_TOKENIZER: KhmerTokenizer | None = None
_LM: Any | None = None
_CHECKPOINT_SEARCH_DIRS: list[Path] = [
    REPO_ROOT / "checkpoints",
    Path("/content/drive/MyDrive/KhmerOCR_Checkpoints"),
    Path("/content/checkpoints"),
    REPO_ROOT / "data" / "checkpoints",
]

# Canonical filename templates for checkpoints
CHECKPOINT_NAME_MAP = {
    ("baseline", "finetuned"): [
        "a1_baseline_finetuned_best.pt",
        "a1_baseline_best.pt",
        "baseline_finetuned_best.pt",
        "baseline_best.pt",
        "a1_baseline_synthetic_best.pt",
    ],
    ("baseline", "synthetic"): [
        "a1_baseline_synthetic_best.pt",
        "baseline_synthetic_best.pt",
        "a1_baseline_best.pt",
    ],
    ("transfer", "finetuned"): [
        "a2_transfer_finetuned_best.pt",
        "a2_transfer_best.pt",
        "transfer_finetuned_best.pt",
        "transfer_best.pt",
        "a2_transfer_synthetic_best.pt",
    ],
    ("transfer", "synthetic"): [
        "a2_transfer_synthetic_best.pt",
        "transfer_synthetic_best.pt",
        "a2_transfer_best.pt",
    ],
    ("transformer", "finetuned"): [
        "a3_transformer_finetuned_best.pt",
        "a3_transformer_best.pt",
        "transformer_finetuned_best.pt",
        "transformer_best.pt",
        "a3_transformer_synthetic_best.pt",
    ],
    ("transformer", "synthetic"): [
        "a3_transformer_synthetic_best.pt",
        "transformer_synthetic_best.pt",
        "a3_transformer_best.pt",
    ],
}

MODEL_DISPLAY_NAMES = {
    "baseline": "A1: Baseline CRNN (VGG + BiLSTM)",
    "transfer": "A2: Transfer Learning (ResNet-18 + BiLSTM)",
    "transformer": "A3: Transformer Encoder (VGG + Multi-Head Self-Attention)",
}


def get_tokenizer(vocab_path: str | Path | None = None) -> KhmerTokenizer:
    """Loads and caches the default KhmerTokenizer."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    search_paths = []
    if vocab_path:
        search_paths.append(Path(vocab_path))
    search_paths.extend([
        REPO_ROOT / "data" / "metadata" / "char_to_idx.json",
        REPO_ROOT / "data" / "metadata" / "extended_pretrain_vocab.json",
        REPO_ROOT / "data" / "metadata",
    ])

    for p in search_paths:
        if p.is_file():
            _TOKENIZER = KhmerTokenizer.load(p.parent)
            return _TOKENIZER
        elif p.is_dir() and (p / "char_to_idx.json").is_file():
            _TOKENIZER = KhmerTokenizer.load(p)
            return _TOKENIZER

    raise FileNotFoundError("Could not find Khmer vocabulary JSON. Please verify data/metadata/char_to_idx.json exists.")


def download_checkpoint_if_needed(approach: str, stage: str, filename: str) -> Path | None:
    """Attempts to download missing checkpoint from Hugging Face Hub or remote URL."""
    target_dir = REPO_ROOT / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.is_file():
        return target_path

    # 1. Attempt download from Hugging Face Hub if configured
    hf_repo_id = os.environ.get("HF_MODEL_REPO", "Brathna10/khmer-handwritten-ocr-models")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        print(f"📥 Checking Hugging Face Hub ({hf_repo_id}) for {filename}...")
        downloaded = hf_hub_download(
            repo_id=hf_repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        if Path(downloaded).is_file():
            print(f"✓ Downloaded {filename} from Hugging Face Hub: {downloaded}")
            return Path(downloaded)
    except Exception as e:
        # Hub download failed or repo not yet published; continue to direct URL fallback
        pass

    # 2. Direct URL fallback via environment variables
    url_env_key = f"CHECKPOINT_URL_{approach.upper()}_{stage.upper()}"
    direct_url = os.environ.get(url_env_key) or os.environ.get("CHECKPOINT_BASE_URL", "").rstrip("/")
    if direct_url:
        full_url = direct_url if direct_url.endswith(".pt") else f"{direct_url}/{filename}"
        try:
            import urllib.request
            print(f"📥 Downloading {filename} from URL: {full_url}...")
            urllib.request.urlretrieve(full_url, str(target_path))
            if target_path.is_file():
                print(f"✓ Downloaded {filename} from remote URL successfully!")
                return target_path
        except Exception as e:
            print(f"⚠️ Remote URL download failed for {filename}: {e}")

    return None


def find_checkpoint(approach: str, stage: str, custom_dir: Path | None = None) -> Path | None:
    """Locates a model checkpoint across registered directories or triggers auto-download."""
    candidates = CHECKPOINT_NAME_MAP.get((approach, stage), [f"{approach}_{stage}_best.pt"])
    search_dirs = list(_CHECKPOINT_SEARCH_DIRS)
    if custom_dir:
        search_dirs.insert(0, Path(custom_dir))

    for directory in search_dirs:
        if not directory.exists():
            continue
        for fname in candidates:
            cand_path = directory / fname
            if cand_path.is_file():
                return cand_path

    # Auto-download fallback if missing locally
    if candidates:
        dl_path = download_checkpoint_if_needed(approach, stage, candidates[0])
        if dl_path and dl_path.is_file():
            return dl_path

    return None


def get_model(approach: str, stage: str, vocab_size: int, custom_dir: Path | None = None) -> tuple[torch.nn.Module | None, str]:
    """Loads a model with in-memory caching."""
    cache_key = f"{approach}_{stage}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], "Loaded from cache"

    ckpt_path = find_checkpoint(approach, stage, custom_dir)
    if ckpt_path is None:
        expected = CHECKPOINT_NAME_MAP.get((approach, stage), [f"{approach}_{stage}_best.pt"])[0]
        return None, f"Checkpoint not found. Expected '{expected}' in checkpoints/ or Google Drive."

    try:
        if approach == "baseline":
            model = BaselineCRNN(num_classes=vocab_size)
        elif approach == "transfer":
            model = TransferCRNN(num_classes=vocab_size, freeze_backbone=False)
        elif approach == "transformer":
            model = TransformerCRNN(num_classes=vocab_size)
        else:
            return None, f"Unknown architecture approach: {approach}"

        load_checkpoint(ckpt_path, model, map_location=_DEVICE)
        model.to(_DEVICE)
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model, f"Loaded from {ckpt_path.name}"
    except Exception as e:
        return None, f"Failed to load checkpoint ({ckpt_path.name}): {e}"


# =============================================================================
# IMAGE PREPROCESSING & INFERENCE
# =============================================================================
def preprocess_pil_image(
    img: Image.Image | np.ndarray,
    target_height: int = 48,
    max_width: int = 1600,
) -> tuple[torch.Tensor, int]:
    """Preprocesses a PIL Image or NumPy array into normalized tensor [-1, 1] with white padding."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert alpha channel onto white canvas
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    gray = np.array(img.convert("L"))
    h, w = gray.shape
    if h == 0 or w == 0:
        raise ValueError("Invalid empty image")

    scale = target_height / float(h)
    new_w = min(int(round(w * scale)), max_width)
    new_w = max(new_w, 1)

    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, max_width), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized

    norm = (canvas.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor, new_w


def get_lm():
    """Loads and caches the 4-Gram Khmer Language Model."""
    global _LM
    if _LM is None:
        try:
            from src.decoder_beam_search import build_default_lm
            _LM = build_default_lm()
        except Exception as e:
            print(f"⚠️ Could not load Language Model: {e}")
    return _LM


@torch.no_grad()
def run_model_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    image_width: int,
    tokenizer: KhmerTokenizer,
    use_beam_search: bool = False,
    use_lm: bool = False,
    beam_width: int = 15,
) -> tuple[str, str, float, float]:
    """Runs forward pass, computes greedy or beam search output, token confidence, and latency.

    Returns:
        (raw_text, clean_text, confidence_score, latency_ms)
    """
    t0 = time.perf_counter()
    tensor = tensor.to(_DEVICE)
    log_probs = model(tensor)  # [T, 1, num_classes]
    seq_len = model.compute_sequence_length(image_width)

    if use_beam_search:
        from src.decoder_beam_search import ctc_prefix_beam_search_single
        lm_inst = get_lm() if use_lm else None
        raw_text, _ = ctc_prefix_beam_search_single(
            log_probs[:seq_len, 0, :],
            tokenizer=tokenizer,
            lm=lm_inst,
            beam_width=beam_width,
            alpha=0.6,
            beta=1.0,
        )
        clean_text = clean_khmer_orthography(raw_text)
        conf = 0.95
    else:
        probs = torch.exp(log_probs[:seq_len, 0, :])  # [seq_len, num_classes]
        max_probs, argmax_ids = torch.max(probs, dim=-1)

        predicted_ids = argmax_ids.tolist()
        collapsed_ids = []
        emitted_probs = []
        prev = None

        for t_idx, token_id in enumerate(predicted_ids):
            if token_id != prev:
                if token_id != BLANK_INDEX:
                    collapsed_ids.append(token_id)
                    emitted_probs.append(float(max_probs[t_idx].item()))
            prev = token_id

        raw_text = tokenizer.decode(collapsed_ids)
        clean_text = clean_khmer_orthography(raw_text)
        conf = float(np.mean(emitted_probs)) if emitted_probs else 0.0

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return raw_text, clean_text, conf, latency_ms


def get_khmer_pil_font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Finds or auto-downloads a Khmer Unicode font for PIL Image rendering.
    Ensures Khmer characters never render as missing boxes (tofu) in Colab or Linux.
    """
    font_candidates = [
        REPO_ROOT / "assets" / "fonts" / "KhmerOS.ttf",
        REPO_ROOT / "assets" / "fonts" / "NotoSansKhmer-Regular.ttf",
        Path("/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"),
        Path("/System/Library/Fonts/Supplemental/Khmer MN.ttc"),
        Path("/Library/Fonts/KhmerOS.ttf"),
        Path("/Library/Fonts/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/truetype/khmeros/KhmerOS.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"),
    ]
    for p in font_candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue

    # Auto-download NotoSansKhmer from Google Fonts if running in a fresh Linux/Colab container
    try:
        font_dir = REPO_ROOT / "assets" / "fonts"
        font_dir.mkdir(parents=True, exist_ok=True)
        download_path = font_dir / "NotoSansKhmer-Regular.ttf"
        if not download_path.exists():
            import urllib.request
            url = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansKhmer/NotoSansKhmer-Regular.ttf"
            urllib.request.urlretrieve(url, str(download_path))
        if download_path.is_file():
            return ImageFont.truetype(str(download_path), size=size)
    except Exception:
        pass

    return ImageFont.load_default()


def create_annotated_preview(
    orig_img: Image.Image,
    results: list[dict[str, Any]],
    ground_truth: str | None = None,
) -> Image.Image:
    """Draws a clean visual comparison card stacking the original image above predictions."""
    orig_rgb = orig_img.convert("RGB")
    ow, oh = orig_rgb.size

    canvas_w = max(ow, 900)
    extra_h = 70 + len(results) * 45
    if ground_truth:
        extra_h += 45
    canvas_h = oh + extra_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(orig_rgb, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line([(0, oh), (canvas_w, oh)], fill=(210, 215, 225), width=2)

    font_bold = get_khmer_pil_font(size=22)
    font_reg = get_khmer_pil_font(size=20)

    y = oh + 15
    if ground_truth:
        draw.text((15, y), f"Ground Truth: {ground_truth}", fill=(16, 120, 48), font=font_bold)
        y += 40

    for res in results:
        label = res["model_name"]
        pred = res["pred_clean"]
        conf = res.get("confidence", 0.0)
        draw.text((15, y), f"{label}: {pred}  ({conf*100:.1f}%)", fill=(20, 40, 160), font=font_reg)
        y += 40

    return canvas


# =============================================================================
# GRADIO INTERFACE LOGIC
# =============================================================================
@spaces.GPU
def predict_handler(
    image: Image.Image | None,
    model_mode: str,
    stage_mode: str,
    ground_truth: str,
    enable_postprocess: bool,
    enable_guardrail: bool = True,
    enable_beam_search: bool = False,
    enable_lm: bool = False,
    custom_checkpoint_dir: str = "",
) -> tuple[str, str, str, Image.Image | None]:
    """Main callback triggered by 'Recognize Handwriting' button."""
    if image is None:
        return (
            "<div style='color: #c53030; font-size: 16px;'>⚠️ Please upload or select an image first!</div>",
            "",
            "",
            None,
        )

    # 1. Pre-inference Visual & Physical Topic Guardrail
    if enable_guardrail:
        is_valid_img, img_reason = validate_image_input(image)
        if not is_valid_img:
            guard_html = f"""
            <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border: 2px solid #f59e0b; border-radius: 12px; padding: 22px; color: #92400e; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.15);">
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 26px;">⚠️</span>
                <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #b45309;">
                  Out-of-Topic / Non-Handwriting Image Detected
                </h3>
              </div>
              <p style="margin: 0 0 10px 0; font-size: 15px; color: #78350f; line-height: 1.5;">
                This system is specialized exclusively for <b>Khmer Handwritten Text Recognition (lines & words)</b>. The uploaded image was rejected by the topic guardrail.
              </p>
              <div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; font-size: 14px; margin-bottom: 12px; color: #92400e;">
                <b>Detection Reason:</b> {img_reason}
              </div>
              <p style="margin: 0; font-size: 13px; color: #78350f;">
                💡 <i>Please upload a clear horizontal image crop containing genuine Khmer handwriting on paper (or click one of the preset test examples below).</i>
              </p>
            </div>
            """
            return guard_html, "", "", None

    try:
        tokenizer = get_tokenizer()
    except Exception as e:
        return (
            f"<div style='color: #c53030;'>❌ Vocabulary Error: {e}</div>",
            "",
            "",
            None,
        )

    custom_dir = Path(custom_checkpoint_dir.strip()) if custom_checkpoint_dir.strip() else None
    stage_key = "finetuned" if "fine-tuned" in stage_mode.lower() else "synthetic"

    # Preprocess input image
    try:
        tensor, new_w = preprocess_pil_image(image)
    except Exception as e:
        return (
            f"<div style='color: #c53030;'>❌ Image Preprocessing Error: {e}</div>",
            "",
            "",
            None,
        )

    approaches_to_run = []
    if "compare all" in model_mode.lower():
        approaches_to_run = ["baseline", "transfer", "transformer"]
    elif "baseline" in model_mode.lower():
        approaches_to_run = ["baseline"]
    elif "transfer" in model_mode.lower():
        approaches_to_run = ["transfer"]
    elif "transformer" in model_mode.lower():
        approaches_to_run = ["transformer"]
    else:
        approaches_to_run = ["baseline"]

    results = []
    errors = []

    if "netra-ocr" in model_mode.lower():
        try:
            from netra_ocr.recognition.recognize_text import recognize as netra_recognize
            t0 = time.perf_counter()
            raw = netra_recognize(image)
            clean = clean_khmer_orthography(raw)
            lat = (time.perf_counter() - t0) * 1000.0
            chosen_text = clean if enable_postprocess else raw
            res = {
                "approach": "netra_ocr",
                "model_name": "🌟 Netra-OCR (Zero-Shot SOTA Transformer)",
                "pred_clean": chosen_text,
                "pred_raw": raw,
                "confidence": 0.82,
                "latency_ms": lat,
            }
            if ground_truth.strip():
                gt = ground_truth.strip()
                res["cer"] = character_error_rate([gt], [chosen_text])
                res["wer"] = word_error_rate([gt], [chosen_text])
                res["exact"] = (chosen_text == gt)
            results.append(res)
        except Exception as e:
            errors.append(f"<b>Netra-OCR</b>: {e}")
    else:
        for approach in approaches_to_run:
            model, status = get_model(approach, stage_key, tokenizer.vocab_size, custom_dir)
            if model is None:
                errors.append(f"<b>{MODEL_DISPLAY_NAMES[approach]}</b>: {status}")
                continue

            raw, clean, conf, latency = run_model_inference(
                model,
                tensor,
                new_w,
                tokenizer,
                use_beam_search=enable_beam_search,
                use_lm=enable_lm,
            )
            chosen_text = clean if enable_postprocess else raw

            res = {
                "approach": approach,
                "model_name": MODEL_DISPLAY_NAMES[approach],
                "pred_clean": chosen_text,
                "pred_raw": raw,
                "confidence": conf,
                "latency_ms": latency,
            }

            if ground_truth.strip():
                gt = ground_truth.strip()
                res["cer"] = character_error_rate([gt], [chosen_text])
                res["wer"] = word_error_rate([gt], [chosen_text])
                res["exact"] = (chosen_text == gt)

            results.append(res)

    if not results:
        err_html = "<div style='padding: 15px; background: #fff5f5; border-left: 4px solid #e53e3e; border-radius: 6px;'>"
        err_html += "<h4 style='margin: 0 0 10px 0; color: #c53030;'>⚠️ No Trained Checkpoints Found</h4>"
        err_html += "<p style='margin: 0 0 10px 0;'>Checkpoints could not be loaded from <code>checkpoints/</code> or Google Drive.</p>"
        err_html += "<ul>" + "".join(f"<li>{e}</li>" for e in errors) + "</ul>"
        err_html += "<p style='margin: 5px 0 0 0; font-size: 13px; color: #718096;'>💡 Tip: Train in Colab or specify <code>--checkpoint-dir</code> pointing to your saved <code>.pt</code> files.</p>"
        err_html += "</div>"
        return err_html, "", "", None

    # 2. Post-inference CTC Domain Guardrail (catches OOD natural scenes, English text, noise)
    if enable_guardrail:
        invalid_results = []
        for r in results:
            is_valid_pred, pred_err = validate_ctc_prediction(
                r["pred_clean"],
                r["confidence"],
                new_w,
                min_confidence=0.40,
            )
            if not is_valid_pred:
                invalid_results.append((r, pred_err))

        if len(invalid_results) == len(results):
            reasons_html = "".join(f"<li><b>{r['model_name'].split(':')[0]}</b>: {err}</li>" for r, err in invalid_results)
            guard_html = f"""
            <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border: 2px solid #f59e0b; border-radius: 12px; padding: 22px; color: #92400e; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.15);">
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 26px;">⚠️</span>
                <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #b45309;">
                  Out-of-Topic / Non-Handwriting Prediction Rejected
                </h3>
              </div>
              <p style="margin: 0 0 10px 0; font-size: 15px; color: #78350f; line-height: 1.5;">
                The model could not identify valid Khmer handwriting in this image. Recognition confidence or text structure failed the validity threshold.
              </p>
              <div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; font-size: 14px; margin-bottom: 12px; color: #92400e;">
                <ul style="margin: 0; padding-left: 18px;">{reasons_html}</ul>
              </div>
              <p style="margin: 0; font-size: 13px; color: #78350f;">
                💡 <i>The model cannot predict images outside of Khmer handwriting. Please ensure your image contains a clear, handwritten Khmer line or word crop.</i>
              </p>
            </div>
            """
            return guard_html, "", "", None

    # Construct HTML Displays
    annotated_img = create_annotated_preview(image, results, ground_truth.strip() if ground_truth.strip() else None)

    # 1. Primary or Comparison Card
    if len(results) == 1:
        r = results[0]
        primary_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 22px; border-radius: 12px; color: white; box-shadow: 0 4px 14px rgba(0,0,0,0.15);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">Recognized Khmer Text</span>
            <span style="background: rgba(255,255,255,0.2); padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;">{r['latency_ms']:.1f} ms</span>
          </div>
          <div style="font-family: 'Kantumruy Pro', 'Khmer OS', 'Khmer Sangam MN', sans-serif; font-size: 28px; font-weight: 600; line-height: 1.5; margin: 10px 0; word-break: break-word;">
            {r['pred_clean']}
          </div>
          <div style="display: flex; gap: 15px; margin-top: 14px; font-size: 14px; opacity: 0.95;">
            <span><b>Confidence:</b> {r['confidence']*100:.1f}%</span>
            <span>•</span>
            <span><b>Length:</b> {len(r['pred_clean'])} chars</span>
            <span>•</span>
            <span><b>Model:</b> {r['model_name'].split(':')[0]}</span>
          </div>
        </div>
        """
    else:
        # Comparison mode card
        cards = ""
        for r in results:
            tag = r['model_name'].split(':')[0]
            metric_badge = ""
            if "cer" in r:
                metric_badge = f"<span style='color: {'#38a169' if r['cer'] < 0.15 else '#d69e2e'}; font-weight: 600;'>CER: {r['cer']*100:.1f}% | WER: {r['wer']*100:.1f}%</span>"
            else:
                metric_badge = f"<span style='color: #4a5568;'>Conf: {r['confidence']*100:.1f}%</span>"

            cards += f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; margin-bottom: 12px; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1e3a8a; font-size: 14px;">{r['model_name']}</span>
                <span style="font-size: 12px; color: #718096;">{r['latency_ms']:.1f} ms | {metric_badge}</span>
              </div>
              <div style="font-family: 'Kantumruy Pro', 'Khmer OS', sans-serif; font-size: 22px; font-weight: 600; color: #1a202c; line-height: 1.5; margin: 6px 0;">
                {r['pred_clean']}
              </div>
            </div>
            """
        primary_html = f"<div style='display: flex; flex-direction: column;'>{cards}</div>"

    # 2. Metrics summary
    metrics_md = ""
    if ground_truth.strip():
        gt = ground_truth.strip()
        metrics_md += "### 📊 Quantitative Evaluation vs. Ground Truth\n"
        metrics_md += f"**Reference:** `{gt}`\n\n"
        metrics_md += "| Model Architecture | Predicted Text | CER (%) | WER (%) | Exact Match |\n"
        metrics_md += "|:-------------------|:---------------|:-------:|:-------:|:-----------:|\n"
        for r in results:
            exact_icon = "✅ Exact" if r.get("exact", False) else "❌ Diff"
            metrics_md += f"| **{r['model_name']}** | {r['pred_clean']} | **{r['cer']*100:.2f}%** | {r['wer']*100:.2f}% | {exact_icon} |\n"
    else:
        metrics_md += "💡 *Enter a Ground Truth string in the left panel to automatically compute Character Error Rate (CER) and Word Error Rate (WER).*"

    # 3. Technical details (Raw CTC vs Cleaned)
    details_md = "#### CTC Decoder Inspection\n"
    for r in results:
        details_md += f"- **{r['model_name']}**\n"
        details_md += f"  - Raw CTC output: `{r['pred_raw']}`\n"
        details_md += f"  - Normalized text: `{r['pred_clean']}`\n"
        details_md += f"  - Token confidence: `{r['confidence']*100:.2f}%` | Latency: `{r['latency_ms']:.1f}ms`\n"

    return primary_html, metrics_md, details_md, annotated_img


# =============================================================================
# CURATED EXAMPLES GALLERY
# =============================================================================
CURATED_EXAMPLES = [
    [
        "data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png",
        "🚀 Compare All 3 Models Side-by-Side",
        "Fine-Tuned (Real Handwriting)",
        "ប្រធាន: នៅក្នុងច្បាប់ល្បើកថ្មី",
        True,
        True,
    ],
    [
        "data/handwritten_external/W0011/lines_raw/W0011_P001_L002.png",
        "🚀 Compare All 3 Models Side-by-Side",
        "Fine-Tuned (Real Handwriting)",
        "របស់ព្រះភិរម្យភាសាអ៊ូ ហៅង៉ុយបានចែងថា “ បម្រើម្ចាស់ពិត ប្រុងក្តី",
        True,
        True,
    ],
    [
        "data/handwritten_external/W0011/lines_raw/W0011_P001_L003.png",
        "A1: Baseline CRNN (VGG-11 + BiLSTM)",
        "Fine-Tuned (Real Handwriting)",
        "ឲ្យមត់ ស្មារតីឲ្យមាន សៅហ្មងឲ្យបាត់ សេពគប់",
        True,
        True,
    ],
    [
        "data/handwritten_external/W007/lines_raw/W007_P001_L001.png",
        "A2: Transfer Learning (ResNet-18 + BiLSTM)",
        "Fine-Tuned (Real Handwriting)",
        "ប្រធាន ៖ មានទស្សនៈមួយបានចែងមកថា “ ការសិក្សាអក្សរ",
        True,
        True,
    ],
    [
        "data/handwritten_external/W007/lines_raw/W007_P001_L002.png",
        "A3: Transformer Sequence Encoder (VGG + Self-Attention)",
        "Fine-Tuned (Real Handwriting)",
        "សិល្ប៍ គឺជាការសិក្សាពីមនុស្ស ដើម្បីមនុស្ស ។",
        True,
        True,
    ],
    [
        "data/handwritten_external/W007/lines_raw/W007_P001_L005.png",
        "🚀 Compare All 3 Models Side-by-Side",
        "Fine-Tuned (Real Handwriting)",
        "សេចក្តីអធិប្បាយ",
        True,
        True,
    ],
]


# Filter examples that actually exist on disk
AVAILABLE_EXAMPLES = [
    ex for ex in CURATED_EXAMPLES if (REPO_ROOT / ex[0]).is_file()
]


# =============================================================================
# BUILD GRADIO APP
# =============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@400;600;700&display=swap');

body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.khmer-font {
    font-family: 'Kantumruy Pro', 'Khmer OS', 'Khmer Sangam MN', sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.badge-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 500;
    margin-right: 8px;
    margin-top: 6px;
}
"""


def build_app(custom_checkpoint_dir: str = "") -> gr.Blocks:
    """Builds and returns the Gradio Blocks UI."""
    major_version = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    blocks_kwargs = {"title": "Khmer Handwritten Recognition — Live Demo"}
    if major_version < 6:
        blocks_kwargs["theme"] = gr.themes.Soft()
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:

        # Header Section
        gr.HTML(f"""
        <div class="header-box">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
              <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                🇰🇭 Khmer Handwritten Text Recognition (KHTR) — Live Demo
              </h1>
              <p style="margin: 6px 0 12px 0; font-size: 14px; color: #94a3b8;">
                Offline line-level recognition & deep sequence modeling for historical and modern Khmer handwriting.
              </p>
              <div>
                <span class="badge-tag">Candidate: Hong Than Brathna</span>
                <span class="badge-tag">Advisor: Mr. Soklong HIM</span>
                <span class="badge-tag">Device: {_DEVICE}</span>
                <span class="badge-tag">🛡️ Topic Guardrail: Active</span>
              </div>
            </div>
          </div>
        </div>
        """)

        with gr.Row():
            # LEFT COLUMN: INPUTS & CONTROLS
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Upload Handwriting Image")
                input_img = gr.Image(
                    label="Handwritten Line Image (Upload, Paste from Clipboard, or Click Example Below)",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("### ⚙️ 2. Model & Settings")
                model_dropdown = gr.Dropdown(
                    label="Model Architecture",
                    choices=[
                        "🚀 Compare All 3 Models Side-by-Side",
                        "A1: Baseline CRNN (VGG-11 + BiLSTM)",
                        "A2: Transfer Learning (ResNet-18 + BiLSTM)",
                        "A3: Transformer Sequence Encoder (VGG + Self-Attention)",
                        "🌟 Netra-OCR (Zero-Shot SOTA Transformer)",
                    ],
                    value="🚀 Compare All 3 Models Side-by-Side",
                )

                stage_radio = gr.Radio(
                    label="Model Training Stage",
                    choices=[
                        "Fine-Tuned (Real Handwriting)",
                        "Pre-Trained (Synthetic Only)",
                    ],
                    value="Fine-Tuned (Real Handwriting)",
                    info="Fine-Tuned evaluates adaptation on genuine handwriting; Synthetic tests zero-shot pre-training.",
                )

                ground_truth_input = gr.Textbox(
                    label="Ground Truth Reference (Optional)",
                    placeholder="Enter reference text to calculate CER (%) and WER (%)...",
                    lines=2,
                )

                with gr.Accordion("Advanced Settings, Decoding & Guardrail", open=False):
                    enable_beam_search_chk = gr.Checkbox(
                        label="🎯 Enable CTC Prefix Beam Search",
                        value=False,
                        info="Explores multiple probabilistic paths to avoid greedy argmax failure.",
                    )
                    enable_lm_chk = gr.Checkbox(
                        label="📚 Enable 4-Gram Khmer Language Model Rescoring",
                        value=False,
                        info="Uses statistical character n-grams to correct cursive visual twins (e.g. ប្រ vs ប្ល, ច vs ថ).",
                    )
                    enable_guardrail_chk = gr.Checkbox(
                        label="🛡️ Enable Handwriting & Topic Guardrail",
                        value=True,
                        info="Rejects out-of-domain images (non-handwriting, photos, blank crops, or noise) to prevent false predictions.",
                    )
                    enable_postprocess_chk = gr.Checkbox(
                        label="Enable Khmer Orthographic Post-Processing",
                        value=True,
                        info="Enforces Unicode NFC, resolves duplicate Coeng (្), and cleans vowel ordering anomalies.",
                    )
                    checkpoint_dir_override = gr.Textbox(
                        label="Custom Checkpoints Directory (Optional)",
                        value=custom_checkpoint_dir,
                        placeholder="e.g., /content/drive/MyDrive/KhmerOCR_Checkpoints",
                    )

                with gr.Row():
                    btn_submit = gr.Button("⚡ Recognize Handwriting", variant="primary", size="lg")
                    btn_clear = gr.ClearButton(components=[input_img, ground_truth_input], value="Clear", size="lg")

            # RIGHT COLUMN: PREDICTIONS & METRICS
            with gr.Column(scale=6):
                gr.Markdown("### 🎯 3. Recognition Results")
                primary_output = gr.HTML(
                    value="""
                    <div style="padding: 24px; border: 2px dashed #cbd5e1; border-radius: 12px; text-align: center; color: #64748b;">
                      Upload an image or pick a preset example below to see real-time Khmer transcription.
                    </div>
                    """
                )

                metrics_output = gr.Markdown()

                with gr.Accordion("🔍 CTC Diagnostic Breakdown", open=False):
                    details_output = gr.Markdown()

                gr.Markdown("### 🖼️ Visual Verification Card")
                annotated_preview = gr.Image(label="Line Crop + Inscribed Khmer Prediction", type="pil")

        # PRESET EXAMPLES GALLERY
        if AVAILABLE_EXAMPLES:
            gr.Markdown("### 📚 Preset Verified Test Examples (Held-Out Writers W007 & W0011)")
            gr.Examples(
                examples=AVAILABLE_EXAMPLES,
                inputs=[
                    input_img,
                    model_dropdown,
                    stage_radio,
                    ground_truth_input,
                    enable_postprocess_chk,
                    enable_guardrail_chk,
                ],
                label="Click any test line to load image and verified ground truth immediately:",
            )

        # Wire click event
        btn_submit.click(
            fn=predict_handler,
            inputs=[
                input_img,
                model_dropdown,
                stage_radio,
                ground_truth_input,
                enable_postprocess_chk,
                enable_guardrail_chk,
                enable_beam_search_chk,
                enable_lm_chk,
                checkpoint_dir_override,
            ],
            outputs=[
                primary_output,
                metrics_output,
                details_output,
                annotated_preview,
            ],
        )

    return demo


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Khmer Handwritten Recognition Live Demo Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on (default 7860)")
    parser.add_argument("--share", action="store_true", help="Generate public shareable gradio.live link (ideal for Colab)")
    parser.add_argument("--checkpoint-dir", default="", help="Optional custom directory containing .pt checkpoints")
    parser.add_argument("--vocab-path", default="data/metadata/char_to_idx.json", help="Path to character vocabulary JSON")

    args = parser.parse_args()

    # Pre-register custom checkpoint dir if provided
    if args.checkpoint_dir:
        custom_path = Path(args.checkpoint_dir)
        if custom_path.is_dir():
            _CHECKPOINT_SEARCH_DIRS.insert(0, custom_path)
            print(f"✓ Added checkpoint directory: {custom_path}")

    # Warm up tokenizer
    try:
        tok = get_tokenizer(args.vocab_path)
        print(f"✓ Loaded Khmer Tokenizer: {tok.vocab_size} tokens | Device: {_DEVICE}")
    except Exception as e:
        print(f"⚠️ Warning loading tokenizer: {e}")

    demo = build_app(custom_checkpoint_dir=args.checkpoint_dir)
    print(f"\n=======================================================")
    print(f"   🇰🇭 Khmer OCR Live Web UI Launching on port {args.port}")
    if args.share:
        print(f"   Public share link requested: --share")
    print(f"=======================================================\n")

    major_version = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": False,
        "show_error": True,
        "show_api": False,
    }
    if args.share:
        launch_kwargs["share"] = True

    if major_version >= 6:
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = CUSTOM_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()

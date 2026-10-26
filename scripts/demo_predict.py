"""Live Demo Inference Script for Khmer Handwritten / Synthetic Text Recognition.

Takes an image (or directory of images), runs it through a trained OCR model checkpoint,
and prints / displays the predicted Khmer text side-by-side with ground truth (if provided).

Usage examples:
    # Single image inference with best fine-tuned model:
    python scripts/demo_predict.py \\
        --image data/line_dataset/images/w11_p2_l14.png \\
        --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\
        --vocab-path data/metadata/char_to_idx.json

    # Compare synthetic model vs. fine-tuned model on the same image:
    python scripts/demo_predict.py \\
        --image data/line_dataset/images/w11_p2_l14.png \\
        --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\
        --compare-checkpoint checkpoints/a1_baseline_synthetic_best.pt \\
        --vocab-path data/metadata/char_to_idx.json

    # Save a visual comparison image with rendered Khmer predictions:
    python scripts/demo_predict.py \\
        --image data/line_dataset/images/w11_p2_l14.png \\
        --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\
        --vocab-path data/metadata/char_to_idx.json \\
        --output-viz demo_result.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works reliably
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.decoder import greedy_decode
from src.decoder_beam_search import build_default_lm, ctc_prefix_beam_search_single
from src.metrics import character_error_rate, word_error_rate
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.postprocessor import clean_khmer_orthography
from src.tokenizer import KhmerTokenizer
from src.utils import get_device, load_checkpoint


def preprocess_image(image_path: str | Path, target_height: int = 48, max_width: int = 1024) -> tuple[torch.Tensor, int]:
    """Loads an image in grayscale, normalizes to [-1, 1], resizes height to target_height,
    maintains aspect ratio up to max_width, and pads with white background (value 1.0).
    Returns (tensor [1, 1, H, W], original_width).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    h, w = img.shape
    scale = target_height / float(h)
    new_w = min(int(round(w * scale)), max_width)
    new_w = max(new_w, 1)

    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

    # Canvas padded with 255 (white)
    canvas = np.full((target_height, max_width), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized

    # Normalize to [-1.0, 1.0] (matching KhmerWordDataset)
    norm = (canvas.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor, new_w


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    vocab_size: int,
    approach: str = "baseline",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiates the correct architecture and loads the saved weights."""
    if device is None:
        device = get_device()

    if approach == "baseline":
        model = BaselineCRNN(num_classes=vocab_size)
    elif approach == "transfer":
        model = TransferCRNN(num_classes=vocab_size, freeze_backbone=False)
    elif approach == "transformer":
        model = TransformerCRNN(num_classes=vocab_size)
    else:
        raise ValueError(f"Unknown approach: {approach}")

    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    image_width: int,
    tokenizer: KhmerTokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    lm: Any | None = None,
    beam_width: int = 20,
    alpha: float = 0.6,
    beta: float = 1.0,
) -> str:
    """Runs forward pass with either greedy or CTC prefix beam search decoding."""
    tensor = tensor.to(device)
    log_probs = model(tensor)  # [T, 1, num_classes]
    seq_len = model.compute_sequence_length(image_width)

    if use_beam_search:
        text, _ = ctc_prefix_beam_search_single(
            log_probs[:seq_len, 0, :],
            tokenizer=tokenizer,
            lm=lm,
            beam_width=beam_width,
            alpha=alpha,
            beta=beta,
        )
        return text
    else:
        input_lengths = torch.tensor([seq_len], dtype=torch.long)
        preds = greedy_decode(log_probs, tokenizer, input_lengths=input_lengths)
        return preds[0]


def render_visual_output(
    image_path: str | Path,
    pred_text: str,
    target_text: str | None = None,
    output_path: str | Path = "demo_result.png",
    second_pred_text: str | None = None,
    second_label: str = "Model 2",
) -> None:
    """Generates an image showing the input line image above the predicted Khmer text."""
    orig_img = Image.open(image_path).convert("RGB")
    w, h = orig_img.size

    # Target canvas width
    canvas_w = max(w, 800)
    extra_h = 160 if second_pred_text else 110
    canvas_h = h + extra_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(orig_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    # Border under original image
    draw.line([(0, h), (canvas_w, h)], fill=(200, 200, 200), width=2)

    # Try loading a Khmer font, fallback to default
    khmer_font_candidates = [
        REPO_ROOT / "assets" / "fonts" / "KhmerOS.ttf",
        REPO_ROOT / "assets" / "fonts" / "NotoSansKhmer-Regular.ttf",
        Path("/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"),
        Path("/System/Library/Fonts/Supplemental/Khmer MN.ttc"),
        Path("/Library/Fonts/KhmerOS.ttf"),
        Path("/Library/Fonts/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/truetype/khmeros/KhmerOS.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"),
    ]
    font = None
    for fpath in khmer_font_candidates:
        p = Path(fpath)
        if p.exists():
            try:
                font = ImageFont.truetype(str(p), size=24)
                break
            except Exception:
                continue

    if font is None:
        try:
            font_dir = REPO_ROOT / "assets" / "fonts"
            font_dir.mkdir(parents=True, exist_ok=True)
            download_path = font_dir / "NotoSansKhmer-Regular.ttf"
            if not download_path.exists():
                import urllib.request
                url = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansKhmer/NotoSansKhmer-Regular.ttf"
                urllib.request.urlretrieve(url, str(download_path))
            if download_path.is_file():
                font = ImageFont.truetype(str(download_path), size=24)
        except Exception:
            pass

    if font is None:
        font = ImageFont.load_default()

    y_offset = h + 15
    if target_text:
        draw.text((15, y_offset), f"Ground Truth: {target_text}", fill=(0, 100, 0), font=font)
        y_offset += 35

    draw.text((15, y_offset), f"Prediction : {pred_text}", fill=(0, 0, 180), font=font)

    if second_pred_text:
        y_offset += 35
        draw.text((15, y_offset), f"{second_label}: {second_pred_text}", fill=(180, 0, 0), font=font)

    canvas.save(output_path)
    print(f"✓ Visual comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Live Khmer OCR Inference Demo")
    parser.add_argument("--image", required=True, help="Path to input image (png/jpg)")
    parser.add_argument("--checkpoint", required=True, help="Path to primary model checkpoint (.pt)")
    parser.add_argument("--vocab-path", required=True, help="Path to char_to_idx.json")
    parser.add_argument("--approach", default="baseline", choices=["baseline", "transfer", "transformer"])
    parser.add_argument("--ground-truth", default=None, help="Optional ground truth text for evaluation")
    parser.add_argument("--compare-checkpoint", default=None, help="Optional second checkpoint for comparison")
    parser.add_argument("--compare-approach", default="baseline", choices=["baseline", "transfer", "transformer"])
    parser.add_argument("--output-viz", default=None, help="Optional path to save annotated visual PNG")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable Khmer orthographic rule post-processing")
    parser.add_argument("--use-beam-search", action="store_true", help="Enable CTC Prefix Beam Search instead of greedy decoding")
    parser.add_argument("--use-lm", action="store_true", help="Enable 4-gram Language Model scoring during beam search")
    parser.add_argument("--beam-width", type=int, default=20, help="Beam width for beam search (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.6, help="LM score weight alpha (default: 0.6)")
    parser.add_argument("--beta", type=float, default=1.0, help="Length bonus beta (default: 1.0)")

    args = parser.parse_args()

    vocab_file = Path(args.vocab_path)
    if not vocab_file.exists():
        print(f"Error: vocab file not found at {args.vocab_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = KhmerTokenizer.load(vocab_file.parent)
    device = get_device()
    print(f"Device: {device} | Vocab size: {tokenizer.vocab_size}")

    lm = None
    if args.use_beam_search and args.use_lm:
        lm = build_default_lm()
        print(f"✓ Loaded Khmer 4-Gram LM ({lm.total_chars:,} training characters)")

    tensor, new_w = preprocess_image(args.image)

    # Primary model inference
    dec_mode = f"Beam Search (width={args.beam_width}, LM={'on' if lm else 'off'})" if args.use_beam_search else "Greedy CTC"
    print(f"\nLoading primary model: {args.checkpoint} ({args.approach}) [{dec_mode}] ...")
    model = load_model_from_checkpoint(args.checkpoint, tokenizer.vocab_size, args.approach, device)
    pred_primary = predict_image(
        model,
        tensor,
        new_w,
        tokenizer,
        device,
        use_beam_search=args.use_beam_search,
        lm=lm,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
    )
    if not args.no_postprocess:
        pred_primary = clean_khmer_orthography(pred_primary)

    # Comparison model inference (if requested)
    pred_secondary = None
    if args.compare_checkpoint:
        print(f"Loading comparison model: {args.compare_checkpoint} ({args.compare_approach}) ...")
        model2 = load_model_from_checkpoint(args.compare_checkpoint, tokenizer.vocab_size, args.compare_approach, device)
        pred_secondary = predict_image(
            model2,
            tensor,
            new_w,
            tokenizer,
            device,
            use_beam_search=args.use_beam_search,
            lm=lm,
            beam_width=args.beam_width,
            alpha=args.alpha,
            beta=args.beta,
        )
        if not args.no_postprocess:
            pred_secondary = clean_khmer_orthography(pred_secondary)

    # Display results
    print("\n" + "=" * 65)
    print("                 KHMER OCR INFERENCE RESULTS")
    print("=" * 65)
    print(f"Input Image : {args.image}")

    if args.ground_truth:
        print(f"Ground Truth: {args.ground_truth}")
        cer = character_error_rate([args.ground_truth], [pred_primary])
        wer = word_error_rate([args.ground_truth], [pred_primary])
        print("-" * 65)
        print(f"Primary Pred: {pred_primary}")
        print(f"Metrics     : CER={cer*100:.2f}% | WER={wer*100:.2f}%")
    else:
        print(f"Prediction  : {pred_primary}")

    if pred_secondary:
        print("-" * 65)
        if args.ground_truth:
            cer2 = character_error_rate([args.ground_truth], [pred_secondary])
            wer2 = word_error_rate([args.ground_truth], [pred_secondary])
            print(f"Compare Pred: {pred_secondary}")
            print(f"Metrics     : CER={cer2*100:.2f}% | WER={wer2*100:.2f}%")
        else:
            print(f"Compare Pred: {pred_secondary}")
    print("=" * 65 + "\n")

    if args.output_viz:
        render_visual_output(
            image_path=args.image,
            pred_text=pred_primary,
            target_text=args.ground_truth,
            output_path=args.output_viz,
            second_pred_text=pred_secondary,
            second_label="Comparison Model",
        )


if __name__ == "__main__":
    main()

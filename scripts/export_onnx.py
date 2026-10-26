#!/usr/bin/env python3
"""Export PyTorch KHTR Models to Standard ONNX with Dynamic Sequence Length.

Features:
1. Supports A1 (Baseline CRNN), A2 (Transfer CRNN), and A3 (Transformer CRNN).
2. Dynamic width dimension [B, 1, 48, W] -> [T, B, num_classes].
3. Performs ONNX graph shape inference and structural validation (`onnx.checker.check_model`).
4. Verifies 100% numerical equivalence between PyTorch and ONNX Runtime predictions.

Usage:
  # Export A1 Baseline CRNN to ONNX:
  python scripts/export_onnx.py \
    --approach baseline \
    --checkpoint checkpoints/a1_baseline_finetuned_best.pt \
    --vocab-path data/metadata/char_to_idx.json \
    --output checkpoints/a1_baseline_finetuned.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnx
import onnx.shape_inference
import onnxruntime as ort
import torch
import torch.nn as nn

from src.decoder import collapse_ctc_sequence
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.tokenizer import KhmerTokenizer
from src.utils import load_checkpoint


def prepare_model_for_export(model: nn.Module) -> nn.Module:
    """Prepares model for ONNX export by replacing symbol-incompatible pooling layers."""
    model.eval()
    # At fixed H=48, the final conv layer leaves H=1, making AdaptiveAvgPool2d an identity.
    # Replacing it with nn.Identity avoids symbolic shape decomposition errors in PyTorch ONNX.
    if hasattr(model, "cnn") and hasattr(model.cnn, "collapse_height"):
        model.cnn.collapse_height = nn.Identity()
    return model


def export_model_to_onnx(
    checkpoint_path: Path,
    vocab_path: Path,
    output_path: Path,
    approach: str = "baseline",
    opset_version: int = 17,
) -> Path:
    """Exports a trained PyTorch model to ONNX with dynamic input width."""
    # 1. Load vocabulary & tokenizer
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    tokenizer = KhmerTokenizer(char_to_idx)
    vocab_size = tokenizer.vocab_size

    print(f"📖 Loaded vocabulary: {vocab_size} classes (86 Khmer Unicode + 1 blank)")

    # 2. Instantiate architecture
    if approach == "baseline":
        model = BaselineCRNN(num_classes=vocab_size)
    elif approach == "transfer":
        model = TransferCRNN(num_classes=vocab_size, freeze_backbone=False)
    elif approach == "transformer":
        model = TransformerCRNN(num_classes=vocab_size)
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # 3. Load weights if file exists
    if checkpoint_path.is_file():
        load_checkpoint(checkpoint_path, model, map_location="cpu")
        print(f"✓ Loaded weights from: {checkpoint_path.name}")
    else:
        print(f"⚠️ Warning: Checkpoint {checkpoint_path} not found. Exporting model with initialized weights.")

    model = prepare_model_for_export(model)

    # 4. Dummy input [1, 1, 48, 800]
    dummy_input = torch.randn(1, 1, 48, 800, dtype=torch.float32)

    # 5. Export to ONNX with dynamic width and time steps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📦 Exporting to ONNX (Opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["log_probs"],
        dynamic_axes={
            "images": {0: "batch_size", 3: "width"},
            "log_probs": {0: "time_steps", 1: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"✓ Saved raw ONNX model to: {output_path}")

    # 6. Apply ONNX Shape Inference & Validation
    print("🔍 Performing ONNX shape inference and graph validation...")
    onnx_model = onnx.load(str(output_path))
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, str(output_path))
    print("✓ ONNX checker verified graph integrity successfully!")

    # 7. Verification forward pass against PyTorch
    print("🧪 Verifying numerical parity between PyTorch and ONNX Runtime...")
    test_widths = [480, 800, 1200]
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    for test_w in test_widths:
        test_tensor = torch.randn(1, 1, 48, test_w, dtype=torch.float32)
        with torch.no_grad():
            torch_out = model(test_tensor).numpy()

        ort_out = session.run(["log_probs"], {"images": test_tensor.numpy()})[0]

        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        assert max_diff < 1e-4, f"Numerical parity failure at width {test_w}: diff = {max_diff}"

        # Test CTC greedy collapse parity
        seq_len = test_w // 4
        torch_ids = collapse_ctc_sequence(torch_out[:seq_len, 0, :].argmax(axis=-1).tolist())
        ort_ids = collapse_ctc_sequence(ort_out[:seq_len, 0, :].argmax(axis=-1).tolist())
        assert torch_ids == ort_ids, f"Decoded token IDs mismatch at width {test_w}!"

        print(f"  ✓ Width {test_w}px: max |torch - ort| = {max_diff:.2e} | 100% token sequence match")

    print("\n" + "=" * 65)
    print(f"🎉 Model export successfully verified!")
    print(f"📁 ONNX Model: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
    print("=" * 65)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Khmer OCR Model to ONNX.")
    parser.add_argument("--approach", default="baseline", choices=["baseline", "transfer", "transformer"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "a1_baseline_finetuned_best.pt",
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=REPO_ROOT / "data" / "metadata" / "char_to_idx.json",
        help="Path to char_to_idx.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "a1_baseline_finetuned.onnx",
        help="Target output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version (default: 17)")
    args = parser.parse_args()

    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        output_path=args.output,
        approach=args.approach,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()

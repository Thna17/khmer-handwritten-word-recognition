#!/usr/bin/env python3
"""Benchmark PyTorch vs ONNX Runtime Inference for Khmer OCR.

Evaluates latency, throughput (FPS), and memory consumption between native
PyTorch and ONNX Runtime across variable handwritten line widths.

Usage:
  python scripts/benchmark_onnx.py \
    --approach baseline \
    --checkpoint checkpoints/a1_baseline_finetuned_best.pt \
    --onnx-model checkpoints/a1_baseline_finetuned.onnx \
    --num-runs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from scripts.export_onnx import export_model_to_onnx, prepare_model_for_export
from src.decoder import collapse_ctc_sequence
from src.model_baseline import BaselineCRNN
from src.model_transfer import TransferCRNN
from src.model_transformer import TransformerCRNN
from src.tokenizer import KhmerTokenizer
from src.utils import load_checkpoint


def benchmark_engine(
    pytorch_model: nn.Module,
    ort_session: ort.InferenceSession,
    test_widths: list[int] = [400, 800, 1200, 1600],
    num_runs: int = 40,
    warmup_runs: int = 5,
) -> None:
    print("\n" + "=" * 78)
    print("      INFERENCE BENCHMARK: PYTORCH CPU VS ONNX RUNTIME (ORT)")
    print("=" * 78)
    print(f"{'Input Width':<14} | {'PyTorch Latency':<18} | {'ONNX Runtime Latency':<22} | {'Speedup':<10}")
    print("-" * 78)

    pytorch_total_times = []
    ort_total_times = []

    for width in test_widths:
        input_tensor = torch.randn(1, 1, 48, width, dtype=torch.float32)
        input_np = input_tensor.numpy()

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = pytorch_model(input_tensor)
            _ = ort_session.run(["log_probs"], {"images": input_np})

        # Benchmark PyTorch
        pt_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = pytorch_model(input_tensor)
            pt_times.append(time.perf_counter() - t0)

        # Benchmark ONNX Runtime
        ort_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = ort_session.run(["log_probs"], {"images": input_np})
            ort_times.append(time.perf_counter() - t0)

        pt_mean = float(np.mean(pt_times)) * 1000.0
        ort_mean = float(np.mean(ort_times)) * 1000.0
        speedup = pt_mean / ort_mean if ort_mean > 0 else 1.0

        pytorch_total_times.extend(pt_times)
        ort_total_times.extend(ort_times)

        print(
            f"{width} px (W/4={width//4}) | "
            f"{pt_mean:>6.2f} ms ({1000.0/pt_mean:>4.1f} fps) | "
            f"{ort_mean:>6.2f} ms ({1000.0/ort_mean:>4.1f} fps) | "
            f"{speedup:>5.2f}x"
        )

    print("-" * 78)
    overall_pt = float(np.mean(pytorch_total_times)) * 1000.0
    overall_ort = float(np.mean(ort_total_times)) * 1000.0
    overall_speedup = overall_pt / overall_ort if overall_ort > 0 else 1.0

    print(
        f"{'OVERALL AVERAGE':<14} | "
        f"{overall_pt:>6.2f} ms ({1000.0/overall_pt:>4.1f} fps) | "
        f"{overall_ort:>6.2f} ms ({1000.0/overall_ort:>4.1f} fps) | "
        f"{overall_speedup:>5.2f}x"
    )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime.")
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
        "--onnx-model",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "a1_baseline_finetuned.onnx",
        help="Path to exported ONNX model",
    )
    parser.add_argument("--num-runs", type=int, default=30, help="Benchmark iterations per width (default: 30)")
    args = parser.parse_args()

    # Load vocab
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    tokenizer = KhmerTokenizer(char_to_idx)

    # Load PyTorch model
    if args.approach == "baseline":
        model = BaselineCRNN(num_classes=tokenizer.vocab_size)
    elif args.approach == "transfer":
        model = TransferCRNN(num_classes=tokenizer.vocab_size, freeze_backbone=False)
    else:
        model = TransformerCRNN(num_classes=tokenizer.vocab_size)

    if args.checkpoint.is_file():
        load_checkpoint(args.checkpoint, model, map_location="cpu")
        print(f"✓ PyTorch model loaded: {args.checkpoint.name}")
    else:
        print(f"ℹ️ Checkpoint not found; benchmarking with initialized architecture weights.")

    model = prepare_model_for_export(model)

    # Ensure ONNX model exists
    if not args.onnx_model.is_file():
        print(f"📦 Exporting ONNX model to {args.onnx_model} before benchmarking...")
        export_model_to_onnx(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab_path,
            output_path=args.onnx_model,
            approach=args.approach,
        )

    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    print(f"✓ ONNX Runtime session initialized: {args.onnx_model.name}")

    benchmark_engine(
        pytorch_model=model,
        ort_session=session,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()

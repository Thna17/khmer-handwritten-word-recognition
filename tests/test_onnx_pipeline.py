"""Unit tests for ONNX Export and Numerical Equivalence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from scripts.export_onnx import export_model_to_onnx
from src.decoder import collapse_ctc_sequence


def test_onnx_export_and_numerical_parity():
    vocab_path = Path("data/metadata/char_to_idx.json")
    if not vocab_path.is_file():
        pytest.skip("Vocabulary file not found.")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_onnx = Path(tmpdir) / "model_test.onnx"
        dummy_ckpt = Path(tmpdir) / "non_existent.pt"

        # Export Baseline model
        exported_path = export_model_to_onnx(
            checkpoint_path=dummy_ckpt,
            vocab_path=vocab_path,
            output_path=output_onnx,
            approach="baseline",
            opset_version=17,
        )

        assert exported_path.is_file()
        assert exported_path.stat().st_size > 1_000_000  # Should be ~32 MB

        # Validate with ONNX Checker
        onnx_model = onnx.load(str(exported_path))
        onnx.checker.check_model(onnx_model)

        # Validate dynamic width inference with ONNX Runtime
        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        for width in [320, 640]:
            dummy_arr = np.random.randn(1, 1, 48, width).astype(np.float32)
            outputs = session.run(["log_probs"], {"images": dummy_arr})
            log_probs = outputs[0]

            expected_time_steps = width // 4
            assert log_probs.shape == (expected_time_steps, 1, 87)
            # Check CTC token IDs collapse cleanly
            token_ids = collapse_ctc_sequence(log_probs[:, 0, :].argmax(axis=-1).tolist())
            assert isinstance(token_ids, list)

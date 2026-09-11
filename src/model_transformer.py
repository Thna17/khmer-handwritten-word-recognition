"""Approach 3 (A3): the SAME VGG-style CNN backbone as A1 (trained from
scratch) + a small Transformer encoder REPLACING the BiLSTM + the same
kind of Linear + LogSoftmax + CTC head. See PROJECT_CONTEXT.md Section 2
— CNN and CTC head are held constant so this comparison isolates exactly
one choice: BiLSTM vs. Transformer encoder for sequence modeling.

Shape flow (H = fixed input height = 48, B = batch size):
    input:        [B, 1, H, W]
    CNN features: [B, 512, 1, T]      identical to A1's CNN; T = W // 4
    sequence:     [T, B, 512]         squeeze height, move width to time dim
    + positional encoding (sinusoidal, added elementwise) -- a Transformer
      has no inherent sense of order the way an RNN does; without this,
      shuffling the time steps would not change its output at all (see
      the permutation-equivariance tests in tests/test_model_transformer.py)
    Transformer:  [T, B, 512]         self-attention encoder layers
    logits:       [T, B, num_classes]
    log_probs:    [T, B, num_classes] LogSoftmax over the class dim, for CTCLoss

Note: like A1 and A2, this model does NOT use an attention padding mask
for batches of mixed-width images -- the Transformer, like A1's BiLSTM,
processes the full (white-padded) width, and only the downstream CTC
input_lengths (from compute_sequence_length) tell the loss/decoder to
ignore predictions past each sample's true length. Adding padding-aware
masking to only this one approach would make the 3-way comparison less
fair (an uncontrolled extra variable); if this is ever revisited, it
should be applied equally to all three approaches at once.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model_baseline import VGGFeatureExtractor


class SinusoidalPositionalEncoding(nn.Module):
    """Classic fixed (non-learnable) sinusoidal positional encoding from
    "Attention Is All You Need" — gives the Transformer a sense of
    time-step order, since self-attention alone is permutation-invariant."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model] -- broadcasts over batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, d_model] -> same shape, with position added."""
        t = x.shape[0]
        if t > self.pe.shape[0]:
            raise ValueError(f"sequence length {t} exceeds this encoding's max_len={self.pe.shape[0]}")
        return x + self.pe[:t]


class TransformerCRNN(nn.Module):
    """A3: VGG CNN (from scratch, same as A1) + small Transformer encoder
    (replacing the BiLSTM) + CTC head."""

    WIDTH_DOWNSAMPLE_FACTOR = 4  # identical to A1 -- same CNN backbone
    CNN_OUT_CHANNELS = 512       # identical to A1's VGGFeatureExtractor output channels

    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        if d_model != self.CNN_OUT_CHANNELS:
            raise ValueError(
                f"d_model must equal the CNN's output channels ({self.CNN_OUT_CHANNELS}) "
                "since no projection layer sits between them; change d_model or add one."
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.cnn = VGGFeatureExtractor(in_channels=1)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # [T, B, C], matching this project's sequence convention everywhere else
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 1, H, W] -> log_probs: [T, B, num_classes]."""
        if images.shape[-1] < 2 * self.WIDTH_DOWNSAMPLE_FACTOR:
            raise ValueError(
                f"Image width {images.shape[-1]} is too small for this CNN's "
                f"downsampling (needs at least {2 * self.WIDTH_DOWNSAMPLE_FACTOR}px wide)."
            )

        features = self.cnn(images)  # [B, 512, 1, T]
        b, c, h, t = features.shape
        assert h == 1, f"expected the CNN to collapse height to 1, got height={h}"

        sequence = features.squeeze(2).permute(2, 0, 1)  # [T, B, 512]
        sequence = self.positional_encoding(sequence)
        encoded = self.transformer(sequence)              # [T, B, 512]
        logits = self.classifier(encoded)                 # [T, B, num_classes]
        return self.log_softmax(logits)                   # [T, B, num_classes]

    def compute_sequence_length(self, input_width: int) -> int:
        """Identical formula to A1 — same CNN backbone, same downsampling."""
        return input_width // self.WIDTH_DOWNSAMPLE_FACTOR

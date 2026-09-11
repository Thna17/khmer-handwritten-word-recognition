"""Approach 1 (A1): VGG-style CNN (trained from scratch) + 2-layer BiLSTM
+ Linear + LogSoftmax + CTC. This is the locked baseline architecture
from PROJECT_CONTEXT.md — A2 and A3 reuse this same CNN-to-sequence
framing, swapping only the backbone (A2) or the sequence model (A3).

Shape flow (H = fixed input height = 48, B = batch size):
    input:        [B, 1, H, W]
    CNN features: [B, 512, 1, T]      height collapsed to 1; T = W // 4
    sequence:     [T, B, 512]         squeeze height, move width to time dim
    BiLSTM:       [T, B, 512]         2 directions * hidden_size(256)
    logits:       [T, B, num_classes]
    log_probs:    [T, B, num_classes] LogSoftmax over the class dim, for CTCLoss

num_classes must equal tokenizer.vocab_size (real characters + 1 blank).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VGGFeatureExtractor(nn.Module):
    """Collapses [B, 1, 48, W] -> [B, 512, 1, W // 4].

    Height goes 48 -> 24 -> 12 -> 6 -> 3 -> 1 across the 5 blocks. Width is
    halved in blocks 1-2 (regular 2x2 pooling) then held fixed in blocks
    3-5 (height-only pooling), so the output sequence keeps enough time
    steps for CTC alignment instead of collapsing width too aggressively.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # H: 48->24, W: W->W//2
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # H: 24->12, W: W//2->W//4
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # H: 12->6, W unchanged
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # H: 6->3, W unchanged
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=0),    # H: 3->1, W unchanged
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x  # [B, 512, 1, T]


class BaselineCRNN(nn.Module):
    """A1: VGG CNN (from scratch) + 2-layer BiLSTM(256, dropout=0.2) + CTC head."""

    WIDTH_DOWNSAMPLE_FACTOR = 4  # two (2,2)-stride-2 maxpools in the CNN backbone

    def __init__(self, num_classes: int, hidden_size: int = 256, num_lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cnn = VGGFeatureExtractor(in_channels=1)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
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
        rnn_out, _ = self.rnn(sequence)                  # [T, B, 2*hidden_size]
        logits = self.classifier(rnn_out)                # [T, B, num_classes]
        return self.log_softmax(logits)                  # [T, B, num_classes]

    def compute_sequence_length(self, input_width: int) -> int:
        """Given one image's true (pre-padding) pixel width, return T — the
        number of time steps the CNN produces for it. Needed to build the
        correct `input_lengths` for CTCLoss: a sample narrower than its
        batch's padded max width must NOT have its white padding columns
        counted as real time steps."""
        return input_width // self.WIDTH_DOWNSAMPLE_FACTOR

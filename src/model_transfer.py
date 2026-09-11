"""Approach 2 (A2): pretrained ResNet-18 backbone (frozen OR fine-tuned)
+ the SAME 2-layer BiLSTM(256, dropout=0.2) + Linear + LogSoftmax + CTC
head as A1. Only the CNN backbone and training strategy differ from the
baseline — see PROJECT_CONTEXT.md Section 2 for why that isolates the
"transfer learning vs. from-scratch" comparison cleanly.

Shape flow (H = fixed input height = 48, B = batch size):
    input:        [B, 1, H, W]
    -> replicated to fake-RGB [B, 3, H, W] (ImageNet weights expect 3 channels)
    CNN features: [B, 128, 1, T]      ResNet-18 TRUNCATED after layer2
                                        (full ResNet-18 downsamples by /32,
                                        which would leave too few time
                                        steps for CTC; stopping after
                                        layer2 downsamples by /8 instead)
    sequence:     [T, B, 128]
    BiLSTM:       [T, B, 512]         2 directions * hidden_size(256)
    logits:       [T, B, num_classes]
    log_probs:    [T, B, num_classes] LogSoftmax over the class dim, for CTCLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class TruncatedResNetBackbone(nn.Module):
    """ResNet-18's stem + layer1 + layer2 (stops BEFORE layer3/layer4),
    plus an adaptive pool that collapses height to 1 while leaving width
    untouched. Downsamples width by /8 total, not ResNet-18's usual /32 —
    the full network would crush a 256px-wide word image down to just 8
    time steps, too few to align longer words under CTC.
    """

    OUT_CHANNELS = 128  # ResNet-18 layer2's output channel count

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.collapse_height = nn.AdaptiveAvgPool2d((1, None))

        self.frozen = freeze
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)  # grayscale [B,1,H,W] -> fake-RGB [B,3,H,W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.collapse_height(x)  # [B, 128, 1, T]

    def train(self, mode: bool = True) -> "TruncatedResNetBackbone":
        """A frozen backbone must also stay in eval mode for BatchNorm —
        otherwise its running statistics would keep adapting to this
        task's batches even though the weights themselves are frozen,
        which is not what "frozen" is supposed to mean."""
        super().train(mode)
        if self.frozen:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


class TransferCRNN(nn.Module):
    """A2: pretrained (or randomly initialized, for testing) ResNet-18
    backbone + 2-layer BiLSTM(256, dropout=0.2) + CTC head."""

    INPUT_HEIGHT = 48

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = TruncatedResNetBackbone(pretrained=pretrained, freeze=freeze_backbone)
        self.rnn = nn.LSTM(
            input_size=self.cnn.OUT_CHANNELS,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self._sequence_length_cache: dict[int, int] = {}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 1, H, W] -> log_probs: [T, B, num_classes]."""
        features = self.cnn(images)  # [B, 128, 1, T]
        b, c, h, t = features.shape
        assert h == 1, f"expected the backbone to collapse height to 1, got height={h}"

        sequence = features.squeeze(2).permute(2, 0, 1)  # [T, B, 128]
        rnn_out, _ = self.rnn(sequence)                  # [T, B, 2*hidden_size]
        logits = self.classifier(rnn_out)                # [T, B, num_classes]
        return self.log_softmax(logits)                  # [T, B, num_classes]

    def compute_sequence_length(self, input_width: int) -> int:
        """Given one image's true (pre-padding) pixel width, return T.

        Unlike A1's simple width//4 VGG backbone, ResNet's stride/padding
        interactions across conv1 + maxpool + a strided residual block
        are complex enough that a hand-derived formula risks a silent
        off-by-one. A real forward pass through the (tiny, CPU-cheap)
        backbone can't be wrong by construction, so that's what this
        does — cached per width, since the same widths recur across many
        batches over a training run.
        """
        if input_width not in self._sequence_length_cache:
            device = next(self.parameters()).device
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.INPUT_HEIGHT, input_width, device=device)
                features = self.cnn(dummy)
            self._sequence_length_cache[input_width] = features.shape[-1]
        return self._sequence_length_cache[input_width]

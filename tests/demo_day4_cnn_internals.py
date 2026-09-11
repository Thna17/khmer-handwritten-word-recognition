"""
Day 4 Educational Demonstration Script: CNN Internals
(Convolution, Feature Maps, ReLU, Pooling, Receptive Field)
Run this script:
    .venv/bin/python tests/demo_day4_cnn_internals.py

Goal: stop treating nn.Conv2d/nn.MaxPool2d as black boxes. See the raw
arithmetic once, confirm it matches PyTorch's implementation, then look
at real feature maps produced from an actual digit image.

No manual convolution matrix calculations are needed elsewhere in this
project — this script exists purely to build intuition, one time.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# -------------------------------------------------------------
# Pillar 1: Convolution — the raw arithmetic, done by hand once
# -------------------------------------------------------------
def demo_convolution_by_hand():
    print_banner("PILLAR 1: CONVOLUTION — THE RAW ARITHMETIC")
    print("Concept: a kernel (small grid of numbers) slides over the image.")
    print("At each position, multiply overlapping numbers element-wise and sum.")
    print("That single sum becomes ONE pixel in the output feature map.")

    # A tiny 5x5 image with a bright vertical edge in the middle
    # (0 = dark background, 1 = bright stroke)
    image = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float32)
    print(f"\n1. Tiny 5x5 input image (a vertical stroke):\n{image}")

    # A classic vertical-edge-detecting kernel (Sobel-style, simplified)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=np.float32)
    print(f"\n2. 3x3 vertical-edge kernel:\n{kernel}")

    # Manual convolution: slide the 3x3 kernel over every valid 3x3 patch.
    # With a 5x5 image and a 3x3 kernel (no padding), output is 3x3.
    output = np.zeros((3, 3), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            patch = image[row:row + 3, col:col + 3]
            output[row, col] = np.sum(patch * kernel)  # element-wise multiply, then sum

    print("\n3. Manual convolution result (3x3 output, computed with nested loops):")
    print(output)
    print("   -> High values sit where the kernel found a strong left-to-right")
    print("      brightness change: exactly where our vertical edge is.")

    # Now verify PyTorch's nn.Conv2d produces the IDENTICAL numbers when we
    # manually load the same kernel as its weight.
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.from_numpy(kernel).view(1, 1, 3, 3)

    image_tensor = torch.from_numpy(image).view(1, 1, 5, 5)  # [B=1, C=1, H=5, W=5]
    torch_output = conv(image_tensor).squeeze().detach().numpy()

    print("\n4. nn.Conv2d output with the SAME kernel loaded as weights:")
    print(torch_output)
    match = np.allclose(output, torch_output, atol=1e-5)
    print(f"\n5. Manual math matches nn.Conv2d? {'YES' if match else 'NO (BUG!)'}")
    assert match, "ERROR: manual convolution does not match nn.Conv2d!"
    print("   -> Confirmed: nn.Conv2d is doing exactly this sliding dot-product,")
    print("      just fast and for many kernels ('output channels') at once.")


# -------------------------------------------------------------
# Pillar 2: ReLU — keep only positive activations
# -------------------------------------------------------------
def demo_relu():
    print_banner("PILLAR 2: ReLU — KEEP ONLY THE POSITIVE SIGNAL")
    print("Concept: ReLU(x) = max(0, x). Negative numbers become 0; positive")
    print("numbers pass through unchanged. This lets the network express")
    print("'this pattern is present' (positive) vs 'ignore this' (zero),")
    print("and without it, stacking layers would collapse into one big")
    print("linear operation (no extra learning power).")

    feature_map = torch.tensor([
        [ 2.0, -1.0,  0.5],
        [-3.0,  4.0, -0.2],
        [ 0.0,  1.5, -5.0],
    ])
    print(f"\n1. Raw feature map (before ReLU):\n{feature_map}")

    activated = F.relu(feature_map)
    print(f"\n2. After ReLU:\n{activated}")
    print("   -> Every negative number became 0; positives are untouched.")


# -------------------------------------------------------------
# Pillar 3: Pooling — shrink the feature map, keep the strongest signal
# -------------------------------------------------------------
def demo_pooling():
    print_banner("PILLAR 3: MAX POOLING — DOWNSAMPLE, KEEP THE STRONGEST RESPONSE")
    print("Concept: slide a small window (e.g. 2x2) over the feature map and")
    print("keep only the maximum value in each window. This shrinks the map")
    print("(less computation downstream) and adds tolerance to small shifts:")
    print("if the edge moves 1 pixel, the max in a 2x2 window often stays the same.")

    feature_map = torch.tensor([
        [1.0, 3.0, 2.0, 0.0],
        [4.0, 2.0, 1.0, 5.0],
        [0.0, 1.0, 6.0, 2.0],
        [2.0, 3.0, 1.0, 1.0],
    ])
    print(f"\n1. 4x4 feature map:\n{feature_map}")

    pooled = F.max_pool2d(feature_map.view(1, 1, 4, 4), kernel_size=2)
    print(f"\n2. After 2x2 max pooling (stride 2):\n{pooled.squeeze()}")
    print("   -> 4x4 became 2x2. Each output number is the max of its 2x2 region.")
    print("      Top-left region [[1,3],[4,2]] -> max is 4. Matches above.")


# -------------------------------------------------------------
# Pillar 4: Receptive field — how far one output pixel "sees"
# -------------------------------------------------------------
def demo_receptive_field():
    print_banner("PILLAR 4: RECEPTIVE FIELD — HOW MUCH INPUT ONE OUTPUT PIXEL SEES")
    print("Concept: one 3x3 conv output pixel is influenced by a 3x3 patch of")
    print("input. Stack a SECOND 3x3 conv on top, and each of ITS output")
    print("pixels depends on a 3x3 patch of the FIRST layer's output — which")
    print("itself came from a 3x3 patch of the original image. The effective")
    print("receptive field grows to 5x5. Deeper CNNs see progressively larger")
    print("regions, which is how they go from detecting edges to whole shapes.")

    torch.manual_seed(0)
    # Use requires_grad on the input so we can trace, via backward(), exactly
    # which input pixels affect a chosen output pixel (a non-zero gradient
    # means "this input pixel influences this output").
    image = torch.randn(1, 1, 9, 9, requires_grad=True)

    conv1 = nn.Conv2d(1, 1, kernel_size=3)
    conv2 = nn.Conv2d(1, 1, kernel_size=3)

    out1 = conv1(image)   # 9x9 -> 7x7 (one 3x3 conv)
    out2 = conv2(out1)    # 7x7 -> 5x5 (a second 3x3 conv stacked on top)
    print(f"\n1. Input shape: {tuple(image.shape)[2:]}  -> after conv1: {tuple(out1.shape)[2:]}  -> after conv2: {tuple(out2.shape)[2:]}")

    # Pick the center output pixel of the final 5x5 map and ask: which
    # input pixels affect it?
    center = out2[0, 0, 2, 2]
    center.backward()

    influence_mask = (image.grad[0, 0].abs() > 1e-8).int()
    print("\n2. Which of the 9x9 input pixels influence that ONE center output pixel?")
    print("   (1 = influences it, 0 = no effect)")
    print(influence_mask.numpy())

    num_influencing = influence_mask.sum().item()
    print(f"\n3. Number of input pixels influencing it: {num_influencing}")
    print("   -> Expected 5x5 = 25 pixels: two stacked 3x3 convs give a 5x5 receptive field.")
    assert num_influencing == 25, f"ERROR: expected 25, got {num_influencing}"
    print("   Verification: PASSED - receptive field grew from 3x3 to 5x5 by stacking layers.")


# -------------------------------------------------------------
# Pillar 5: Real feature maps on a real digit image (visual proof)
# -------------------------------------------------------------
def demo_real_feature_maps():
    print_banner("PILLAR 5: REAL FEATURE MAPS ON A REAL DIGIT IMAGE")
    print("Concept: apply the same conv -> ReLU -> pool sequence to an actual")
    print("handwritten digit and SAVE a picture, so you can see what each")
    print("stage produces instead of only reading numbers.")

    try:
        from torchvision import datasets, transforms
        import matplotlib
        matplotlib.use("Agg")  # no display needed, just save to file
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\nSkipping visual demo — missing dependency: {e}")
        return

    dataset = datasets.MNIST(root="data/raw/mnist_cache", train=True, download=True,
                              transform=transforms.ToTensor())
    image, label = dataset[0]  # [1, 28, 28], a real handwritten digit
    image_batch = image.unsqueeze(0)  # [1, 1, 28, 28]

    # Hand-picked kernels: vertical edge, horizontal edge, blur.
    vertical_kernel = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
    horizontal_kernel = torch.tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]])
    blur_kernel = torch.full((3, 3), 1.0 / 9.0)

    conv = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight[0, 0] = vertical_kernel
        conv.weight[1, 0] = horizontal_kernel
        conv.weight[2, 0] = blur_kernel

    conv_out = conv(image_batch)          # [1, 3, 28, 28]
    relu_out = F.relu(conv_out)           # same shape, negatives zeroed
    pool_out = F.max_pool2d(relu_out, 2)  # [1, 3, 14, 14]

    print(f"\n1. Digit label in this example: {label}")
    print(f"2. conv output shape: {tuple(conv_out.shape)}  [B=1, 3 kernels, H=28, W=28]")
    print(f"3. relu output shape: {tuple(relu_out.shape)}  (same shape, negatives zeroed)")
    print(f"4. pool output shape: {tuple(pool_out.shape)}  [B=1, 3 kernels, H=14, W=14]")

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    kernel_names = ["Vertical-edge kernel", "Horizontal-edge kernel", "Blur kernel"]

    axes[0, 0].imshow(image.squeeze(), cmap="gray")
    axes[0, 0].set_title(f"Original digit ({label})")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for i in range(3):
        axes[1, i].imshow(conv_out[0, i].detach(), cmap="gray")
        axes[1, i].set_title(f"{kernel_names[i]}\n(after conv)")
        axes[2, i].imshow(pool_out[0, i].detach(), cmap="gray")
        axes[2, i].set_title(f"{kernel_names[i]}\n(after ReLU + pool)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    import os
    os.makedirs("results", exist_ok=True)
    output_path = "results/day4_cnn_feature_maps.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\n5. Saved visual comparison to: {output_path}")
    print("   Row 1: original digit. Row 2: raw conv output per kernel.")
    print("   Row 3: same, after ReLU + 2x2 max pooling (smaller, cleaner).")


if __name__ == "__main__":
    demo_convolution_by_hand()
    demo_relu()
    demo_pooling()
    demo_receptive_field()
    demo_real_feature_maps()
    print("\n" + "=" * 65)
    print("  ALL 5 CNN-INTERNALS PILLARS SUCCESSFULLY DEMONSTRATED!")
    print("=" * 65 + "\n")

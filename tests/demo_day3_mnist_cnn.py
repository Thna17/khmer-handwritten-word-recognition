"""
Day 3 Educational Demonstration Script: Train a Real CNN on MNIST (Gate #1)
Run this script:
    .venv/bin/python tests/demo_day3_mnist_cnn.py

Goal: train an actual image classifier end-to-end on real data, using the
exact same 4-step loop practiced on toy data in Day 2
(forward -> loss -> backward() -> optimizer.step()). This is the first
time the loop touches real images, so it doubles as Gate #1: "can I train
an image CNN myself, with AI assistance, and understand every step?"

MNIST is used only as a rehearsal dataset. It has nothing to do with
Khmer OCR — it is grayscale digit images (0-9), one digit per image,
which is structurally similar enough to "one grayscale image -> one
label" to practice the full pipeline before touching real Khmer data.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def get_device() -> torch.device:
    # Apple Silicon GPU (MPS) if available, else CPU. No CUDA on this machine.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------
# Pillar 1: Real data — where it comes from, what shape it has
# -------------------------------------------------------------
def load_mnist():
    print_banner("PILLAR 1: LOADING REAL DATA (MNIST)")
    print("Concept: torchvision.datasets.MNIST downloads 70,000 handwritten")
    print("digit images (0-9) and wraps them as a PyTorch Dataset for us.")
    print("This is the same Dataset interface from Day 2 — someone else")
    print("already wrote __len__ and __getitem__ for MNIST.")

    # ToTensor(): converts a PIL image [0,255] -> float tensor [0,1], shape [C,H,W]
    # Normalize(mean, std): rescales pixel values to roughly [-1, 1], which
    # helps training converge faster and more stably.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST's known mean/std
    ])

    train_dataset = datasets.MNIST(
        root="data/raw/mnist_cache", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data/raw/mnist_cache", train=False, download=True, transform=transform
    )

    print(f"\n1. len(train_dataset) = {len(train_dataset)}")
    print(f"2. len(test_dataset)  = {len(test_dataset)}")

    image, label = train_dataset[0]
    print(f"\n3. train_dataset[0] -> image.shape = {tuple(image.shape)}  [C=1, H=28, W=28]")
    print(f"   label = {label}  (a plain Python int, the digit class)")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    batch_images, batch_labels = next(iter(train_loader))
    print(f"\n4. One training batch: images.shape = {tuple(batch_images.shape)}  [B=128, C=1, H=28, W=28]")
    print(f"   labels.shape = {tuple(batch_labels.shape)}  [B=128], values in 0..9")

    return train_loader, test_loader


# -------------------------------------------------------------
# Pillar 2: The model — a small CNN classifier
# -------------------------------------------------------------
class MnistCNN(nn.Module):
    """Conv -> ReLU -> Pool, twice, then flatten -> Linear -> 10 class logits.

    This mirrors the "CNN extracts features, then a final layer maps to
    class scores" pattern the CRNN will reuse later (except the CRNN feeds
    its CNN features into a BiLSTM instead of flattening immediately).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # [B,1,28,28] -> [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B,16,14,14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # -> [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B,32,7,7]
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)          # [B, 32, 7, 7]
        x = x.flatten(start_dim=1)    # [B, 32*7*7] = [B, 1568]
        return self.classifier(x)     # [B, num_classes]


def demo_model_shapes(device: torch.device):
    print_banner("PILLAR 2: THE MODEL — CNN FORWARD PASS SHAPES")
    model = MnistCNN().to(device)
    print(f"\n1. Model architecture:\n{model}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. Total learnable parameters: {num_params:,}")

    dummy_batch = torch.randn(8, 1, 28, 28, device=device)
    output = model(dummy_batch)
    print(f"\n3. Input shape:  {tuple(dummy_batch.shape)}  [B=8, C=1, H=28, W=28]")
    print(f"4. Output shape: {tuple(output.shape)}  [B=8, classes=10]")
    print("   -> Each row is 10 raw scores ('logits'), one per digit class.")
    return model


# -------------------------------------------------------------
# Pillar 3: Training loop over real data (multiple epochs)
# -------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch_idx):
    model.train()  # enables dropout/batchnorm training behavior (none here, but good habit)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 150 == 0:
            print(f"   epoch {epoch_idx} | batch {batch_idx:3d}/{len(loader)} | loss = {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def demo_training(model, train_loader, device):
    print_banner("PILLAR 3: TRAINING LOOP ON REAL DATA")
    print("Concept: identical 4-step loop from Day 2 (zero_grad -> forward ->")
    print("loss -> backward -> step), just now with 60,000 real images.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        epoch_loss, epoch_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        elapsed = time.time() - start
        print(f"-> Epoch {epoch} done in {elapsed:.1f}s | avg train loss = {epoch_loss:.4f} | train acc = {epoch_acc*100:.2f}%")

    return model


# -------------------------------------------------------------
# Pillar 4: Evaluation on held-out test data
# -------------------------------------------------------------
def demo_evaluation(model, test_loader, device):
    print_banner("PILLAR 4: EVALUATION ON UNSEEN TEST DATA")
    print("Concept: test images were NEVER shown during training. Accuracy")
    print("here estimates how well the model generalizes, not memorizes.")

    model.eval()  # disables training-only behavior
    correct = 0
    total = 0
    examples = []

    with torch.no_grad():  # no gradients needed for evaluation -> faster, less memory
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if len(examples) < 5:
                for i in range(min(5 - len(examples), images.size(0))):
                    examples.append((labels[i].item(), predicted[i].item()))

    accuracy = correct / total
    print(f"\n1. Test set size: {total}")
    print(f"2. Test accuracy: {accuracy*100:.2f}%")
    print("\n3. A few example predictions (ground_truth -> predicted):")
    for true_label, pred_label in examples:
        marker = "OK" if true_label == pred_label else "WRONG"
        print(f"   {true_label} -> {pred_label}  [{marker}]")

    assert accuracy > 0.90, f"ERROR: accuracy {accuracy*100:.2f}% is too low — something is likely broken."
    print("\n   Verification: PASSED - model generalizes well above chance (>90%).")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = load_mnist()
    model = demo_model_shapes(device)
    model = demo_training(model, train_loader, device)
    demo_evaluation(model, test_loader, device)

    print("\n" + "=" * 65)
    print("  GATE #1 PASSED: MNIST CNN TRAINED AND EVALUATED END-TO-END!")
    print("=" * 65 + "\n")

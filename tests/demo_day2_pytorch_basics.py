"""
Day 2 Educational Demonstration Script: Core PyTorch Building Blocks
Run this script:
    .venv/bin/python tests/demo_day2_pytorch_basics.py

Goal: get comfortable with the exact pieces (Tensor, nn.Module, Dataset,
DataLoader, optimizer, loss.backward(), optimizer.step()) that the real
CRNN training loop will use later. Nothing here is Khmer-specific yet —
this is pure PyTorch mechanics practice.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# -------------------------------------------------------------
# Pillar 1: Tensors and the Batch x Channel x Height x Width shape
# -------------------------------------------------------------
def demo_tensor_shapes():
    print_banner("PILLAR 1: TENSORS & THE [B, C, H, W] SHAPE")
    print("Concept: every image batch in PyTorch is one 4D tensor:")
    print("   B = batch size (how many images at once)")
    print("   C = channels (1 = grayscale, 3 = RGB)")
    print("   H = height in pixels")
    print("   W = width in pixels")

    # A tiny batch of 4 grayscale images, 48px tall, 100px wide.
    batch = torch.zeros(4, 1, 48, 100)
    print(f"\n1. batch.shape = {tuple(batch.shape)}  -> [B=4, C=1, H=48, W=100]")

    # Basic tensor ops you will use constantly:
    print(f"2. batch.dtype = {batch.dtype}  (default float32)")
    print(f"3. batch.numel() = {batch.numel()} total scalar values")

    # Indexing: grab image #0 out of the batch -> shape drops the B dim
    single_image = batch[0]
    print(f"4. batch[0].shape = {tuple(single_image.shape)}  -> [C=1, H=48, W=100] (one image)")

    # unsqueeze/squeeze: how you add/remove the batch dimension
    restored = single_image.unsqueeze(0)
    print(f"5. single_image.unsqueeze(0).shape = {tuple(restored.shape)}  -> batch dim added back")

    # A common bug source: reshape/view must preserve total element count
    flat = single_image.view(-1)
    print(f"6. single_image.view(-1).shape = {tuple(flat.shape)}  -> flattened to 1D ({flat.numel()} values)")


# -------------------------------------------------------------
# Pillar 2: nn.Module — defining a model
# -------------------------------------------------------------
class TinyLinearModel(nn.Module):
    """The simplest possible model: input_dim -> output_dim, one linear layer."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def demo_nn_module():
    print_banner("PILLAR 2: nn.Module — HOW A MODEL IS DEFINED")
    print("Concept: every PyTorch model is a class with:")
    print("   __init__  -> declare the layers (they own learnable weights)")
    print("   forward   -> define how input tensors flow through those layers")

    model = TinyLinearModel(input_dim=10, output_dim=3)
    print(f"\n1. Model architecture:\n{model}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. Total learnable parameters: {num_params}")
    print("   (10 inputs x 3 outputs = 30 weights + 3 biases = 33)")

    dummy_input = torch.randn(5, 10)  # batch of 5 samples, 10 features each
    output = model(dummy_input)
    print(f"\n3. Input shape:  {tuple(dummy_input.shape)}  [batch=5, features=10]")
    print(f"4. Output shape: {tuple(output.shape)}  [batch=5, features=3]")
    print("   -> The model transformed each 10-dim sample into a 3-dim prediction.")


# -------------------------------------------------------------
# Pillar 3: Dataset + DataLoader — feeding data in batches
# -------------------------------------------------------------
class ToyDataset(Dataset):
    """A fake dataset of (feature_vector, label) pairs, mimicking the
    (image, label) shape our real Khmer dataset.py will have later."""

    def __init__(self, num_samples: int = 20, feature_dim: int = 10):
        # Fixed seed so the "data" is reproducible across runs.
        generator = torch.Generator().manual_seed(0)
        self.features = torch.randn(num_samples, feature_dim, generator=generator)
        self.labels = torch.randint(0, 3, (num_samples,), generator=generator)

    def __len__(self):
        # PyTorch calls this to know how many samples exist.
        return len(self.features)

    def __getitem__(self, idx):
        # PyTorch calls this once per sample when building a batch.
        return self.features[idx], self.labels[idx]


def demo_dataset_and_dataloader():
    print_banner("PILLAR 3: Dataset & DataLoader")
    print("Concept: Dataset answers 'how many samples?' and 'give me sample i'.")
    print("DataLoader wraps a Dataset to automatically build shuffled batches.")

    dataset = ToyDataset(num_samples=20, feature_dim=10)
    print(f"\n1. len(dataset) = {len(dataset)}")

    feature, label = dataset[0]
    print(f"2. dataset[0] -> feature.shape={tuple(feature.shape)}, label={label.item()}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    first_batch_features, first_batch_labels = next(iter(loader))
    print(f"\n3. One batch from DataLoader(batch_size=4):")
    print(f"   features.shape = {tuple(first_batch_features.shape)}  [batch=4, feature_dim=10]")
    print(f"   labels.shape   = {tuple(first_batch_labels.shape)}  [batch=4]")
    print(f"4. Number of batches per epoch: {len(loader)}  (20 samples / batch_size 4)")


# -------------------------------------------------------------
# Pillar 4: The training loop — forward, loss, backward, step
# -------------------------------------------------------------
def demo_training_step():
    print_banner("PILLAR 4: forward -> loss -> backward() -> optimizer.step()")
    print("Concept: this 4-step loop is IDENTICAL for our future CRNN+CTC training.")
    print("Only the model and loss function will change; the loop shape will not.")

    torch.manual_seed(0)
    dataset = ToyDataset(num_samples=20, feature_dim=10)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TinyLinearModel(input_dim=10, output_dim=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Snapshot one weight BEFORE training so we can prove it changes.
    weight_before = model.linear.weight.clone().detach()

    features, labels = next(iter(loader))

    # Step 1: forward pass -> raw prediction scores ("logits")
    predictions = model(features)
    print(f"\n1. Forward pass: predictions.shape = {tuple(predictions.shape)} [batch=4, classes=3]")

    # Step 2: loss -> a single number measuring how wrong the predictions are
    loss = loss_fn(predictions, labels)
    print(f"2. Loss (CrossEntropy) = {loss.item():.4f}  <- one scalar for the whole batch")

    # Step 3: backward() -> compute how much each weight contributed to the loss
    optimizer.zero_grad()  # clear old gradients first (PyTorch accumulates by default)
    loss.backward()
    grad_norm = model.linear.weight.grad.norm().item()
    print(f"3. loss.backward() computed gradients. grad norm = {grad_norm:.4f}")
    print("   -> Every weight in model.linear now has a .grad tensor (its 'blame' for the loss).")

    # Step 4: optimizer.step() -> actually update the weights using those gradients
    optimizer.step()
    weight_after = model.linear.weight.clone().detach()
    changed = not torch.equal(weight_before, weight_after)
    print(f"4. optimizer.step() applied the update.")
    print(f"   Did the weights actually change? {'YES' if changed else 'NO (BUG!)'}")
    assert changed, "ERROR: weights did not change — optimizer/backward pipeline is broken!"

    # Run a few more steps and show the loss trending down (learning is happening).
    print("\n5. Running 20 more steps on the SAME batch to watch loss decrease:")
    for step in range(20):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == 19:
            print(f"   step {step:2d}: loss = {loss.item():.4f}")
    print("   -> Loss dropping confirms the model is memorizing this tiny batch,")
    print("      which is exactly the 'overfit test' idea we'll use later (Gate #5).")


if __name__ == "__main__":
    demo_tensor_shapes()
    demo_nn_module()
    demo_dataset_and_dataloader()
    demo_training_step()
    print("\n" + "=" * 65)
    print("  ALL 4 PYTORCH PILLARS SUCCESSFULLY DEMONSTRATED AND VERIFIED!")
    print("=" * 65 + "\n")

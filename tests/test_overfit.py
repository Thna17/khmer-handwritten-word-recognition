"""
Gate #5: the tiny overfit test.

Before ever training on the full dataset, the model must be able to
memorize a TINY set of samples (10-30) almost perfectly. If it can't,
there's a bug somewhere in the pipeline (tokenizer, dataset, CTC lengths,
model shapes) that would otherwise waste hours of real training time
before being noticed. See PROJECT_CONTEXT.md / TASKS.md Phase 8.

Real handwriting doesn't exist yet, so this uses SYNTHETIC PLACEHOLDER
images (same rendering helper as test_dataset.py / test_model_baseline.py)
purely to prove the pipeline itself is correct end-to-end. This is NOT a
claim that the model works on real handwriting — only that nothing in the
tokenizer/dataset/model/loss wiring is broken.

Run:
    .venv/bin/pytest tests/test_overfit.py -v -s
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import KhmerWordDataset, collate_fn, load_metadata_csv
from src.decoder import greedy_decode
from src.metrics import evaluate_predictions
from src.model_baseline import BaselineCRNN
from src.tokenizer import KhmerTokenizer
from tests.test_dataset import WORDS, WRITER_IDS, _render_word_image

TINY_SUBSET_SIZE = 20
MAX_EPOCHS = 300
TARGET_CONSECUTIVE_PERFECT_EPOCHS = 5  # stop early once memorization is stable, not just lucky once


def _build_tiny_dataset(tmp_path, tokenizer):
    root = tmp_path / "overfit_synthetic"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    subset_words = WORDS[:TINY_SUBSET_SIZE]
    rows = []
    for i, word in enumerate(subset_words):
        font_size = 30 + (i % 4) * 5
        filename = f"{i:04d}.png"
        _render_word_image(word, font_size, images_dir / filename)
        rows.append({"image": filename, "label": word, "writer_id": WRITER_IDS[i % len(WRITER_IDS)]})

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)

    samples = load_metadata_csv(csv_path, images_dir)
    return KhmerWordDataset(samples, tokenizer, image_height=48, max_width=256)


def test_tiny_overfit_gate_5(tmp_path):
    torch.manual_seed(0)

    # Vocabulary built from the FULL word list (not just this tiny subset),
    # so the model's output layer is the same size it will be in real
    # training -- overfitting 20 words with a 30-word-sized head is a
    # more honest test than artificially shrinking the vocabulary.
    tokenizer = KhmerTokenizer.build_from_labels(WORDS)
    dataset = _build_tiny_dataset(tmp_path, tokenizer)
    assert len(dataset) == TINY_SUBSET_SIZE

    loader = DataLoader(dataset, batch_size=TINY_SUBSET_SIZE, shuffle=True, collate_fn=collate_fn)

    model = BaselineCRNN(num_classes=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    loss_history = []
    word_acc_history = []
    consecutive_perfect_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch = next(iter(loader))  # one full-batch "epoch" since the set is tiny

        log_probs = model(batch["images"])
        input_lengths = torch.tensor(
            [model.compute_sequence_length(w.item()) for w in batch["image_widths"]],
            dtype=torch.long,
        )
        target_lengths = batch["target_lengths"]

        loss = criterion(log_probs, batch["targets"], input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Track memorization progress every few epochs (decoding every
        # single epoch would slow the test down for no benefit).
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                eval_log_probs = model(batch["images"])
                eval_input_lengths = input_lengths  # same batch, same widths
                predictions = greedy_decode(eval_log_probs, tokenizer, input_lengths=eval_input_lengths)
            metrics = evaluate_predictions(batch["labels"], predictions)
            word_acc_history.append((epoch, metrics["word_accuracy"]))
            print(f"epoch {epoch:3d} | loss={loss.item():.4f} | CER={metrics['cer']*100:.1f}% | WordAcc={metrics['word_accuracy']*100:.1f}%")

            if metrics["word_accuracy"] == 1.0:
                consecutive_perfect_epochs += 1
                if consecutive_perfect_epochs >= TARGET_CONSECUTIVE_PERFECT_EPOCHS:
                    print(f"Stopping early at epoch {epoch}: perfect memorization held for {TARGET_CONSECUTIVE_PERFECT_EPOCHS} checks in a row.")
                    break
            else:
                consecutive_perfect_epochs = 0

    # Final evaluation pass on the same tiny training set (this IS the point
    # of an overfit test -- can the model memorize what it was shown?).
    model.eval()
    with torch.no_grad():
        final_batch = next(iter(DataLoader(dataset, batch_size=TINY_SUBSET_SIZE, shuffle=False, collate_fn=collate_fn)))
        final_log_probs = model(final_batch["images"])
        final_input_lengths = torch.tensor(
            [model.compute_sequence_length(w.item()) for w in final_batch["image_widths"]],
            dtype=torch.long,
        )
        final_predictions = greedy_decode(final_log_probs, tokenizer, input_lengths=final_input_lengths)
    final_metrics = evaluate_predictions(final_batch["labels"], final_predictions)

    print(f"\nFinal: loss={loss_history[-1]:.4f} | CER={final_metrics['cer']*100:.2f}% | WordAcc={final_metrics['word_accuracy']*100:.2f}%")
    print("Sample predictions vs targets:")
    for target, pred in list(zip(final_batch["labels"], final_predictions))[:8]:
        marker = "OK" if target == pred else "WRONG"
        print(f"  {target} -> {pred}  [{marker}]")

    # Save the loss curve as evidence, matching the project's "show loss
    # curves" requirement, even for this small-scale sanity check.
    import os
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Gate #5: Tiny Overfit Test — Training Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/overfit_loss_curve.png", dpi=120)
    plt.close(fig)

    # --- Gate #5 pass/fail criteria ---
    assert loss_history[-1] < loss_history[0] * 0.05, (
        f"loss did not drop substantially: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}. "
        "This usually means a real bug in the CTC wiring, not just slow convergence."
    )
    assert final_metrics["word_accuracy"] == 1.0, (
        f"model failed to memorize the tiny dataset (word accuracy = "
        f"{final_metrics['word_accuracy']*100:.1f}%). Do NOT proceed to full training "
        "until this passes -- see TASKS.md Phase 8 debugging order."
    )
    assert final_metrics["cer"] == 0.0

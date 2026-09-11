# EXPERIMENTS LOG

This document tracks every controlled experiment conducted in this project, across all three
approaches (see [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) Section 2: A1 baseline CRNN, A2 transfer
learning backbone, A3 Transformer sequence encoder).

**Rule:** Only change one primary variable at a time (e.g. learning rate, augmentation, hidden
dimension) to ensure scientific rigor. All approaches are evaluated on the identical, writer-disjoint
test split with identical preprocessing — this table is the source of truth for the final comparison.

---

## Experiment Summary Table

| Experiment ID | Date | Approach | Description | Parameter Changes | Val Loss | CER (%) | Word Acc (%) | WER (%) | Params | Train Time | Notes |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `E000` | 2026-09-07 | A1 | Tiny Overfit Verification | 10–20 samples, 100 epochs, AdamW lr=1e-3 | - | - | - | - | - | - | Verification of pipeline correctness (Gate #5) |
| `E001` | Pending | A1 | Baseline CRNN, full dataset | No augmentation, lr=1e-3 | - | - | - | - | - | - | Initial benchmark on unseen writers |
| `E002` | Pending | A1 | + Data Augmentation | Rotation ±3–5°, slight scaling | - | - | - | - | - | - | Test generalization against overfitting |
| `E003` | Pending | A1 | Learning Rate Tuning | lr=5e-4 vs 1e-3 | - | - | - | - | - | - | Assess convergence stability |
| `E010` | Pending | A2 | Transfer learning, frozen backbone | ResNet-18 (ImageNet), backbone frozen | - | - | - | - | - | - | Linear-probe style: only BiLSTM+head trained |
| `E011` | Pending | A2 | Transfer learning, fine-tuned | ResNet-18 (ImageNet), backbone unfrozen | - | - | - | - | - | - | Compare against E010 to see if fine-tuning helps |
| `E020` | Pending | A3 | Transformer encoder, full dataset | Same CNN as A1, BiLSTM replaced by Transformer encoder | - | - | - | - | - | - | Isolates sequence-model architecture choice |
| `E021` | Pending | A3 | Hyperparameter tuning | (fill in once A3 baseline run exists) | - | - | - | - | - | - | Required tuning pass on best-performing approach |

---

## Detailed Experiment Reports

### E000: Tiny Overfit Verification (A1)
- **Date:** 2026-09-07
- **Dataset:** 10–20 sample images
- **Model:** Baseline CRNN (VGG CNN + 2-layer BiLSTM 256)
- **Objective:** Verify loss decreases monotonically towards zero and the model reaches 100% Word Accuracy (Gate #5).
- **Outcome:** Pending implementation.

<!--
Add one dated report per experiment ID below, in the same format as E000, as soon as it actually runs.
Never fill in numbers before the run has happened — fabricated results are an academic integrity
violation per the course rubric.
-->

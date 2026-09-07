# EXPERIMENTS LOG

This document tracks every controlled experiment conducted in this project.
**Rule:** Only change one primary variable at a time (e.g. learning rate, augmentation, hidden dimension) to ensure scientific rigor.

---

## Experiment Summary Table

| Experiment ID | Date | Description | Parameter Changes | Val Loss | CER (%) | Word Acc (%) | Notes |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| `E000` | 2026-09-07 | Tiny Overfit Verification | 10–20 samples, 100 epochs, AdamW lr=1e-3 | - | - | - | Verification of pipeline correctness |
| `E001` | Pending | Baseline CRNN Model | Full dataset, no augmentation, lr=1e-3 | - | - | - | Initial benchmark on unseen writers |
| `E002` | Pending | + Data Augmentation | Rotation ±3–5°, slight scaling | - | - | - | Test generalization against overfitting |
| `E003` | Pending | Learning Rate Tuning | lr=5e-4 vs 1e-3 | - | - | - | Assess convergence stability |

---

## Detailed Experiment Reports

### E000: Tiny Overfit Verification
- **Date:** 2026-09-07
- **Dataset:** 10–20 sample images
- **Model:** Baseline CRNN (VGG CNN + 2-layer BiLSTM 256)
- **Objective:** Verify loss decreases monotonically towards zero and the model reaches 100% Word Accuracy (Gate #5).
- **Outcome:** Pending implementation.


# Checkpoints Directory

This directory stores trained PyTorch model weights (`.pt`) for the Khmer Handwritten Text Recognition system.

## 📌 Standard Checkpoint Filenames

The system and web demo (`app.py`) automatically look for weights matching these filenames:

| Approach | Stage 1 (Synthetic Pre-Trained) | Stage 2 (Real Ink Fine-Tuned) |
| :--- | :--- | :--- |
| **A1: Baseline CRNN (VGG + BiLSTM)** | `a1_baseline_synthetic_best.pt` | `a1_baseline_finetuned_best.pt` |
| **A2: Transfer Learning (ResNet-18)** | `a2_transfer_synthetic_best.pt` | `a2_transfer_finetuned_best.pt` |
| **A3: Transformer Sequence Encoder** | `a3_transformer_synthetic_best.pt` | `a3_transformer_finetuned_best.pt` |

## 🚀 Obtaining Pretrained Weights

- In local development, place downloaded `.pt` files directly into this directory (`checkpoints/`).
- If training in Google Colab, mount Google Drive or point `--checkpoint-dir` to your Drive directory.
- Model weights are saved as standard PyTorch state dicts:
  ```python
  import torch
  from src.model_baseline import BaselineCRNN

  checkpoint = torch.load("checkpoints/a1_baseline_finetuned_best.pt", map_location="cpu")
  model = BaselineCRNN(num_classes=87)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  ```

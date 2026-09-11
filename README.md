# Khmer Handwritten Short-Word Recognition: Comparing Three CRNN-Based Deep Learning Approaches

**Author:** Hong Than Brathna
**Course:** Deep Learning — Final Project (Bachelor of Software Engineering)

A Deep Learning system that recognizes single handwritten Khmer short words from images and
transcribes them into standard Khmer Unicode text — implemented as **three distinct approaches**,
fairly compared on the same held-out, writer-disjoint test set.

> Status: project scaffolding and foundational PyTorch/CNN practice complete. Dataset collection and
> the three approaches are in progress — see [TASKS.md](TASKS.md) for the current week's tasks and
> [EXPERIMENTS.md](EXPERIMENTS.md) for run-by-run results as they land. This README's results section
> will be filled in as real numbers exist; no numbers are fabricated ahead of time.

---

## Problem Statement

**Task:** Given an image containing exactly one handwritten Khmer short word, predict the Khmer
Unicode text it represents. This is a **sequence labeling / OCR** problem: the model outputs a
variable-length sequence of characters aligned (via CTC) to a fixed-length sequence of image
features, without needing pre-segmented per-character images.

**Why it matters:** Khmer handwriting recognition has very little public data or tooling compared to
Latin-script OCR. A working pipeline — dataset, tokenizer, evaluation harness — is itself a
contribution, independent of which model wins the comparison.

---

## Dataset

- **Source:** Self-collected. Real handwriting samples gathered from multiple volunteer writers, each
  assigned a `writer_id` (W001, W002, ...).
- **Format:** `data/metadata/*.csv` with columns `image,label,writer_id`.
- **Size / vocabulary:** documented here once collection is complete (see [TASKS.md](TASKS.md) Week 1)
  — reported honestly, including any known bias or limitation (e.g. handwriting style skew, uneven
  samples per writer).
- **Split:** writer-disjoint train/validation/test — a writer that appears in training **never**
  appears in validation or test. This is the same fixed split used, unchanged, by all three approaches.
- **Preprocessing:** grayscale, resized to height 48px preserving aspect ratio (no stretching), max
  width ≈256px, dynamically padded to the batch's max width. Identical preprocessing is applied at
  test time for every approach.

---

## Approaches Compared

All three approaches share the same dataset pipeline, Khmer tokenizer, CTC greedy decoder, and
evaluation code (CER / WER / Word Accuracy) — only the model (and, for A2, the training strategy)
differs. Full architecture diagrams are in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#2-the-three-approaches-share-dataset-tokenizer-decoder-metrics-eval-harness).

| # | Approach | CNN backbone | Sequence model | Training strategy | Distinctness dimension |
|---|---|---|---|---|---|
| A1 | Baseline CRNN | VGG-style, from scratch | 2-layer BiLSTM (256) | Trained from scratch | — (baseline) |
| A2 | Transfer learning | ResNet-18 (ImageNet-pretrained) | Same BiLSTM as A1 | Frozen backbone vs. fine-tuned | (B) training strategy |
| A3 | Transformer encoder | Same VGG-style CNN as A1 | Small Transformer encoder | Trained from scratch | (A) architecture |

---

## Results Summary

_To be filled in once all three approaches have been trained and evaluated on the identical test set
(Week 4 of [TASKS.md](TASKS.md)). Will include: CER, WER, Word Accuracy, trainable parameter count,
training time, and hardware, side by side for A1/A2/A3, plus learning-curve and metric-comparison
figures in `results/`._

---

## Repository Structure

```text
khmer-handwritten-word-recognition/
├── data/
│   ├── raw/                 # Original scanned / photographed handwriting images (gitignored)
│   ├── processed/           # Resized & normalized images (gitignored)
│   └── metadata/            # CSV files containing image,label,writer_id
├── notebooks/               # Educational notebooks and Google Colab runners
├── src/
│   ├── tokenizer.py         # Khmer Unicode NFC tokenizer and vocabulary manager
│   ├── dataset.py           # PyTorch Dataset, aspect-ratio resize, pad collate_fn
│   ├── model_baseline.py    # A1: VGG CNN (from scratch) + BiLSTM CRNN
│   ├── model_transfer.py    # A2: pretrained CNN backbone + BiLSTM CRNN
│   ├── model_transformer.py # A3: VGG CNN (from scratch) + Transformer encoder
│   ├── decoder.py           # CTC greedy decoding (shared)
│   ├── metrics.py           # CER, WER, Word Accuracy (shared)
│   ├── train.py             # Training loop with AdamW and checkpointing (shared, model-agnostic)
│   ├── evaluate.py          # Unseen-writer evaluation script (shared)
│   └── utils.py             # Seed-fixing, checkpointing, logging helpers
├── configs/                 # YAML configuration files per approach/experiment
├── checkpoints/             # Saved model checkpoints (best_model.pt, latest_model.pt per approach)
├── results/                 # Evaluation logs, comparison table, figures, error analysis
├── slides/                  # Final presentation deck (PDF/PPTX)
├── tests/                   # Educational demo scripts + tiny overfit verification
├── PROJECT_CONTEXT.md       # Locked scope, 3-approach architecture, course rubric requirements
├── TASKS.md                 # Week-by-week task tracking
├── EXPERIMENTS.md           # Controlled experiment tracking table (all 3 approaches)
├── requirements.txt         # Exact project dependencies
└── README.md                # This file
```

---

## How to Install and Run

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Foundational Demo Scripts
```bash
.venv/bin/python tests/demo_day1_concepts.py        # CNN role, BiLSTM, CTC, writer-disjoint split
.venv/bin/python tests/demo_day2_pytorch_basics.py  # Tensor/nn.Module/Dataset/DataLoader/training loop
.venv/bin/python tests/demo_day3_mnist_cnn.py        # Real end-to-end CNN training (Gate #1)
.venv/bin/python tests/demo_day4_cnn_internals.py    # Convolution/ReLU/pooling/receptive field
```

### 3. Run the Test Suite
```bash
.venv/bin/pytest tests/ -q --ignore=tests/test_overfit.py   # fast (~20s)
.venv/bin/pytest tests/test_overfit.py -v -s                # Gate #5, ~4-5 minutes (actually trains)
```

### 4. Train Any of the 3 Approaches
`src/run_experiment.py` wires dataset + tokenizer + writer-disjoint split + model + training +
test-set evaluation together for whichever approach you choose. Every path is a CLI argument, so
the same command works locally (small/synthetic data) or in Colab against Google Drive paths.
```bash
.venv/bin/python -m src.run_experiment \
  --approach baseline \
  --metadata-csv data/metadata/labels.csv --images-dir data/raw \
  --checkpoint-dir checkpoints --results-dir results \
  --vocab-path data/metadata/char_to_idx.json --split-path data/metadata/writer_split.json \
  --max-epochs 100
# --approach transfer [--freeze-backbone]   for A2
# --approach transformer                    for A3
```
The first run creates the tokenizer vocabulary and the writer-disjoint split and saves both;
every later run of any approach reuses them automatically — this is what guarantees A1/A2/A3 are
compared on the identical test set.

### 5. Train on Google Colab (recommended if local disk/RAM is limited)
Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Colab. It mounts Google Drive,
clones this repo fresh into the Colab session, and runs all three approaches with data/checkpoints/
results stored entirely on Drive — nothing large ever touches local disk. Upload your dataset to
Drive first, at `MyDrive/khmer-ocr/data/metadata/labels.csv` and `MyDrive/khmer-ocr/data/raw/`
(paths are editable in the notebook). **Requires this repo's latest work to be pushed to GitHub
first**, since the notebook clones from the remote.

---

## Key Rules & Guidelines
- **Writer Disjoint Splitting:** Test writers must never appear in the training set. Enforced for all 3 approaches via one shared split.
- **Khmer Unicode NFC:** Labels are strictly normalized using Unicode NFC to ensure consistent character encoding.
- **Never Stretch Images:** All images maintain their proportional aspect ratio when resized to height 48, with remaining width padded to batch maximum.
- **Same test set, same preprocessing, for every approach** — required for a fair comparison.

---

## Citations
_Any reused code or pretrained weights (e.g. ImageNet-pretrained ResNet-18 for Approach 2, from
`torchvision.models`) will be cited here with source and license as they are integrated._

---

## AI Assistance Disclosure
- **Tools Used:** Claude (Claude Code).
- **Scope of Use:** Pair-programming assistant throughout the project — drafting PyTorch/dataset/model
  code, debugging tensor-shape and CTC issues, writing educational demo scripts to build the author's
  own understanding of CNN/BiLSTM/CTC internals, and structuring project documentation
  (`PROJECT_CONTEXT.md`, `TASKS.md`, `EXPERIMENTS.md`, this README).
- **Verification:** Every script referenced above was executed locally and its output inspected before
  being accepted (e.g. Gate #1's 98.29% MNIST test accuracy, Gate #4's tokenizer round-trip test, the
  receptive-field autograd check in Day 4). No results in this repository are reported without having
  actually been run. The author can explain and modify every line of code in this repository.

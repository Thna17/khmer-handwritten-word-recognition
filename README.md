# Khmer Handwritten Short-Word Recognition Using CRNN with CTC Loss

A Deep Learning system designed to recognize single handwritten Khmer short words from images and transcribe them into standard Khmer Unicode text.

---

## Architecture Overview

```
Input Image (Grayscale: 1 × 48 × W)
  │
  ▼
[VGG-style CNN]              ── Extracts hierarchical visual features (strokes, loops, sub-consonants)
  │
  ▼
[Spatial-to-Sequence]        ── Squeezes height to 1 and outputs horizontal time slices [T, B, C]
  │
  ▼
[2-Layer BiLSTM]             ── Scans time steps left-to-right and right-to-left for context
  │
  ▼
[Linear + LogSoftmax]        ── Maps recurrent features to Khmer character probabilities
  │
  ▼
[CTC Loss / Greedy Decoder]  ── Resolves alignments, collapses duplicates, outputs Unicode text
```

---

## Repository Structure

```text
khmer-handwritten-word-recognition/
├── data/
│   ├── raw/                 # Original scanned / photographed handwriting images
│   ├── processed/           # Resized & normalized images
│   └── metadata/            # CSV files containing image,label,writer_id
├── notebooks/               # Educational notebooks and Google Colab runners
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset, aspect-ratio resize, pad collate_fn
│   ├── tokenizer.py         # Khmer Unicode NFC tokenizer and vocabulary manager
│   ├── model.py             # VGG + BiLSTM CRNN architecture
│   ├── decoder.py           # CTC greedy decoding
│   ├── metrics.py           # CER, WER, and Word Accuracy calculators
│   ├── train.py             # Modular training loop with AdamW and checkpointing
│   ├── evaluate.py          # Unseen writer evaluation script
│   └── utils.py             # Checkpointing, logging, and visualization helpers
├── configs/                 # YAML configuration files for experiments
├── checkpoints/             # Saved model checkpoints (best_model.pt, latest_model.pt)
├── results/                 # Evaluation logs, error analysis, and prediction CSVs
├── tests/                   # Automated unit tests and tiny overfit verification
├── PROJECT_CONTEXT.md       # Locked architecture, scope, and non-negotiable rules
├── TASKS.md                 # Daily roadmap and task tracking
├── EXPERIMENTS.md           # Controlled experiment tracking table
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Quickstart

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Run Tiny Overfit Verification
```bash
python -m tests.test_overfit
```

---

## Key Rules & Guidelines
- **Writer Disjoint Splitting:** Test writers must never appear in the training set.
- **Khmer Unicode NFC:** Labels are strictly normalized using Unicode NFC to ensure consistent character encoding.
- **Never Stretch Images:** All images maintain their proportional aspect ratio when resized to height 48, with remaining width padded to batch maximum.

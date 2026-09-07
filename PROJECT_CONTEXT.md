# PROJECT CONTEXT

## Title
**Khmer Handwritten Short-Word Recognition Using CRNN with CTC Loss**

## Objective
Build an accurate, reproducible, and beginner-friendly Deep Learning OCR model that receives an image containing one handwritten Khmer short word and outputs the corresponding Khmer Unicode text.

---

## 1. Locked Scope
- **Domain:** Handwritten Khmer short words.
- **Image content:** Exactly one short word per image (isolated word recognition).
- **Vocabulary size:** 30–50 selected Khmer words for the final experiment.
- **Dataset size target:** ~4,000–6,000 real samples across 20–25+ human writers (4–5 repetitions per word).
- **Out of scope:**
  - No sentence recognition.
  - No paragraph OCR.
  - No text detection or document layout analysis.
  - No Transformer decoder, TrOCR, or large pretrained models unless explicitly added as late-stage experiments.

---

## 2. Locked Baseline Architecture
```text
Input Image (Grayscale [B, 1, 48, W])
  ↓
VGG-style CNN feature extractor
  ↓
Spatial-to-sequence transformation [W', B, features]
  ↓
2-layer Bidirectional LSTM (hidden_size=256, dropout=0.2)
  ↓
Linear classification layer (num_classes = len(vocab) + 1 for CTC blank)
  ↓
LogSoftmax
  ↓
CTC Loss (torch.nn.CTCLoss(blank=0, zero_infinity=True))
  ↓
CTC Greedy Decoding
  ↓
Khmer Unicode Text
```

---

## 3. Technology Stack
- **Languages & Frameworks:** Python 3.11+, PyTorch (>= 2.0).
- **Image Processing & Math:** Pillow, OpenCV (if needed), NumPy, pandas.
- **Evaluation & Metrics:** jiwer (CER, WER), scikit-learn, matplotlib.
- **Development & Training:** Local development in VS Code; GPU training in Google Colab; version control via Git/GitHub.

---

## 4. Data Split & Writer Integrity Rules
- **Data format:**
  ```csv
  image,label,writer_id
  000001.png,សាលា,W001
  000002.png,ខ្មែរ,W001
  000003.png,សាលា,W002
  ```
- **Strict writer-disjoint splitting:**
  - Writers in `train_writers` must NEVER appear in `val_writers` or `test_writers`.
  - Zero writer leakage ensures true evaluation of generalization to unseen handwriting styles.
  - `writer_id` must NEVER be discarded.

---

## 5. Khmer Unicode Handling Rules
1. Apply consistent Unicode normalization (Unicode NFC standard: `unicodedata.normalize('NFC', text)`).
2. Deterministic vocabulary generation: Character-to-index mapping with index `0` strictly reserved for `<blank>`.
3. Save mapping files (`char_to_idx.json`, `idx_to_char.json`) to disk.
4. Mandatory round-trip test:
   `original Khmer → encode → token IDs → decode → exact original Khmer`.
5. Preserve original spelling and verify actual code-point sequences (base consonants, subscript `U+17D2`, vowels, diacritics).

---

## 6. Preprocessing & Input Constraints
- **Grayscale (1 channel)**.
- **Height = 48 pixels** (fixed).
- **Maximum width ≈ 256 pixels**.
- **Preserve aspect ratio:** Resize proportionally to height 48, clamp max width to 256, and dynamically pad to batch maximum width with white background (value 1.0 or 255).
- **No horizontal stretching:** Never force all images to fixed width by squishing or stretching.

---

## 7. Priority Order
```text
Phase 0: Lock Project & Understand Flow
Phase 1: Deep Learning Foundations & MNIST Practice
Phase 2: Tiny Khmer Character CNN Experiment
Phase 3: Dataset Design & Collection (Strict Writer IDs)
Phase 4: Sequence Modeling Foundations (RNN, LSTM, BiLSTM)
Phase 5: CTC Theory & Alignment Understanding
Phase 6: Dataset Pipeline & Khmer Tokenizer (Round-trip verified)
Phase 7: Implement CRNN Architecture
Phase 8: Tiny Overfit Verification (10–30 samples to 0% CER)
Phase 9: Scaling Training
Phase 10: Baseline Evaluation (CER, Word Accuracy, WER)
Phase 11: Controlled Experiments (documented in EXPERIMENTS.md)
Phase 12: Error Analysis
Phase 13: Lightweight Interactive Demo (Gradio/Streamlit)
Phase 14: Project Freeze & Academic Report
```

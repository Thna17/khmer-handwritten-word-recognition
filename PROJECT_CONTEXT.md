# PROJECT CONTEXT

## Title
**Khmer Handwritten Short-Word Recognition: A Comparison of Three CRNN-Based Deep Learning Approaches**

## Course context
This is the individual Final Project for a Deep Learning course (Bachelor of Software Engineering,
lecturer Mr. Soklong HIM). **Topic has been verbally approved.** Full grading rubric and rules are
summarized in [0. Course Requirements](#0-course-requirements) below — that section governs scope
decisions whenever it conflicts with earlier, single-model planning.

## Objective
Build an accurate, reproducible, and beginner-friendly Deep Learning OCR system that receives an
image containing one handwritten Khmer short word and outputs the corresponding Khmer Unicode text
— implemented as **three distinct deep learning approaches**, fairly compared on the same held-out
test set, with an evidence-based discussion of which one works best and why.

---

## 0. Course Requirements (drives every scope decision below)
Source: official Final Project Instruction and Evaluation Rubric.

- **Minimum 2, target 3 distinct approaches.** A single model cannot pass the Technical & Intellectual
  Depth criterion (50% of the grade). Distinctness must come from a different **architecture** and/or
  a different **training strategy** — not just a different learning rate/augmentation/optimizer.
- **Same test set for every approach.** One fixed, writer-disjoint train/val/test split, used unchanged
  by all three approaches, with identical test-time preprocessing.
- **PyTorch only.** No Keras/TensorFlow.
- **Reproducibility.** Fix seeds (`torch.manual_seed`, NumPy, Python `random`) for every run.
- **Checkpointing.** `torch.save`/load for every approach, to survive Colab disconnects.
- **Hyperparameter tuning** (at least learning rate + one regularization choice) documented for at
  least the best-performing approach.
- **Per-approach reporting:** trainable parameter count, training time, hardware used.
- **Evaluation:** CER, Word Accuracy, WER for every approach on the identical test set; training/val
  curves for every approach; error analysis (misclassified examples + likely causes); limitations.
- **Comparison:** one results table with all approaches side by side + at least 2 comparison figures
  (e.g. bar chart of CER/WordAcc, overlaid learning curves).
- **Deliverables:** GitHub repo (`README.md`, `requirements.txt`, code per approach, `results/`,
  `slides/`), regular commit history, 10–20 slide deck (PDF/PPTX) for a 10-min talk + 5-min Q&A.
- **AI-use disclosure required in README** (tool, scope of use, verification) — see
  [README.md](README.md) once populated. Every line of code must be explainable/modifiable by the
  student during Q&A — this project's working style (AI drafts, student reviews, everything is run
  and tested before being trusted) supports that.
- **Timeline** (course weeks; already-approved topic means we skip straight to building):
  Progress check = show dataset split + at least one trained approach's results.
  Final submission = repo + slides. Presentation = 10 min + 5 min Q&A.
  This project has **~4 weeks remaining** — see [TASKS.md](TASKS.md) for the week-by-week plan.

---

## 1. Locked Scope
- **Domain:** Handwritten Khmer short words.
- **Image content:** Exactly one short word per image (isolated word recognition).
- **Vocabulary size:** 20–50 selected Khmer words (exact number driven by how much real data can be
  collected in the time available — see Risk note below).
- **Dataset size target:** as many real samples as time allows across 15–25 human writers, writer-disjoint
  train/val/test split; describe the final count and any limitations honestly in the README (the rubric
  explicitly allows and expects an honest, imperfect self-collected dataset over a fabricated large one).
- **Out of scope:** sentence recognition, paragraph OCR, text detection/document layout analysis,
  large pretrained OCR systems (TrOCR, etc.) as a fourth approach — three approaches is the target, not four.
- **Risk note:** dataset collection is the highest-risk item on a 4-week clock. If collection lags,
  shrink the vocabulary (not the writer-disjoint rule, not the honesty of reporting) before the deadline
  forces a rushed, undocumented dataset.

---

## 2. The Three Approaches (share dataset, tokenizer, decoder, metrics, eval harness)

All three consume the identical preprocessed `[B, 1, 48, W]` grayscale tensors, the identical Khmer
tokenizer/vocabulary, the identical CTC greedy decoder, and are scored with the identical metrics code
on the identical held-out test writers. Only the model definition (and, for A2, the training strategy)
differs — this maximizes correctness/reuse while satisfying the rubric's distinctness rules on two
separate axes.

### Approach 1 (A1) — Baseline CRNN, trained from scratch
```text
Input Image (Grayscale [B, 1, 48, W])
  ↓
VGG-style CNN feature extractor (trained from scratch)
  ↓
Spatial-to-sequence transformation [W', B, features]
  ↓
2-layer Bidirectional LSTM (hidden_size=256, dropout=0.2)
  ↓
Linear classification layer (num_classes = len(vocab) + 1 for CTC blank)
  ↓
LogSoftmax → CTC Loss → CTC Greedy Decoding → Khmer Unicode Text
```
This is the original locked baseline architecture — unchanged.

### Approach 2 (A2) — Transfer learning backbone
```text
Input Image (Grayscale [B, 1, 48, W], replicated to 3ch for the pretrained net)
  ↓
Pretrained CNN backbone (e.g. ResNet-18, ImageNet weights, truncated to keep enough
horizontal resolution for CTC) — compared frozen (linear probe) vs. fine-tuned
  ↓
Spatial-to-sequence transformation [W', B, features]
  ↓
SAME 2-layer Bidirectional LSTM (hidden_size=256, dropout=0.2) — architecture held constant
  ↓
SAME Linear + LogSoftmax → CTC Loss → CTC Greedy Decoding → Khmer Unicode Text
```
Distinctness dimension: **(B) training strategy** — transfer learning vs. training from scratch —
and partly **(A) architecture** (ResNet vs. custom VGG-style backbone).

### Approach 3 (A3) — Transformer sequence encoder
```text
Input Image (Grayscale [B, 1, 48, W])
  ↓
SAME VGG-style CNN feature extractor as A1 (trained from scratch) — held constant
  ↓
Spatial-to-sequence transformation [W', B, features] + positional encoding
  ↓
Small Transformer encoder (a few self-attention layers) REPLACING the BiLSTM
  ↓
SAME Linear + LogSoftmax → CTC Loss → CTC Greedy Decoding → Khmer Unicode Text
```
Distinctness dimension: **(A) architecture** — BiLSTM vs. Transformer encoder (an explicitly
rubric-approved example pairing), CNN and CTC head held constant so the comparison isolates the
sequence-modeling choice.

Do not add a fourth approach or swap in attention/seq2seq decoding beyond this — three well-executed,
fairly-compared approaches score higher than four rushed ones.

---

## 3. Technology Stack
- **Languages & Frameworks:** Python 3.11+, PyTorch (>= 2.0), torchvision (for the A2 pretrained backbone only).
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
Phase 0: Lock Project & Understand Flow                          [DONE]
Phase 1: Deep Learning Foundations & MNIST Practice               [DONE]
Phase 2: Tiny Khmer Character CNN Experiment (Gate #2)
Phase 3: Dataset Design & Collection (Strict Writer IDs)
Phase 4: Sequence Modeling Foundations (RNN, LSTM, BiLSTM)
Phase 5: CTC Theory & Alignment Understanding
Phase 6: Dataset Pipeline & Khmer Tokenizer (Round-trip verified) — SHARED by all 3 approaches
Phase 7: Implement A1 — Baseline CRNN (VGG CNN + BiLSTM + CTC)
Phase 8: Tiny Overfit Verification on A1 (10–30 samples to 0% CER, Gate #5)
Phase 9: Scale A1 Training on Full Dataset + Baseline Evaluation (CER, WordAcc, WER)
Phase 10: Implement & Train A2 — Transfer Learning Backbone (frozen + fine-tuned)
Phase 11: Implement & Train A3 — Transformer Sequence Encoder
Phase 12: Hyperparameter Tuning (best approach) + Controlled Experiments (EXPERIMENTS.md)
Phase 13: Cross-Approach Comparison (results table, 2+ figures) + Error Analysis
Phase 14: README, Slides, Commit Hygiene, Freeze & Rehearse Presentation
```
See [TASKS.md](TASKS.md) for this mapped onto the actual ~4-week calendar.

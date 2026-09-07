# PROJECT TASKS TRACKER

## Phase 0: Lock Project & Scaffolding (Day 1)
- [x] Create project repository directory structure (`data/`, `notebooks/`, `src/`, `configs/`, `checkpoints/`, `results/`, `tests/`)
- [x] Create `PROJECT_CONTEXT.md` with locked scope, architecture, and rules
- [x] Create `TASKS.md`, `EXPERIMENTS.md`, `README.md`, and `requirements.txt`
- [x] Set up Python 3.11 virtual environment and install dependencies
- [ ] Understand conceptual flow: **Input → CNN → BiLSTM → CTC → Output**

---

## Phase 1: Deep Learning Foundations (Days 2–4)
- [ ] Day 2: Master PyTorch basics (`Tensor`, `nn.Module`, `Dataset`, `DataLoader`, `optimizer`, `loss.backward()`)
- [ ] Day 3: Train an MNIST digit classifier CNN end-to-end (**Gate #1**)
- [ ] Day 4: Deep dive into CNNs (Convolutions, ReLUs, Pooling, Receptive Fields)

---

## Phase 2: Tiny Khmer Character Experiment (Days 5–7)
- [ ] Prepare 5 Khmer characters (20–50 samples each)
- [ ] Train a lightweight CNN character classifier (**Gate #2**)
- [ ] Verify image loading, preprocessing, tensor conversion, and training loops

---

## Phase 3: Dataset Design & Collection (Days 5–10)
- [ ] Define target vocabulary: 30–50 common Khmer short words
- [ ] Design collection sheets with explicit `writer_id` tracking (e.g. W001–W025)
- [ ] Collect 4,000–6,000 clean real samples (4–5 repetitions per word per writer)
- [ ] Build metadata CSV (`image,label,writer_id`)
- [ ] Implement and verify writer-disjoint splitting (Train: W001–W018, Val: W019–W021, Test: W022–W025)

---

## Phase 4: Sequence Modeling Foundations (Days 8–10)
- [ ] Understand sequence processing (RNN limitations vs LSTM memory cells)
- [ ] Understand Bidirectional LSTM (scanning left-to-right and right-to-left)

---

## Phase 5: CTC Foundations (Days 10–11)
- [ ] Master CTC concepts: blank token `<blank>`, collapsed repeated tokens, time-step to character alignment
- [ ] Verify Gate #3: Explain why CTC is needed for word-level OCR

---

## Phase 6: Dataset Pipeline & Khmer Tokenizer (Days 11–14)
- [ ] Implement `src/tokenizer.py` with Unicode NFC normalization
- [ ] Implement deterministic vocabulary mapping (`char_to_idx.json`, `idx_to_char.json`, blank=0)
- [ ] Verify Gate #4: Round-trip test (`original -> encode -> decode -> original`)
- [ ] Implement `src/dataset.py` with aspect-ratio preserving resize (H=48, max W=256) and dynamic batch padding
- [ ] Implement dataset integrity validator script (`check_data.py`)

---

## Phase 7: Implement CRNN Architecture (Days 14–17)
- [ ] Implement `src/model.py` (VGG CNN backbone + BiLSTM + Linear classifier)
- [ ] Implement `src/decoder.py` (CTC greedy decoder)
- [ ] Verify forward pass tensor shapes at each stage:
  - Input: `[B, 1, 48, W]`
  - CNN features: `[B, 512, 1, T]`
  - Sequence: `[T, B, 512]`
  - BiLSTM: `[T, B, 512]`
  - LogSoftmax: `[T, B, num_classes]`

---

## Phase 8: Tiny Overfit Test (Day 17)
- [ ] Select 10–30 sample images
- [ ] Train CRNN repeatedly on tiny dataset until loss -> 0 and CER -> 0% (**Gate #5**)
- [ ] Verify CTC alignment, gradients, and tokenizer under memorization

---

## Phase 9: Baseline Training (Days 18–20)
- [ ] Implement `src/train.py` with AdamW, learning rate scheduling, gradient clipping, and checkpointing
- [ ] Train on 10 words, then 20 words, then full 30–50 words on Google Colab GPU
- [ ] Save `best_model.pt` and `latest_model.pt`

---

## Phase 10: Baseline Evaluation (Days 20–23)
- [ ] Implement `src/metrics.py` (CER, WER, Word Accuracy)
- [ ] Evaluate baseline model on unseen test writers
- [ ] Generate prediction log CSV (`image,target,prediction,writer_id,correct`)
- [ ] Plot loss and accuracy curves

---

## Phase 11: Controlled Experiments (Days 23–26)
- [ ] Document all runs in `EXPERIMENTS.md` (E001 Baseline, E002 Data Augmentation, etc.)
- [ ] Test safe augmentations (slight rotation ±3–5°, small scaling, slight blur/contrast)

---

## Phase 12: Error Analysis (Days 26–27)
- [ ] Categorize common model errors (subscripts, vowels, similar consonants, noisy handwriting)
- [ ] Extract visual examples of failures for presentation

---

## Phase 13: Lightweight Interactive Demo (Days 27–28)
- [ ] Build simple web demo (Gradio or Streamlit) for single image upload and prediction

---

## Phase 14: Freeze & Report (Days 28–30)
- [ ] Clean code, freeze final weights, finalize graphs, finish project report and presentation slides

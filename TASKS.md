# PROJECT TASKS TRACKER

Course context: topic already approved. ~4 weeks remain until final submission + presentation.
See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) Section 0 for the rubric requirements driving this plan,
and Section 2 for the three approaches (A1 baseline, A2 transfer learning, A3 Transformer encoder).

Target checkpoint: aim to have A1 fully trained + evaluated on the real dataset by the end of Week 2,
so there is always something real to show if a progress check happens on short notice.

---

## Phase 0: Lock Project & Scaffolding — DONE
- [x] Create project repository directory structure
- [x] Create `PROJECT_CONTEXT.md` with locked scope, architecture, and rules
- [x] Create `TASKS.md`, `EXPERIMENTS.md`, `README.md`, and `requirements.txt`
- [x] Set up Python 3.11 virtual environment and install dependencies
- [x] Understand conceptual flow: **Input → CNN → BiLSTM → CTC → Output**

## Phase 1: Deep Learning Foundations — DONE
- [x] Day 2: PyTorch basics (`Tensor`, `nn.Module`, `Dataset`, `DataLoader`, optimizer, `loss.backward()`) — [tests/demo_day2_pytorch_basics.py](tests/demo_day2_pytorch_basics.py)
- [x] Day 3: MNIST CNN trained end-to-end, 98.29% test accuracy (**Gate #1 passed**) — [tests/demo_day3_mnist_cnn.py](tests/demo_day3_mnist_cnn.py)
- [x] Day 4: CNN internals — convolution, ReLU, pooling, receptive field, verified against real feature maps (**Gate #2 pending, separate from this**) — [tests/demo_day4_cnn_internals.py](tests/demo_day4_cnn_internals.py)
- [x] Also completed: hands-on demo of the 4 CRNN pillars (CNN role, BiLSTM, CTC, writer-disjoint split) — [tests/demo_day1_concepts.py](tests/demo_day1_concepts.py)

---

## WEEK 1 — Dataset foundation + shared pipeline + A1 implementation

### Dataset design & collection (highest-risk item — start immediately, runs in parallel with everything else)
- [x] Draft word vocabulary (30 Khmer short words) — [data/metadata/word_list.csv](data/metadata/word_list.csv) — **pending your final sign-off on spelling before wide distribution**; easy to edit the CSV if any word needs changing
- [x] Collection sheet built and published (writer ID field, consent statement, instructions, one write-in box per word) — printable, links each writer's samples to a Writer ID
- [ ] Recruit writers (target 15–25 people; more writers matters more than more repetitions per writer, since it's the writer-disjoint split that proves generalization)
- [ ] Print/share the collection sheet, assign each writer a Writer ID (W001, W002, ...), collect filled sheets
- [x] Scan-to-crop pipeline built and verified on synthetic sheets — [src/scan_pipeline.py](src/scan_pipeline.py) (`GridLayout` grid calibration from 4 points, `deskew_page` for phone photos, `crop_word_boxes`, `process_sheet`), [tests/test_scan_pipeline.py](tests/test_scan_pipeline.py) 20/20 passed, including an exact geometric proof of the perspective-correction math via a sheared-parallelogram test. **Not yet run on a real scan** — calibration points (4 pixel coordinates) must be measured once against an actual scanned/photographed sheet before this can process real data; automatic corner/grid detection (no manual marking) is a reasonable later improvement once real scans exist to tune it against.
- [ ] Calibrate `GridLayout` against a real scanned sheet (once the first filled sheet comes back) and process it end-to-end
- [ ] Build metadata CSV (`image,label,writer_id`) incrementally as samples come in (via `process_sheet`, one call per scanned page)
- [x] Implement writer-disjoint split (train/val/test by `writer_id`, zero overlap), auto-frozen on first use and reused unchanged by A1/A2/A3 — [src/split.py](src/split.py), [tests/test_split.py](tests/test_split.py) 11/11 passed
- [x] Unified training CLI (`src/run_experiment.py`) wiring dataset+tokenizer+split+model+train+test-evaluation together for any of the 3 approaches, with every path as a CLI argument (works identically locally or in Colab against Drive paths) — [tests/test_run_experiment.py](tests/test_run_experiment.py) 4/4 passed, including proof that a second approach run with a different seed reuses the exact same vocab/split files rather than recomputing them
- [x] Colab + Google Drive training workflow set up — [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) (mount Drive, clone repo, install deps, run sanity tests, train all 3 approaches, compare results) — see note below on the GitHub push this depends on

### Tiny Khmer character experiment (Gate #2 — validates the pipeline before real CRNN work)
- [ ] Prepare ~5 Khmer characters × 20–50 samples (self-written is fine for this proof-of-pipeline step)
- [ ] Train a lightweight CNN classifier on it; confirm image loading, tensors, labels, and training all work

### Khmer tokenizer (shared by all 3 approaches) — DONE
- [x] Implement `src/tokenizer.py`: Unicode NFC normalization + deterministic `char_to_idx`/`idx_to_char` (index 0 = `<blank>`) — [src/tokenizer.py](src/tokenizer.py)
- [x] `save()`/`load()` for vocabulary mapping files, tested round-trip via `tmp_path` (real project vocab files get generated into `data/metadata/` once the word list is locked and/or the real dataset exists)
- [x] Round-trip test: `original Khmer → encode → decode → identical original` (**Gate #4 passed** — 39/39 tests in [tests/test_tokenizer.py](tests/test_tokenizer.py), including the ខ្មែរ = 5-codepoints-not-4-glyphs check and rejection of unknown characters instead of silently dropping them)

### Dataset pipeline (shared) — core implementation DONE, verified with synthetic placeholder images
- [x] Implement `src/dataset.py`: aspect-ratio-preserving resize (H=48, max W≈256), grayscale, dynamic batch padding, safe augmentation hook (`transform=`) — [src/dataset.py](src/dataset.py)
- [x] Verified end-to-end with synthetic (rendered-font, NOT real handwriting) placeholder images in [tests/test_dataset.py](tests/test_dataset.py) — 10/10 passed: preserved aspect ratio (no stretching), per-batch dynamic padding (not a hardcoded 256), writer_id never dropped, tokenizer round-trip through the full Dataset, empty labels raise instead of being silently skipped
- [ ] Implement a small integrity-check script (empty labels, duplicate files, missing writer IDs, unsupported characters) — do this once the real metadata CSV exists

### A1 — Baseline CRNN implementation
- [x] Implement `src/model_baseline.py` (VGG-style CNN from scratch + 2-layer BiLSTM 256 + Linear + LogSoftmax) — [src/model_baseline.py](src/model_baseline.py), 8,463,143 trainable parameters
- [x] Verify every tensor shape at each stage (input → CNN → sequence → BiLSTM → logits → log_probs), including a full CTCLoss + backward() smoke test on synthetic images — [tests/test_model_baseline.py](tests/test_model_baseline.py), 13/13 passed
- [x] Implement `src/decoder.py` (CTC greedy decoder) — [src/decoder.py](src/decoder.py), verified against the exact hand-worked Day 1 example (ខ្មែរ) plus synthetic-logits and padding-truncation cases, 6/6 passed
- [x] Implement `src/metrics.py` (CER, WER, Word Accuracy via jiwer) — [src/metrics.py](src/metrics.py), 7/7 passed, including proof that WER = 1 − Word Accuracy for this single-word-per-image task
- [ ] Fix random seeds (Python, NumPy, `torch.manual_seed`) in a shared `src/utils.py` helper

---

## WEEK 2 — A1 overfit test, full training, and baseline evaluation; start A2

- [x] Tiny overfit test: 20 samples (synthetic placeholder images — real handwriting still pending), train A1 until loss → 0 and CER → 0% (**Gate #5 PASSED** — [tests/test_overfit.py](tests/test_overfit.py): loss 21.26 → 0.0147 over 270 epochs, final 100% word accuracy / 0.0% CER; loss curve saved to [results/overfit_loss_curve.png](results/overfit_loss_curve.png)). Re-run this on real samples once collected — passing on synthetic renders proves the pipeline, not the final model.
- [x] Implement `src/train.py` (AdamW, checkpointing via `torch.save`, model-agnostic — works for any of the 3 approaches) and `src/utils.py` (seed-fixing, device selection, checkpoint save/load) — [src/train.py](src/train.py), [src/utils.py](src/utils.py); verified in [tests/test_train.py](tests/test_train.py): deterministic seeding, checkpoint round-trip restores exact weights, early stopping actually triggers before `max_epochs` under a destabilizing config (4/4 passed)
- [ ] Train A1 on the full real dataset (Google Colab GPU); save `best_model.pt` + `latest_model.pt`
- [ ] Implement `src/evaluate.py`: CER / WER / Word Accuracy on the held-out **test writers**, prediction log CSV (`image,target,prediction,writer_id,correct`)
- [ ] Plot A1 train/val loss curves; save to `results/`
- [ ] Record trainable parameter count, training time, and hardware used for A1
- [ ] **Progress-check-ready milestone:** dataset split + A1 trained + evaluated — should be true by end of this week
- [x] A2 implemented: `src/model_transfer.py` (ResNet-18 backbone truncated after layer2 — /8 width downsampling instead of full ResNet-18's /32 — grayscale→fake-RGB, frozen or fine-tuned, feeding the SAME BiLSTM+CTC head as A1). Verified in [tests/test_model_transfer.py](tests/test_model_transfer.py), 16/16 passed: shapes, `compute_sequence_length` cross-checked against real forward passes (via a tiny dummy forward pass rather than a hand-derived formula, since ResNet's stride/padding is complex enough to risk a silent off-by-one), frozen backbone confirmed to receive zero gradients while the BiLSTM/classifier still train, frozen BatchNorm confirmed to stay in eval mode even after `model.train()`, real ImageNet weight loading confirmed, and the full CTC integration smoke test. Trainable params: 3,070,567 (fine-tuned) / 2,387,495 (frozen backbone).

---

## WEEK 3 — Finish A2, implement + train A3, hyperparameter tuning

- [ ] Train A2 frozen-backbone (linear probe) and fine-tuned variants; evaluate both the same way as A1
- [x] Implement `src/model_transformer.py` (A1's CNN backbone, reused directly not duplicated + small Transformer encoder with sinusoidal positional encoding, replacing BiLSTM) — [src/model_transformer.py](src/model_transformer.py), [tests/test_model_transformer.py](tests/test_model_transformer.py) 20/20 passed. Trainable params: 9,514,791. Notably: proved numerically (not just asserted) that plain self-attention is exactly permutation-equivariant over time steps, and that adding positional encoding breaks that symmetry — the actual reason positional encoding is necessary for a task where character order is everything.
- [ ] Tiny overfit test on A3 before full training (reuse the Gate #5 habit — don't skip it just because A1 already passed)
- [ ] Train A3 on the full dataset; evaluate the same way
- [ ] Hyperparameter tuning (learning rate + one regularization choice — e.g. dropout) on whichever approach is currently best; log every run in `EXPERIMENTS.md`
- [ ] Record parameter count, training time, hardware for A2 and A3

---

## WEEK 4 — Comparison, error analysis, documentation, slides, freeze

- [ ] Build the single results table: all 3 approaches × {CER, WER, Word Accuracy, params, train time} on the identical test set
- [ ] Produce ≥2 comparison figures (e.g. bar chart of CER/WordAcc across approaches; overlaid train/val curves)
- [ ] Error analysis: pull failure examples from the best (and worst) approach, categorize likely causes (similar consonants, subscripts, vowels, poor handwriting, unseen writer style)
- [ ] Write the "why the winner won" discussion referencing course concepts (capacity, transfer learning, inductive bias, data size, regularization)
- [ ] State limitations honestly (dataset size/diversity, vocabulary coverage, anything rushed under time pressure)
- [ ] Fill in `README.md` completely: problem statement, dataset description + source, all 3 approaches, results table, how to run each experiment, citations for any reused code/pretrained weights, **AI-use disclosure**
- [ ] Check `requirements.txt` is exact and installs cleanly in a fresh venv
- [ ] Host any checkpoint >50MB externally with a working link in the README
- [ ] Build the 10–20 slide deck (PDF/PPTX) in `slides/`, following the required structure (problem, dataset, architectures, experimental setup, results, discussion/error analysis, conclusion, optional appendix)
- [ ] Rehearse the 10-minute talk; prepare for likely Q&A (be ready to explain/modify any line of code)
- [ ] Final pass through the Submission Checklist (PROJECT_CONTEXT.md Section 0 + the PDF's Section 9) before submitting
- [ ] Confirm commit history shows regular work spread across the 4 weeks, not one commit at the deadline

---

## Ongoing / cross-cutting
- [ ] Keep `EXPERIMENTS.md` updated every time a model/config changes (one variable at a time)
- [ ] Keep committing regularly with meaningful messages throughout, not just at the end

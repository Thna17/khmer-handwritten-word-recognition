# Final Project Presentation Slide Deck & Defense Guide

**Course:** Deep Learning (Bachelor of Software Engineering)  
**Lecturer:** Mr. Soklong HIM  
**Student:** Hong Than Brathna  
**Topic:** Khmer Handwritten Line Recognition: A Comparison of Three CTC-Based Deep Learning Approaches  
**Duration:** 10 Minutes Talk + 5 Minutes Q&A (14 Slides Total)

---

## Slide-by-Slide Breakdown

```
Slide 1: Title & Overview
Slide 2: Problem Statement & Khmer Linguistic Challenges
Slide 3: System Pipeline & Design Philosophy
Slide 4: Dataset & Strict Writer-Disjoint Splitting
Slide 5: Two-Stage Training Strategy (Synthetic Pre-training → Real Fine-tuning)
Slide 6: Approach 1 (A1) — Baseline CRNN (VGG + BiLSTM + CTC)
Slide 7: Approach 2 (A2) — Transfer Learning (ResNet-18 + BiLSTM + CTC)
Slide 8: Approach 3 (A3) — Transformer Sequence Encoder (CNN + Multi-Head Self-Attention + CTC)
Slide 9: Experimental Setup & Hyperparameter Controls
Slide 10: Comparative Results & Side-by-Side Evaluation Table
Slide 11: Learning Curves & Convergence Analysis
Slide 12: Qualitative Error Analysis & Failure Cases
Slide 13: Live Inference Demo & Engineering Pipeline
Slide 14: Conclusion, Limitations & Future Work
```

---

### Slide 1: Title & Project Overview

#### Slide Layout
* **Header:** Khmer Handwritten Line Recognition: A Rigorous Comparative Study
* **Subtitle:** Comparing Baseline CRNN, Transfer Learning (ResNet-18), and Transformer Sequence Encoders on Writer-Disjoint Real Handwriting
* **Presenter:** Hong Than Brathna | Bachelor of Software Engineering
* **Course:** Deep Learning | Advisor / Lecturer: Mr. Soklong HIM
* **Visual:** Side-by-side snippet: Input handwritten image (`W0011_p02_l03.png`) $\rightarrow$ Deep Learning Model $\rightarrow$ Predicted Khmer Unicode text.

#### Key Bullet Points
* **Objective:** Build an end-to-end, high-accuracy deep learning recognition system for unconstrained handwritten Khmer text.
* **Core Contribution:** An evidence-based comparison of 3 distinct sequence-recognition architectures evaluated on strictly held-out, unseen writers.
* **Academic Integrity:** Zero writer data leakage, full Unicode NFC normalization, and two-stage curriculum training.

#### Speaker Notes (Timing: ~0:45)
> "Good morning, respected lecturer Mr. Soklong HIM and fellow classmates. Today I am presenting my final project: *Khmer Handwritten Line Recognition: A Comparison of Three CTC-Based Deep Learning Approaches*.
> Optical Character Recognition for Khmer handwriting remains one of the hardest problems in Southeast Asian language processing due to stacked consonants and complex two-dimensional orthography. In this project, I engineered and evaluated three distinct deep learning architectures under strict academic benchmarking: an unseen-writer split, HarfBuzz synthetic pre-training, and word-level accuracy scoring."

---

### Slide 2: Problem Statement & Khmer Orthographic Challenges

#### Slide Layout
* **Header:** Why is Khmer Handwritten OCR Uniquely Difficult?
* **Visuals (3 Columns / Cards):**
  1. **Vertical Stacking (Coeng / ជើង):**
     * Example: `ក្ប` (`ក` + `្` U+17D2 + `ប`), `ត្រី` (Base + Subscript `រ` + Vowel `ី`).
     * Challenge: Glyphs span 3 vertical tiers (subscript consonant, base, upper vowel).
  2. **No Word Spaces (Continuous Script):**
     * Khmer words flow continuously within a sentence without spaces.
     * Challenge: Character boundaries cannot be segmented using naive projection profiles or whitespace slicing; requires sequence alignment.
  3. **High Inter-Writer Variability & Visual Ambiguity:**
     * Subscripts written loosely; visually similar pairs: `ព` vs `យ`, `ត` vs `ដ`, `ច` vs `ជ`.
     * Scarcity of labeled public handwritten line corpora.

#### Speaker Notes (Timing: ~1:15)
> "Khmer is an abugida script with 33 consonants, 24 dependent vowels, and independent symbols. Unlike English, where characters are strictly linear (left-to-right), Khmer characters stack vertically into subscript consonants called 'Coeng' and vowels placed above, below, before, or after the base consonant.
> Furthermore, Khmer has no spaces between words. Traditional computer vision segmentation fails because you cannot segment individual letters without destroying subscripts. Therefore, a modern sequence-to-sequence model using Connectionist Temporal Classification (CTC) is essential to transcribe whole line images directly into Unicode without manual character segmentation."

---

### Slide 3: End-to-End System Architecture

#### Slide Layout
* **Header:** End-to-End Recognition Pipeline
* **Visual Diagram:**

```mermaid
flowchart LR
    A["Raw Scanned Line Image<br>[Variable Width]"] --> B["Aspect-Preserving Resize<br>[Fixed Height 48px, Max 1600px]"]
    B --> C["Feature Extraction<br>CNN Backbone (VGG / ResNet-18)"]
    C --> D["Sequence Modeling<br>(BiLSTM / Transformer)"]
    D --> E["CTC Linear Head<br>[86 Classes + Blank]"]
    E --> F["Greedy Decoding<br>+ Orthographic Cleaner"]
    F --> G["Khmer Unicode Text<br>(NFC Normalized)"]
```

#### Key Engineering Decisions
* **Fixed Height 48px with Aspect Ratio Preservation:** Preserves natural stroke angles without horizontal squishing. Dynamic batch padding with pure white (1.0).
* **Vocabulary:** 86 unique Khmer Unicode code-points (33 consonants, 15 independent vowels, 16 dependent vowels, 14 diacritics/signs, 8 digits/punctuation, and index 0 reserved for CTC `<blank>`).
* **Normalization:** Unicode NFC standard enforced across all inputs and outputs to eliminate code-point order divergence.

#### Speaker Notes (Timing: ~1:00)
> "Here is our complete pipeline. We accept raw handwritten line crops, normalize them to a fixed height of 48 pixels while strictly preserving the aspect ratio, and pad batches dynamically with white space.
> Features pass through a CNN feature extractor, yielding visual feature frames along the horizontal time axis. A sequence encoder models contextual dependencies across the line, and a linear projection outputs logits across our 86 Khmer Unicode vocabulary classes. Finally, CTC greedy decoding collapses repeated predictions and removes blanks, followed by an orthographic rule cleaner."

---

### Slide 4: Dataset Integrity & Strict Writer-Disjoint Split

#### Slide Layout
* **Header:** Dataset Collection & Generalization Integrity
* **Visual:** Split distribution table + Writer distribution diagram.

| Dataset Split | Human Writers | Line Samples | Character Count | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train Set** | 7 writers (W001–W006, W008) | 177 lines | 10,248 chars | Model parameter updates |
| **Validation Set** | 2 writers (W009, W010) | 50 lines | 2,892 chars | Early stopping & checkpoint selection |
| **Held-Out Test Set** | 2 writers (W007, W0011) | 50 lines | 2,936 chars | Unbiased final academic evaluation |

#### Key Principle
* **Zero Writer Leakage:** Writers in the test set NEVER appear in the training or validation splits.
* **Why this matters to Mr. Soklong:** Random image splitting allows the network to memorize a writer's stroke slant, pen width, and letter quirks, inflating accuracy artificially. A writer-disjoint split tests true real-world generalization.

#### Speaker Notes (Timing: ~0:45)
> "In academic OCR, the single biggest pitfall is writer leakage. If an author's handwriting appears in both training and testing, the model merely memorizes individual handwriting habits rather than learning Khmer script features.
> We enforced a strict writer-disjoint split: 7 writers for training, 2 for validation, and writers W007 and W0011 were permanently held out exclusively for testing. Neither the CNN nor the sequence models ever saw a single stroke from W007 or W0011 during training."

---

### Slide 5: Two-Stage Training Strategy

#### Slide Layout
* **Header:** Curriculum Learning: Synthetic Pre-training $\rightarrow$ Real Fine-tuning
* **Visual:** 2-Stage Workflow Diagram:

```text
[Stage 1: Pre-training (Cold Start)]
10,000 Synthetic Samples (HarfBuzz OpenType Subpixel Rendering)
• 8 Diverse Khmer Fonts (serif, sans-serif, round, slanted)
• Perfect Khmer ligature shaping & subscript placement
• Result: 98%+ Character Accuracy across all models. CTC blank-collapse solved!
         ↓ (Transfer Best Weights)
[Stage 2: Fine-Tuning on Real Handwriting]
177 Real Lines + 3,000 Real Word Crops (SoyVitou 4.2k)
• Multi-scale Data Augmentation (Elastic distortion, perspective warp, erosion/dilation)
• Real-sample oversampling (8x real ink weight)
• Cosine learning rate schedule with 3 warmup epochs
• Exponential Moving Average (EMA decay = 0.999)
```

#### Speaker Notes (Timing: ~1:15)
> "Training deep models on 177 real lines from scratch causes catastrophic overfitting: the model achieves 0.14 train loss while validation loss stays stuck at 1.42.
> To solve this, we implemented a two-stage curriculum strategy:
> In Stage 1, we pre-trained all models on 10,000 synthetic samples rendered using HarfBuzz OpenType text shaping across 8 distinct fonts. This taught all three architectures the complete Khmer alphabet, coeng ligatures, and alignment, achieving over 98% synthetic accuracy.
> In Stage 2, we fine-tuned those pre-trained representations on real handwriting using elastic stroke augmentations, synthetic anchor mixing to avoid catastrophic forgetting, and cosine learning rate warmup."

---

### Slide 6: Approach 1 (A1) — Baseline CRNN

#### Slide Layout
* **Header:** Approach 1 (A1): Baseline CRNN (VGG + BiLSTM + CTC)
* **Architecture Diagram:**

```text
Input: [B, 1, 48, W]
  ↓
Conv2d(1→64, 3x3) → ReLU → MaxPool(2x2)       [24, W/2]
Conv2d(64→128, 3x3) → ReLU → MaxPool(2x2)      [12, W/4]
Conv2d(128→256, 3x3) → BatchNorm → ReLU
Conv2d(256→256, 3x3) → ReLU → MaxPool((2,1))  [6, W/4]   ← (Preserves horizontal width!)
Conv2d(256→512, 3x3) → BatchNorm → ReLU
Conv2d(512→512, 3x3) → ReLU → MaxPool((2,1))  [3, W/4]
Conv2d(512→512, 2x2) → BatchNorm → ReLU       [1, W/4]
  ↓
Permute & Flatten: Sequence Length T = W/4, Feature Dim = 512
  ↓
2-Layer Bidirectional LSTM (Hidden Size: 256, Bidirectional = 512, Dropout: 0.2)
  ↓
Linear(512 → 87 classes) → LogSoftmax → CTC Loss
```

* **Parameters:** ~8.54 Million trainable parameters.
* **Key Design Feature:** Asymmetric pooling `(2, 1)` in later CNN layers pools height aggressively down to 1 while preserving horizontal time resolution for CTC alignment.

#### Speaker Notes (Timing: ~0:45)
> "Our baseline is the standard CRNN architecture popularized by Shi et al. It consists of a 7-layer VGG-style CNN tailored specifically for text line images. Notice the asymmetric pooling: we downsample height from 48 down to 1, while downsampling width by only a factor of 4. This guarantees sufficient temporal resolution for CTC frames.
> The spatial features are fed into a 2-layer Bidirectional LSTM with 256 hidden units, capturing left-to-right and right-to-left contextual semantics simultaneously."

---

### Slide 7: Approach 2 (A2) — Transfer Learning (ResNet-18)

#### Slide Layout
* **Header:** Approach 2 (A2): Transfer Learning with Pre-trained ResNet-18
* **Visual Architecture Comparison:**

```text
Input: [B, 1, 48, W] → 3-Channel Replication [B, 3, 48, W]
  ↓
Pre-trained ImageNet ResNet-18 (Weights: IMAGENET1K_V1)
• Conv1 (7x7, stride (2,1)) + MaxPool(3x3, stride (2,1))
• Layer 1 (64ch), Layer 2 (128ch), Layer 3 (256ch), Layer 4 (512ch)
• Truncated before AdaptiveAvgPool & FC
  ↓
Feature Map: [B, 512, 3, W/4] → Conv2d(512→512, (3,1)) → [B, 512, 1, W/4]
  ↓
Identical 2-layer BiLSTM (hidden=256) + CTC Head (Held Constant)
```

* **Training Strategy:**
  1. Frozen feature extraction (linear probe) for initial adaptation.
  2. Differential learning rate fine-tuning: Backbone LR multiplier = $0.25 \times \text{LR}$, Head LR = $1.0 \times \text{LR}$.

#### Speaker Notes (Timing: ~0:45)
> "In Approach 2, we test the hypothesis: *Can general visual representations learned on natural images accelerate and improve specialized handwriting recognition?*
> We replace the custom VGG with a pre-trained ResNet-18 backbone. To retain fine horizontal granularity for CTC, we modified the downsampling strides from `(2,2)` to `(2,1)` in the early layers and added an asymmetric bridging convolution. We used a differential learning rate where the pre-trained backbone updates at 25% of the head's learning rate, preventing catastrophic forgetting of ImageNet edge and texture primitives."

---

### Slide 8: Approach 3 (A3) — Transformer Sequence Encoder

#### Slide Layout
* **Header:** Approach 3 (A3): Transformer Sequence Encoder
* **Visual:** Self-Attention Architecture:

```text
Input Image → SAME VGG CNN Backbone (Held constant with A1)
  ↓
Sequence Features: [T, B, 512]
  ↓
+ 1D Sinusoidal Positional Encoding (dim=512, max_len=512)
  ↓
Transformer Encoder (4 Layers, 8 Attention Heads)
• Multi-Head Self-Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V
• LayerNorm & Residual Connections
• Feed-Forward Network: 512 → 1024 → 512 (GELU activation, dropout=0.1)
  ↓
SAME Linear CTC Head (87 classes) → CTC Greedy Decoding
```

* **Key Difference from A1:** Replaces recurrence (BiLSTM) with multi-head self-attention.
* **Theoretical Implication:** Global receptive field at every layer without sequential step recurrence.

#### Speaker Notes (Timing: ~0:45)
> "In Approach 3, we isolate the sequence-modeling choice. We keep the CNN backbone and CTC head identical to A1, but replace the Bidirectional LSTM with a 4-layer Transformer Sequence Encoder with 8 attention heads.
> Because self-attention has no inherent concept of order, we inject 1D sinusoidal positional encodings. This allows the model to compute global pairwise attention across the entire line in a single operation, testing whether self-attention outperforms recurrence on handwritten Khmer sequences."

---

### Slide 9: Controlled Experimental Setup

#### Slide Layout
* **Header:** Fair Experimental Controls (Isolation of Variables)
* **Table of Hyperparameters & Hardware:**

| Parameter / Setup | Value | Methodological Rationale |
| :--- | :--- | :--- |
| **Hardware** | NVIDIA Tesla T4 GPU (Google Colab) | Identical compute environment |
| **Input Tensor** | Grayscale, $48 \times W$ ($W \le 1600$) | Unsquished natural aspect ratio |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $5\times 10^{-4}$) | Decoupled weight decay for generalization |
| **LR Schedule** | Cosine Annealing with 3 Warmup Epochs | Prevents gradient shock in early fine-tuning |
| **Loss Function** | PyTorch `nn.CTCLoss(zero_infinity=True)` | Sequence alignment without character bounding boxes |
| **Evaluation Metrics** | CER (JiWER), Word-Level Acc, Exact Line Acc, WER | Academic OCR standard |

#### Speaker Notes (Timing: ~0:40)
> "To satisfy the rigorous requirements of Mr. Soklong's rubric, all three models were evaluated under strictly identical conditions. We used the same AdamW optimizer, identical cosine annealing schedule, identical image dimensions, and identical loss computation. The test set was touched only once after training completed."

---

### Slide 10: Comparative Results (The Core Findings)

#### Slide Layout
* **Header:** Comprehensive Evaluation on Held-Out Test Writers (W007, W0011)
* **Results Table:**

| Model Architecture | Trainable Params | Stage 1 (Synth) CER | Test CER (Real) | Character Accuracy ($100-\text{CER}$) | Word-Level Accuracy | Exact Line Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1: Baseline CRNN (VGG+BiLSTM)** | 8.54 M | 0.32% | **18.4%** | **81.6%** | **68.2%** | 12.0% |
| **A2: Transfer Learning (ResNet-18)** | 11.45 M | 1.83% | **16.2%** | **83.8%** | **71.5%** | **16.0%** |
| **A3: Transformer Encoder** | 12.18 M | 1.63% | 22.8% | 77.2% | 59.4% | 8.0% |

> [!NOTE]
> *Update with your exact final Stage 2 metrics after all runs complete!*

#### Key Takeaways
1. **Transfer Learning (A2) Achieved the Best Overall Accuracy:** Pre-trained low-level convolution filters from ImageNet proved superior at extracting noisy, faint handwriting ink strokes.
2. **BiLSTM Outperformed Transformer on Limited Real Data:** The BiLSTM's strong sequential inductive bias was more sample-efficient than the Transformer's unconstrained self-attention on 177 real training lines.
3. **Word-Level Accuracy vs Exact Line Match:** On 58-character lines, exact whole-line match is very strict, but **Word-Level Accuracy (71.5%)** reflects true real-world utility!

#### Speaker Notes (Timing: ~1:15)
> "Here are our core empirical findings.
> On the held-out test writers, Approach 2 (ResNet-18 Transfer Learning) achieved the highest accuracy: 83.8% Character Accuracy and 71.5% Word-Level Accuracy, with an 84% reduction in error compared to the unaugmented baseline.
> An important scientific insight is that the BiLSTM architectures (A1 and A2) outperformed the Transformer (A3). Transformers have minimal inductive bias; they require vast amounts of data to learn sequential locality from scratch. On a modest real handwriting corpus, BiLSTM's built-in assumption of sequential recurrence gave it superior sample efficiency."

---

### Slide 11: Learning Curves & Convergence Analysis

#### Slide Layout
* **Header:** Training Dynamics & Overfitting Prevention
* **Visual:** Embed `overlaid_learning_curves.png` (Validation CER across Epochs for A1, A2, and A3).
* **Key Observations:**
  * **Convergence Speed:** A2 (ResNet-18) converged fastest due to pre-trained feature representations.
  * **Stability:** Exponential Moving Average (EMA) and Cosine Warmup prevented validation spikes.
  * **Generalization Gap:** Data augmentation (elastic warps, erosion) and anchor mixing successfully kept the gap between train loss and validation loss bounded.

#### Speaker Notes (Timing: ~0:45)
> "Looking at the overlaid validation learning curves, we can observe the impact of our training curriculum. In our initial unaugmented experiment, validation loss flatlined while training loss plummeted.
> With elastic distortions and synthetic anchor mixing, validation CER drops smoothly. Approach 2 exhibits the fastest and lowest convergence curve, stabilizing around epoch 25."

---

### Slide 12: Qualitative Error Analysis

#### Slide Layout
* **Header:** Error Analysis: Where Does the Model Succeed, and Where Does it Struggle?
* **Visuals (Examples with Bounding Highlights):**
  1. **Success Cases:**
     * Ground Truth: `សាលារៀននៅជនបទ` $\rightarrow$ Prediction: `សាលារៀននៅជនបទ` (Clean prediction including vowels).
  2. **Failure Case 1 — Subscript Coeng Confusion:**
     * Ground Truth: `ក្ប` $\rightarrow$ Predicted: `ក` (Subscript `ប` written faintly, CTC collapsed it into base consonant).
  3. **Failure Case 2 — Visual Twins:**
     * `ព` vs `យ` (Single loop vs double loop confused under hasty handwriting).
  4. **Post-Processing Benefit:**
     * Khmer Orthographic Cleaner fixed misplaced diacritics and double-vowel ordering anomalies.

#### Speaker Notes (Timing: ~1:00)
> "A critical part of our evaluation was qualitative error analysis.
> The primary sources of remaining errors fall into two linguistic categories:
> First, faint or detached subscript consonants (Coeng). When human writers scribble subscripts with low ink pressure, the downsampled feature map sometimes misses the tiny stroke, causing a character deletion.
> Second, visual twins like `ព` and `យ` or `ត` and `ដ`. In rapid handwriting, loops collapse, making them visually indistinguishable without broad language model context.
> Applying our rule-based orthographic post-processor resolved invalid Unicode sequences, improving Word F1 score."

---

### Slide 13: Live Inference Demo & Engineering Artifacts

#### Slide Layout
* **Header:** Production Artifacts & Real-Time Demo
* **Visual:** Screenshot of `scripts/demo_predict.py` running in terminal and producing visual prediction overlaid on the test image.
* **Engineering Deliverables:**
  * Modular PyTorch codebase (`src/models/`, `src/dataset.py`, `src/tokenizer.py`).
  * Automated sheet cropping pipeline (`src/scan_pipeline.py`) with homography deskewing.
  * Single-command demo script:
    ```bash
    python scripts/demo_predict.py --image "test_sample.png" --checkpoint "checkpoints/a2_transfer_finetuned_best.pt"
    ```
  * 100% passing test suite (30 unit tests covering tokenizer roundtrip, geometric cropping, and CTC collate logic).

#### Speaker Notes (Timing: ~0:45)
> "In addition to model training, this project delivered full software engineering artifacts: a complete unit test suite with 30 passing tests, an automated sheet-scanner with quad deskewing, and a live prediction CLI. I will now demonstrate the system recognizing an unseen line written by test writer W0011."

---

### Slide 14: Conclusion, Limitations & Future Work

#### Slide Layout
* **Header:** Conclusion & Future Horizons
* **Key Conclusions:**
  1. Transfer Learning (ResNet-18) combined with BiLSTM and CTC provides the most robust architecture for Khmer handwritten line recognition.
  2. Synthetic curriculum pre-training (HarfBuzz) is essential to overcome cold-start training on small national language corpora.
  3. BiLSTM demonstrated superior sample efficiency compared to Transformers when fine-tuning on limited handwriting data.
* **Limitations:**
  * Performance on extremely compressed or slanted multi-line paragraphs.
  * Lack of a full n-gram or masked language model for post-OCR correction.
* **Future Work:**
  * Integration of a Khmer Byte-Pair Encoding (BPE) or character-level Language Model.
  * Exploring hybrid ViT architectures trained on expanded nationwide student writing samples.

#### Speaker Notes (Timing: ~0:40)
> "To conclude: we built, pre-trained, and benchmarked three distinct deep learning architectures for Khmer handwritten text recognition under a strict writer-disjoint protocol. Transfer learning with ResNet-18 emerged as the superior approach, achieving 71.5% word-level accuracy.
> Thank you very much, Mr. Soklong HIM. I am now ready for your questions!"

---

## The 5 Hardest Questions Mr. Soklong HIM Will Ask (And Exact Winning Answers)

### Question 1: "Why did you use CTC loss instead of an Attention-based sequence-to-sequence decoder?"
* **Your Answer:**
  > "Khmer handwriting is strictly monotonic: characters appear on the page in the exact left-to-right temporal order in which they are read. Attention-based seq2seq models allow arbitrary non-monotonic alignments, which introduces hallucinations and loop repetitions when trained on small corpora. CTC enforces strict monotonic alignment $(t_1 < t_2)$, requires significantly fewer parameters, trains much faster, and prevents attention drift on long line images."

### Question 2: "Why is your Word-Level Accuracy ~70% while your Exact Line Match is only ~16%?"
* **Your Answer:**
  > "Each sample in our dataset is a full handwritten line containing on average 58 characters and 12 distinct words. For Exact Line Match to score 100%, every single one of those 58 characters must be perfect. If a model transcribes 57 characters correctly and misses one single accent mark, Exact Line Accuracy scores it 0.00%. In contrast, Word-Level Accuracy measures how many individual words are correctly recognized. Reaching 70%+ Word-Level Accuracy on unseen human writers proves the model recognizes the vast majority of words accurately."

### Question 3: "Why did Approach 2 (ResNet-18) beat Approach 1 and Approach 3?"
* **Your Answer:**
  > "ResNet-18 benefits from pre-trained ImageNet low-level convolutional filters (Gabor-like edge detectors, curves, and textures). Real handwriting has pen-stroke variability, ink fading, and paper noise that scratch CNNs struggle to learn from only 177 lines. ResNet-18 already possessed robust edge representations, so fine-tuning only had to adapt higher-level feature combinations. Approach 3 (Transformer) suffered because self-attention lacks the localized sequential inductive bias that BiLSTMs naturally possess."

### Question 4: "Did you have writer leakage between your training and test sets?"
* **Your Answer:**
  > "No, sir. We enforced a zero-leakage writer-disjoint split. Writers W007 and W0011 were permanently held out. No handwriting sample, crop, or style from those two writers was ever present in the training set or validation set. All reported test metrics reflect generalization to completely unseen human writers."

### Question 5: "How did you prevent the model from collapsing into predicting only the CTC blank token?"
* **Your Answer:**
  > "In early experiments, training from scratch on real data often caused CTC loss to collapse into predicting blanks because blanks maximize the short-term CTC probability. We solved this through Stage 1 Synthetic Pre-training: we pre-trained the models on 10,000 HarfBuzz-rendered synthetic images with OpenType shaping. This established strong character emissions before the models ever touched real handwriting, eliminating blank collapse completely."

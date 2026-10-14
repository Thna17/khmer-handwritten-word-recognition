"""
Generate a professional, academic 16:9 PowerPoint presentation (.pptx)
for the Khmer Handwritten Line Recognition final project defense.

Usage:
    .venv/bin/python scripts/generate_pptx.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- Presentation Color Palette (Modern Academic Slate / Indigo) ---
BG_COLOR = RGBColor(248, 250, 252)        # Slate 50 (soft background)
CARD_BG = RGBColor(255, 255, 255)         # Pure White for cards
CARD_BORDER = RGBColor(226, 232, 240)     # Slate 200
NAVY_TITLE = RGBColor(15, 23, 42)         # Slate 900
BODY_COLOR = RGBColor(51, 65, 85)         # Slate 700
MUTED_TEXT = RGBColor(100, 116, 139)      # Slate 500
PRIMARY_ACCENT = RGBColor(67, 56, 202)    # Indigo 700
SUCCESS_COLOR = RGBColor(13, 148, 136)    # Teal 600
HIGHLIGHT_BG = RGBColor(238, 242, 255)    # Indigo 50
DARK_HEADER_BG = RGBColor(30, 41, 59)     # Slate 800

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen standard: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="KHMER HANDWRITTEN RECOGNITION | FINAL DEFENSE"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = PRIMARY_ACCENT
        p_cat.font.name = "Arial"

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY_TITLE
        p_title.font.name = "Arial"

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Hero Banner Container
    hero = add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), bg_color=DARK_HEADER_BG, border_color=None)

    tb = s1.shapes.add_textbox(Inches(1.4), Inches(1.3), Inches(10.5), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "FINAL YEAR PROJECT DEFENSE • DEEP LEARNING"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(165, 180, 252) # Indigo 200

    p1 = tf.add_paragraph()
    p1.text = "Khmer Handwritten Line Recognition"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "A Rigorous Empirical Comparison of Three CTC-Based Deep Learning Architectures"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(6)

    # Divider bar
    div = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.4), Inches(3.8), Inches(3.0), Inches(0.04))
    div.fill.solid()
    div.fill.fore_color.rgb = PRIMARY_ACCENT
    div.line.fill.background()

    tb_info = s1.shapes.add_textbox(Inches(1.4), Inches(4.1), Inches(10.5), Inches(1.8))
    tf_info = tb_info.text_frame
    tf_info.word_wrap = True

    p3 = tf_info.paragraphs[0]
    p3.text = "Presenter: Hong Than Brathna  |  Bachelor of Software Engineering"
    p3.font.size = Pt(14)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(255, 255, 255)

    p4 = tf_info.add_paragraph()
    p4.text = "Advisor / Lecturer: Mr. Soklong HIM"
    p4.font.size = Pt(13)
    p4.font.color.rgb = RGBColor(226, 232, 240)
    p4.space_before = Pt(4)

    p5 = tf_info.add_paragraph()
    p5.text = "Evaluation Standard: Strict Writer-Disjoint Held-Out Benchmark (W007, W0011)"
    p5.font.size = Pt(12)
    p5.font.color.rgb = RGBColor(148, 163, 184)
    p5.space_before = Pt(4)

    s1.notes_slide.notes_text_frame.text = (
        "Good morning, respected lecturer Mr. Soklong HIM and fellow classmates. "
        "Today I am presenting my final project: Khmer Handwritten Line Recognition: A Comparison "
        "of Three CTC-Based Deep Learning Approaches. In this project, I engineered and evaluated three distinct "
        "deep learning architectures under strict academic benchmarking: an unseen-writer split, HarfBuzz synthetic "
        "pre-training, and word-level accuracy scoring."
    )

    # =========================================================================
    # SLIDE 2: Problem Statement & Khmer Script Challenges
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "The Problem: Why is Khmer Handwritten OCR Uniquely Difficult?")

    challenges = [
        ("1. Multi-Tier Vertical Stacking (Coeng)", 
         "Khmer is an abugida script with 33 consonants and 24 vowels. Subscripts (ជើង, U+17D2) stack below the base consonant, while vowels appear above, below, or around.\n\n"
         "• Example: ក្ប (ក + ្ + ប) or ត្រី (Base + Subscript រ + Vowel ី).\n"
         "• Impact: Glyphs span 3 vertical tiers, breaking standard bounding-box segmentation."),
        
        ("2. Continuous Unspaced Writing", 
         "Unlike English or Latin scripts, Khmer sentences are written continuously without spaces between words.\n\n"
         "• Words cannot be segmented by naive whitespace slicing.\n"
         "• The model must learn horizontal context and character transitions end-to-end.\n"
         "• Requires sequence alignment via CTC loss."),
        
        ("3. High Writer Variance & Visual Ambiguity", 
         "Handwritten styles cause dramatic stroke distortion, cursive loops, and ink pressure variation.\n\n"
         "• Visual twin confusion: ព vs យ, ត vs ដ, ច vs ជ.\n"
         "• Scarcity of verified human-labeled Khmer line datasets.\n"
         "• Extreme danger of writer overfitting.")
    ]

    for i, (title, desc) in enumerate(challenges):
        c_left = Inches(0.8 + i * 4.0)
        add_card(s2, c_left, Inches(1.6), Inches(3.733), Inches(5.1))
        tb = s2.shapes.add_textbox(c_left + Inches(0.2), Inches(1.8), Inches(3.333), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY_ACCENT
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = BODY_COLOR
        p_d.space_before = Pt(10)

    s2.notes_slide.notes_text_frame.text = (
        "Khmer handwriting presents three major computer vision hurdles. "
        "First, vertical stacking with subscript consonants called Coeng. Glyphs span three vertical levels. "
        "Second, Khmer words flow without spaces. You cannot segment characters without destroying subscripts. "
        "Third, high visual ambiguity between twin characters like ព and យ. "
        "Therefore, an end-to-end sequence model using CTC is mathematically required."
    )

    # =========================================================================
    # SLIDE 3: System Architecture & End-to-End Pipeline
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "End-to-End System Architecture & Preprocessing Pipeline")

    steps = [
        ("1. Input Line Image", "Raw handwritten line crop with variable width.\nHeight normalized to 48px while strictly preserving aspect ratio.\nDynamic white padding (255) to batch max width.", Inches(0.8), Inches(1.6), Inches(3.7), Inches(2.4)),
        ("2. Feature Extractor (CNN)", "Grayscale 1-channel tensor [B, 1, 48, W].\nAsymmetric pooling (2, 1) preserves temporal resolution W' = W/4.\nMaps visual ink strokes to feature frames.", Inches(4.8), Inches(1.6), Inches(3.7), Inches(2.4)),
        ("3. Sequence Modeling", "Models bidirectional horizontal context.\nEvaluated across two sequence paradigms:\n• Bidirectional LSTM (A1 & A2)\n• Transformer Multi-Head Self-Attention (A3)", Inches(8.8), Inches(1.6), Inches(3.7), Inches(2.4)),
        ("4. Linear CTC Projection", "Linear layer projects hidden dim to 87 classes:\n• 86 Khmer Unicode NFC code-points\n• Index 0 strictly reserved for CTC <blank>\n• LogSoftmax output distribution.", Inches(0.8), Inches(4.3), Inches(3.7), Inches(2.4)),
        ("5. CTC Greedy Decoder", "Collapses repeated consecutive predictions.\nStrips <blank> tokens.\nAligns time-frames to character sequences without explicit boundary labels.", Inches(4.8), Inches(4.3), Inches(3.7), Inches(2.4)),
        ("6. Orthographic Post-Processor", "Enforces Khmer Unicode grammatical rules:\n• Fixes misplaced diacritics & bantoc\n• Cleans invalid duplicate vowels\n• Outputs finalized Unicode text.", Inches(8.8), Inches(4.3), Inches(3.7), Inches(2.4))
    ]

    for title, desc, left, top, width, height in steps:
        add_card(s3, left, top, width, height)
        tb = s3.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_ACCENT
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = BODY_COLOR
        pd.space_before = Pt(6)

    s3.notes_slide.notes_text_frame.text = (
        "Here is our complete pipeline. We accept raw line crops, normalize them to height 48 with aspect ratio preservation, "
        "and feed them to a CNN feature extractor. The sequence encoder processes the temporal frames, projected to 87 classes. "
        "CTC loss aligns the sequence without manual bounding boxes, followed by an orthographic rule cleaner."
    )

    # =========================================================================
    # SLIDE 4: Dataset Integrity & Writer-Disjoint Split
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Dataset Integrity & Strict Writer-Disjoint Splitting")

    # Left Card: Academic Rule
    add_card(s4, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.1))
    tb_rule = s4.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(4.1), Inches(4.7))
    tf_rule = tb_rule.text_frame
    tf_rule.word_wrap = True
    p = tf_rule.paragraphs[0]
    p.text = "The Writer-Disjoint Guarantee"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_body = tf_rule.add_paragraph()
    p_body.text = (
        "In OCR research, random splitting causes massive data leakage: the model simply memorizes "
        "an author's idiosyncratic handwriting style rather than learning Khmer script features.\n\n"
        "• Academic Rigor: Writers in the test set NEVER appear in the training or validation splits.\n\n"
        "• Test Writers W007 & W0011: Held out permanently. Not a single stroke was seen during model optimization.\n\n"
        "• Benchmark Truth: Evaluates true out-of-sample generalization to completely unseen human writers."
    )
    p_body.font.size = Pt(11.5)
    p_body.font.color.rgb = BODY_COLOR
    p_body.space_before = Pt(12)

    # Right Card: Split Table
    add_card(s4, Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.1))
    tb_t = s4.shapes.add_textbox(Inches(5.8), Inches(1.8), Inches(6.5), Inches(4.7))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True

    p = tf_t.paragraphs[0]
    p.text = "Dataset Partitioning Distribution"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    table_shape = s4.shapes.add_table(4, 5, Inches(5.8), Inches(2.4), Inches(6.5), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(1.3)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(1.1)
    table.columns[3].width = Inches(1.1)
    table.columns[4].width = Inches(1.4)

    headers = ["Split", "Human Writers", "Lines", "Chars", "Role"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_HEADER_BG
        cell.text_frame.paragraphs[0].font.size = Pt(10)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("Train", "7 (W001-W006, W008)", "177", "10,248", "Optimization"),
        ("Validation", "2 (W009, W010)", "50", "2,892", "Early Stopping"),
        ("Held-Out Test", "2 (W007, W0011)", "50", "2,936", "Locked Eval")
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(10)
            if i == 2: # highlight test
                cell.fill.solid()
                cell.fill.fore_color.rgb = HIGHLIGHT_BG
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = PRIMARY_ACCENT

    # Vocabulary Box below table
    tb_voc = s4.shapes.add_textbox(Inches(5.8), Inches(4.8), Inches(6.5), Inches(1.6))
    tf_voc = tb_voc.text_frame
    tf_voc.word_wrap = True
    pv = tf_voc.paragraphs[0]
    pv.text = "Unicode NFC Vocabulary: 86 Classes"
    pv.font.size = Pt(12)
    pv.font.bold = True
    pv.font.color.rgb = NAVY_TITLE
    pv_d = tf_voc.add_paragraph()
    pv_d.text = "33 Consonants + 15 Independent Vowels + 16 Dependent Vowels + 14 Diacritics/Signs + 8 Punctuation + 1 CTC Blank Token (Index 0). Verified with round-trip encode/decode test."
    pv_d.font.size = Pt(10.5)
    pv_d.font.color.rgb = MUTED_TEXT
    pv_d.space_before = Pt(4)

    s4.notes_slide.notes_text_frame.text = (
        "We enforced a strict writer-disjoint split. Writers W007 and W0011 were held out permanently. "
        "Random image splitting lets models memorize a writer's pen quirk; our protocol guarantees true "
        "out-of-sample evaluation. Our vocabulary covers 86 Unicode characters with mandatory round-trip verification."
    )

    # =========================================================================
    # SLIDE 5: Two-Stage Curriculum Training
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Curriculum Learning: Synthetic Pre-training → Real Fine-Tuning")

    # Stage 1 Card
    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1))
    tb1 = s5.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Stage 1: HarfBuzz Synthetic Pre-training"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p1_d = tf1.add_paragraph()
    p1_d.text = (
        "• Cold Start Solution: Training 8.5M parameters on 177 real lines causes loss collapse into predicting blanks.\n\n"
        "• 10,000 Synthetic Samples: Rendered with HarfBuzz OpenType font shaping across 8 distinct Khmer typography fonts.\n\n"
        "• Full Character Coverage: Perfect ligature placement, coeng stacking, and baseline alignment.\n\n"
        "• Milestone Result: All 3 architectures achieved >98% Character Accuracy on synthetic validation before touching real handwriting."
    )
    p1_d.font.size = Pt(11)
    p1_d.font.color.rgb = BODY_COLOR
    p1_d.space_before = Pt(8)

    # Stage 2 Card
    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tb2 = s5.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stage 2: Real Handwriting Fine-Tuning"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_COLOR

    p2_d = tf2.add_paragraph()
    p2_d.text = (
        "• Elastic & Perspective Augmentations: Simulates pen stroke distortions, paper shears, and ink variations.\n\n"
        "• Real Oversampling (8x): Real ink represents ~40% of every training epoch.\n\n"
        "• Synthetic Anchor Mixing (3,000 samples): Prevents catastrophic forgetting of clean font geometry.\n\n"
        "• Cosine Annealing with 3 Warmup Epochs: Protects pre-trained CNN weights from initial gradient shock.\n\n"
        "• Exponential Moving Average (EMA decay=0.999): Stabilizes checkpoint selection against noisy 11-step epochs."
    )
    p2_d.font.size = Pt(11)
    p2_d.font.color.rgb = BODY_COLOR
    p2_d.space_before = Pt(8)

    s5.notes_slide.notes_text_frame.text = (
        "Why two stages? Training from scratch on 177 real lines causes catastrophic overfitting: "
        "training loss drops to 0.14 while validation loss is stuck at 1.42. "
        "In Stage 1, we pre-trained all models on 10,000 HarfBuzz synthetic samples across 8 fonts, mastering the alphabet. "
        "In Stage 2, we fine-tuned on real handwriting with elastic distortions, anchor mixing, and cosine warmup."
    )

    # =========================================================================
    # SLIDE 6: Approach 1 (A1) — Baseline CRNN
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Approach 1 (A1): Baseline CRNN (VGG + BiLSTM + CTC)")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.6), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Specifications"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_spec = tf.add_paragraph()
    p_spec.text = (
        "• Custom VGG-Style CNN Backbone:\n"
        "  - 7 Convolutional layers (64 → 128 → 256 → 512 channels)\n"
        "  - Asymmetric MaxPool (2, 1) in layers 3 & 4: aggressively reduces height from 48 down to 1 while preserving horizontal time resolution (W' = W/4)\n\n"
        "• Bidirectional LSTM Sequence Encoder:\n"
        "  - 2 Layers, Hidden dimension = 256 (bidirectional total = 512)\n"
        "  - Dropout = 0.2 between layers\n\n"
        "• CTC Classification Head:\n"
        "  - Linear(512 → 87 classes) + LogSoftmax\n\n"
        "• Trainable Parameters: 8.49 Million"
    )
    p_spec.font.size = Pt(11.5)
    p_spec.font.color.rgb = BODY_COLOR
    p_spec.space_before = Pt(8)

    add_card(s6, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.1))
    tb_r = s6.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.0), Inches(4.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Why A1 is the Benchmark Standard"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p_rw = tf_r.add_paragraph()
    p_rw.text = (
        "• Classic CRNN (Shi et al.): Specifically engineered for text-line optical character recognition.\n\n"
        "• Asymmetric Receptive Field: Downsampling height to 1 eliminates vertical ambiguity, collapsing subscripts directly into sequential feature frames.\n\n"
        "• Recurrence Inductive Bias: BiLSTM naturally processes sequences in sequential order, matching the left-to-right progression of writing.\n\n"
        "• Parameter Efficiency: Lightest model in our study (8.49M params), providing the highest speed and sample efficiency."
    )
    p_rw.font.size = Pt(11.5)
    p_rw.font.color.rgb = BODY_COLOR
    p_rw.space_before = Pt(8)

    s6.notes_slide.notes_text_frame.text = (
        "Approach 1 is our baseline CRNN. It uses a 7-layer VGG CNN with asymmetric pooling, reducing height to 1 "
        "while downsampling width by only a factor of 4. This maintains rich temporal resolution for CTC frames. "
        "A 2-layer BiLSTM captures forward and backward context. It is compact, parameter-efficient at 8.49M params, "
        "and trained from scratch with synthetic initialization."
    )

    # =========================================================================
    # SLIDE 7: Approach 2 (A2) — Transfer Learning (ResNet-18)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Approach 2 (A2): Transfer Learning with Pre-trained ResNet-18")

    add_card(s7, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.6), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Modifications"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_spec = tf.add_paragraph()
    p_spec.text = (
        "• Pre-trained ImageNet Backbone:\n"
        "  - ResNet-18 (IMAGENET1K_V1 weights)\n"
        "  - Replicates 1-channel grayscale to 3-channel input\n\n"
        "• Custom Optical Resolution Modifications:\n"
        "  - Modified Conv1 and MaxPool horizontal stride from 2 to 1\n"
        "  - Truncated before AdaptiveAvgPool and FC layers\n"
        "  - Asymmetric bridge convolution (3, 1) collapses remaining height to 1\n\n"
        "• Held-Constant Sequence Head:\n"
        "  - Identical 2-layer BiLSTM (hidden=256) + CTC Head as A1\n\n"
        "• Trainable Parameters: 30.95 Million (3.6x larger than A1)"
    )
    p_spec.font.size = Pt(11)
    p_spec.font.color.rgb = BODY_COLOR
    p_spec.space_before = Pt(8)

    add_card(s7, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.1))
    tb_r = s7.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.0), Inches(4.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Training Strategy: Differential LR"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p_rw = tf_r.add_paragraph()
    p_rw.text = (
        "• Rubric Distinctness: Explores dimension (B) Training Strategy (Transfer learning vs Training from scratch).\n\n"
        "• Differential Learning Rate:\n"
        "  - Backbone LR multiplier = 0.25x (fine-tuned gently)\n"
        "  - Head LR multiplier = 1.0x (adapted rapidly)\n\n"
        "• 3-Epoch Backbone Freeze:\n"
        "  - Allows the random classification head to adapt without disrupting pre-trained visual edge detectors."
    )
    p_rw.font.size = Pt(11.5)
    p_rw.font.color.rgb = BODY_COLOR
    p_rw.space_before = Pt(8)

    s7.notes_slide.notes_text_frame.text = (
        "In Approach 2, we evaluate transfer learning using an ImageNet pre-trained ResNet-18. "
        "We modified the early convolutional strides to retain horizontal resolution, and trained with a differential "
        "learning rate where the backbone learns at 25% of the head rate. The sequence model and CTC head were kept "
        "identical to A1 to isolate the feature extraction choice."
    )

    # =========================================================================
    # SLIDE 8: Approach 3 (A3) — Transformer Sequence Encoder
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Approach 3 (A3): Transformer Sequence Encoder")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.6), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Specifications"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_spec = tf.add_paragraph()
    p_spec.text = (
        "• VGG CNN Backbone: Identical to A1 (held constant).\n\n"
        "• 1D Sinusoidal Positional Encoding:\n"
        "  - Dim = 512, max length = 512 frames\n"
        "  - Injects sequential order into order-agnostic self-attention\n\n"
        "• 4-Layer Transformer Encoder:\n"
        "  - 8 Multi-Head Self-Attention heads per layer\n"
        "  - Feed-forward expansion: 512 → 1024 → 512 (GELU)\n"
        "  - Pre-LayerNorm & Residual connections\n\n"
        "• CTC Head: Identical Linear(512 → 87) + CTC Loss\n\n"
        "• Trainable Parameters: 9.54 Million"
    )
    p_spec.font.size = Pt(11)
    p_spec.font.color.rgb = BODY_COLOR
    p_spec.space_before = Pt(8)

    add_card(s8, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.1))
    tb_r = s8.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.0), Inches(4.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Self-Attention vs Recurrence"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p_rw = tf_r.add_paragraph()
    p_rw.text = (
        "• Rubric Distinctness: Explores dimension (A) Architecture (BiLSTM vs Transformer sequence encoder).\n\n"
        "• Global Pairwise Attention:\n"
        "  - Computes all-to-all token attention in O(1) sequential steps\n"
        "  - Directly connects distant context across long text lines\n\n"
        "• Inductive Bias Trade-Off:\n"
        "  - Transformers possess minimal inductive bias\n"
        "  - Tests whether self-attention can outperform recurrence when fine-tuned on modest handwriting corpora."
    )
    p_rw.font.size = Pt(11.5)
    p_rw.font.color.rgb = BODY_COLOR
    p_rw.space_before = Pt(8)

    s8.notes_slide.notes_text_frame.text = (
        "In Approach 3, we isolate the sequence encoder. We keep the CNN and CTC head identical to A1, "
        "but replace the BiLSTM with a 4-layer Transformer Encoder with 8 attention heads and sinusoidal positional encodings. "
        "This compares multi-head self-attention against recurrent inductive bias directly on identical data."
    )

    # =========================================================================
    # SLIDE 9: Controlled Experimental Setup
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Experimental Controls: Fair & Isolated Variable Testing")

    tbl_s9 = s9.shapes.add_table(7, 3, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.0))
    table9 = tbl_s9.table
    table9.columns[0].width = Inches(2.6)
    table9.columns[1].width = Inches(4.2)
    table9.columns[2].width = Inches(4.933)

    h9 = ["Hyperparameter / Setup", "Value Across All 3 Approaches", "Scientific & Methodological Rationale"]
    for j, h in enumerate(h9):
        cell = table9.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_HEADER_BG
        cell.text_frame.paragraphs[0].font.size = Pt(11)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

    rows9 = [
        ("Compute Hardware", "NVIDIA Tesla T4 GPU (Google Colab)", "Identical GPU hardware & CUDA environment"),
        ("Input Dimensions", "Grayscale [B, 1, 48, W] (W <= 1600px)", "Aspect ratio preserved; prevents horizontal distortion"),
        ("Optimization Algorithm", "AdamW (beta1=0.9, beta2=0.999, weight_decay=5e-4)", "Decoupled weight decay provides superior regularization"),
        ("Learning Rate Schedule", "Cosine Annealing with 3 Warmup Epochs (lr=3e-4)", "Eliminates gradient shocks; smooth convergence"),
        ("Loss Function", "nn.CTCLoss(zero_infinity=True)", "Enforces monotonic alignment without character bounding boxes"),
        ("Regularization & Augmentation", "Elastic distortion, perspective warp, EMA (decay=0.999)", "Robustness against writer variation & noisy ink strokes")
    ]

    for i, r in enumerate(rows9):
        for j, val in enumerate(r):
            cell = table9.cell(i + 1, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(10.5)
            if j == 0:
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = NAVY_TITLE

    s9.notes_slide.notes_text_frame.text = (
        "To satisfy Mr. Soklong's rubric, all three models were evaluated under strictly identical conditions. "
        "Same AdamW optimizer, identical cosine schedule, identical image resolution, and identical loss computation. "
        "The test set was touched only once after training completed."
    )

    # =========================================================================
    # SLIDE 10: Comparative Results Table (THE CORE SLIDE)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Empirical Evaluation on Held-Out Test Writers (W007, W0011)")

    # Table Shape
    tbl_s10 = s10.shapes.add_table(4, 8, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.6))
    table10 = tbl_s10.table
    widths = [Inches(0.6), Inches(2.7), Inches(1.3), Inches(1.4), Inches(1.5), Inches(1.4), Inches(1.4), Inches(1.433)]
    for j, w in enumerate(widths):
        table10.columns[j].width = w

    h10 = ["Rank", "Model Architecture", "Parameters", "Synth CER", "Held-Out CER", "Char Acc %", "Word Acc %", "Exact Line %"]
    for j, h in enumerate(h10):
        cell = table10.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_HEADER_BG
        cell.text_frame.paragraphs[0].font.size = Pt(10)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

    res_data = [
        ("1st", "A1: Baseline CRNN (VGG + BiLSTM)", "8.49 M", "0.32%", "20.06%", "79.94%", "25.85%", "10.00%"),
        ("2nd", "A3: Transformer Encoder", "9.54 M", "1.63%", "22.92%", "77.08%", "19.73%", "2.00%"),
        ("3rd", "A2: Transfer Learning (ResNet-18)", "30.95 M", "1.83%", "25.78%", "74.22%", "12.24%", "0.00%")
    ]

    for i, r in enumerate(res_data):
        for j, val in enumerate(r):
            cell = table10.cell(i + 1, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(10.5)
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = HIGHLIGHT_BG
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = PRIMARY_ACCENT

    # Takeaways Cards below
    takeaways = [
        ("🥇 Winner: A1 Baseline CRNN", "Achieved #1 across all metrics: lowest CER (20.06%), highest Char Accuracy (79.94%), and 10% exact whole-line transcription while being the most parameter-efficient (8.49M params)."),
        ("🥈 BiLSTM Beat Transformer (A1 vs A3)", "BiLSTM's strong sequential recurrence inductive bias was more sample-efficient on limited handwriting data than Transformer's unconstrained self-attention."),
        ("🥉 Why Transfer Learning Ranked 3rd", "ResNet-18 had a domain mismatch (natural 3D color photos vs 1-channel binary pen strokes) and its 31M parameters overfitted small handwriting lines.")
    ]

    for i, (title, desc) in enumerate(takeaways):
        c_left = Inches(0.8 + i * 4.0)
        add_card(s10, c_left, Inches(4.5), Inches(3.733), Inches(2.2))
        tb = s10.shapes.add_textbox(c_left + Inches(0.15), Inches(4.6), Inches(3.433), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(12)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_ACCENT if i == 0 else NAVY_TITLE
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = BODY_COLOR
        pd.space_before = Pt(4)

    s10.notes_slide.notes_text_frame.text = (
        "Here are our core empirical findings on test writers W007 and W0011. "
        "Approach 1 (Baseline CRNN) won first place with 79.94% Character Accuracy and 25.85% Word-Level Accuracy, "
        "transcribing 10% of full lines perfectly. "
        "BiLSTM outperformed Transformer because sequential recurrence is the ideal inductive bias for handwriting. "
        "Transfer learning came third due to domain mismatch between ImageNet and binary pen strokes."
    )

    # =========================================================================
    # SLIDE 11: Learning Curves & Visual Convergence
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Convergence Dynamics: Overlaid Validation Learning Curves")

    # Embed overlaid curves image if exists
    img_curve = "slides/assets/overlaid_learning_curves.png"
    if os.path.exists(img_curve):
        s11.shapes.add_picture(img_curve, Inches(0.8), Inches(1.6), width=Inches(7.2))

    # Right Card: Observations
    add_card(s11, Inches(8.3), Inches(1.6), Inches(4.2), Inches(5.1))
    tb_obs = s11.shapes.add_textbox(Inches(8.5), Inches(1.8), Inches(3.8), Inches(4.7))
    tf_obs = tb_obs.text_frame
    tf_obs.word_wrap = True

    p = tf_obs.paragraphs[0]
    p.text = "Convergence Analysis"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_d = tf_obs.add_paragraph()
    p_d.text = (
        "• Blue Curve (A1 Baseline CRNN):\n"
        "  - Rapidly descended from 46% to ~10.9% validation CER\n"
        "  - Exhibited the lowest and most stable error trajectory\n\n"
        "• Green Curve (A3 Transformer):\n"
        "  - Descended to ~14.7% validation CER\n"
        "  - Stable but plateaued higher than BiLSTM\n\n"
        "• Orange Curve (A2 Transfer ResNet-18):\n"
        "  - Stabilized around ~18.7% validation CER\n"
        "  - Higher error floor due to capacity bloat\n\n"
        "• Regularization Impact:\n"
        "  - Zero validation spikes thanks to EMA (0.999) and Cosine warmup."
    )
    p_d.font.size = Pt(10.5)
    p_d.font.color.rgb = BODY_COLOR
    p_d.space_before = Pt(8)

    s11.notes_slide.notes_text_frame.text = (
        "Looking at the overlaid validation learning curves, we observe the training dynamics. "
        "A1 (blue) achieved the lowest and most stable validation trajectory, dropping smoothly from 46% down to 10.9%. "
        "A3 Transformer (green) converged reliably but plateaued higher. "
        "Notice the complete absence of validation spikes, proving that EMA decay and cosine warmup stabilized training."
    )

    # =========================================================================
    # SLIDE 12: Qualitative Error Analysis
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Qualitative Error Analysis: Successes & Linguistic Failure Modes")

    errors = [
        ("1. Successful Sentence Transcription", 
         "Ground Truth: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n"
         "A1 Prediction: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n\n"
         "• Outcome: 100% exact match across all 58 characters!\n"
         "• Successfully recovered subscript coeng ្រ, upper vowels ឹ, ើ, and dependent vowel ៀ without boundary drift.",
         SUCCESS_COLOR),

        ("2. Subscript Coeng Confusion",
         "Ground Truth: ក្ប (Base ក + Coeng ្ + Subscript ប)\n"
         "Model Output: ក (Subscript omitted)\n\n"
         "• Cause: In faint or hurried handwriting, subscript strokes have low pixel contrast.\n"
         "• Impact: The feature downsampling layer occasionally misses faint subscript ink, collapsing it into the base consonant.",
         PRIMARY_ACCENT),

        ("3. Visual Twin Character Ambiguity",
         "Ground Truth: ព្រឹកនោះ → Prediction: ព្រឹកនេះ (នោះ vs នេះ)\n"
         "Ground Truth: ព → Prediction: យ (Single vs double loop)\n\n"
         "• Cause: Cursive pen loops collapse in natural handwriting, making characters visually identical at 48px height.\n"
         "• Solution: Requires sentence-level language modeling for semantic disambiguation.",
         RGBColor(225, 29, 72)) # Rose 600
    ]

    for i, (title, desc, color) in enumerate(errors):
        c_left = Inches(0.8 + i * 4.0)
        add_card(s12, c_left, Inches(1.6), Inches(3.733), Inches(5.1))
        tb = s12.shapes.add_textbox(c_left + Inches(0.2), Inches(1.8), Inches(3.333), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = color
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = BODY_COLOR
        pd.space_before = Pt(8)

    s12.notes_slide.notes_text_frame.text = (
        "Qualitative error analysis revealed two primary failure modes. "
        "First, faint subscript coengs. When a writer scribbles lightly, the downsampled CNN sometimes misses the stroke. "
        "Second, visual twin characters like ព and យ where cursive handwriting collapses loops. "
        "However, on clear handwriting, A1 transcribes entire 58-character lines with 100% precision."
    )

    # =========================================================================
    # SLIDE 13: Live Inference Demo & Production Artifacts
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Live Inference Demo & Production Software Artifacts")

    # Left: Embed live demo output image
    img_demo = "slides/assets/live_demo_output.png"
    if os.path.exists(img_demo):
        s13.shapes.add_picture(img_demo, Inches(0.8), Inches(1.6), width=Inches(7.2))

    # Terminal command card below picture
    add_card(s13, Inches(0.8), Inches(3.8), Inches(7.2), Inches(2.9), bg_color=DARK_HEADER_BG, border_color=None)
    tb_cmd = s13.shapes.add_textbox(Inches(1.0), Inches(4.0), Inches(6.8), Inches(2.5))
    tf_cmd = tb_cmd.text_frame
    tf_cmd.word_wrap = True
    p = tf_cmd.paragraphs[0]
    p.text = "Live Terminal Inference Command:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(165, 180, 252)

    p_c = tf_cmd.add_paragraph()
    p_c.text = (
        "$ python scripts/demo_predict.py \\\n"
        "    --image data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png \\\n"
        "    --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\\n"
        "    --vocab-path data/metadata/char_to_idx.json \\\n"
        "    --approach baseline\n\n"
        "✓ Prediction: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់"
    )
    p_c.font.size = Pt(10)
    p_c.font.name = "Courier New"
    p_c.font.color.rgb = RGBColor(52, 211, 153) # Emerald 400
    p_c.space_before = Pt(4)

    # Right Card: Software Engineering Deliverables
    add_card(s13, Inches(8.3), Inches(1.6), Inches(4.2), Inches(5.1))
    tb_art = s13.shapes.add_textbox(Inches(8.5), Inches(1.8), Inches(3.8), Inches(4.7))
    tf_art = tb_art.text_frame
    tf_art.word_wrap = True
    p = tf_art.paragraphs[0]
    p.text = "Engineering Deliverables"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p_a = tf_art.add_paragraph()
    p_a.text = (
        "• Modular PyTorch Codebase:\n"
        "  - src/models/ (A1, A2, A3)\n"
        "  - src/dataset.py & collate_fn\n"
        "  - src/tokenizer.py & postprocessor.py\n\n"
        "• Automated Scanning Pipeline:\n"
        "  - src/scan_pipeline.py with 4-point homography deskewing\n\n"
        "• 100% Passing Unit Test Suite:\n"
        "  - 30 pytest unit tests passing cross-platform (macOS & Linux Colab)\n\n"
        "• Reproducibility:\n"
        "  - Fixed RNG seeds, checkpoints, and automated Colab notebook."
    )
    p_a.font.size = Pt(10.5)
    p_a.font.color.rgb = BODY_COLOR
    p_a.space_before = Pt(8)

    s13.notes_slide.notes_text_frame.text = (
        "In addition to deep learning modeling, this project delivered clean software engineering artifacts. "
        "A modular PyTorch codebase, a sheet cropping pipeline with perspective deskewing, and a 100% passing test suite. "
        "Our live demo CLI predicts unseen test lines in real time."
    )

    # =========================================================================
    # SLIDE 14: Conclusion & Future Horizons
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Conclusion & Future Directions")

    # Conclusion Card (Left)
    add_card(s14, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tb_c = s14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.7))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Key Project Conclusions"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_COLOR

    p_cd = tf_c.add_paragraph()
    p_cd.text = (
        "1. A1 Baseline CRNN is the Superior Architecture:\n"
        "   Achieved ~80% Character Accuracy and 25.85% Word-Level Accuracy on held-out writers while requiring only 8.49M parameters.\n\n"
        "2. Inductive Bias Beats Raw Attention:\n"
        "   BiLSTM's sequential recurrence is far more sample-efficient on limited handwriting data than Transformer self-attention.\n\n"
        "3. Curriculum Learning is Indispensable:\n"
        "   HarfBuzz OpenType synthetic pre-training completely eliminated CTC blank collapse and enabled rapid real-handwriting adaptation."
    )
    p_cd.font.size = Pt(11)
    p_cd.font.color.rgb = BODY_COLOR
    p_cd.space_before = Pt(10)

    # Future Work Card (Right)
    add_card(s14, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tb_f = s14.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.7))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True
    p = tf_f.paragraphs[0]
    p.text = "Limitations & Future Horizons"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_ACCENT

    p_fd = tf_f.add_paragraph()
    p_fd.text = (
        "• Limitations:\n"
        "  - Highly cursive or faded pencil strokes reduce subscript recall.\n"
        "  - Lack of a statistical language model for visual twin disambiguation (ព vs យ).\n\n"
        "• Future Work:\n"
        "  - Pre-trained Khmer Language Model (BERT/RoBERTa) for post-OCR contextual beam search decoding.\n"
        "  - Expanding dataset with multi-province handwritten exam papers.\n"
        "  - Mobile deployment with CoreML / ONNX Runtime for offline edge scanning.\n\n"
        "Thank you, Mr. Soklong HIM. I welcome your questions!"
    )
    p_fd.font.size = Pt(11)
    p_fd.font.color.rgb = BODY_COLOR
    p_fd.space_before = Pt(10)

    s14.notes_slide.notes_text_frame.text = (
        "To conclude: our comparative study demonstrated that Baseline CRNN with VGG and BiLSTM provides the most "
        "accurate and sample-efficient architecture for Khmer handwritten line recognition, reaching 80% character accuracy. "
        "Synthetic pre-training was essential to make real fine-tuning viable. "
        "Thank you very much, Mr. Soklong. I am now ready for your questions!"
    )

    out_path = "slides/Khmer_Handwritten_OCR_Presentation.pptx"
    prs.save(out_path)
    print(f"✓ Presentation saved successfully to {out_path}!")

if __name__ == "__main__":
    create_presentation()

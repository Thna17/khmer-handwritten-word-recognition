"""
Generate an executive, high-impact 16:9 PowerPoint presentation (.pptx)
designed for effortless presenting:
- Zero text clutter / no cognitive overload
- Generous breathing room & high-contrast dark navy / cyan aesthetic
- Embedded custom-generated visual infographics & real project charts
- Large KPI statistics & 1-line takeaways
- Full presenter notes attached to every slide

Output:
    slides/Khmer_Handwritten_OCR_Presentation.pptx
    /Users/macbookpro/Desktop/Khmer_Handwritten_OCR_Presentation.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Asset Paths
BG_IMG = "/Users/macbookpro/Documents/slide/bg.png"
ANATOMY_IMG = "slides/assets/khmer_script_anatomy.jpg"
BILSTM_TRANS_IMG = "slides/assets/bilstm_vs_transformer.jpg"
CHART_IMG = "slides/assets/comparison_chart.png"
CURVES_IMG = "slides/assets/overlaid_learning_curves.png"
DEMO_IMG = "slides/assets/live_demo_output.png"
TRAIN_LINE = "data/handwritten_external/W008/lines_processed/W008_P002_L006.png"
VAL_LINE = "data/handwritten_external/W009/lines_processed/W009_P002_L008.png"
TEST_LINE = "data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png"

# Colors
CARD_BG = RGBColor(10, 22, 48)            # Translucent deep navy
CARD_BORDER = RGBColor(30, 64, 175)       # Electric royal blue
CARD_BORDER_CYAN = RGBColor(56, 189, 248) # Cyan highlight
WHITE = RGBColor(255, 255, 255)
CYAN = RGBColor(56, 189, 248)
TEXT_MUTED = RGBColor(148, 163, 184)
TEXT_LIGHT = RGBColor(226, 232, 240)
PILL_BG = RGBColor(29, 78, 216)
GREEN = RGBColor(34, 197, 94)
AMBER = RGBColor(245, 158, 11)
RED = RGBColor(239, 68, 68)

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_bg(slide):
        if os.path.exists(BG_IMG):
            slide.shapes.add_picture(BG_IMG, 0, 0, width=prs.slide_width, height=prs.slide_height)

    def add_header(slide, badge_txt, title_w, title_c, subtitle=""):
        # Pill badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.42), Inches(2.8), Inches(0.36))
        badge.fill.solid()
        badge.fill.fore_color.rgb = PILL_BG
        badge.line.color.rgb = CYAN
        badge.line.width = Pt(1)
        tf = badge.text_frame
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = badge_txt
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

        # Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.88), Inches(11.7), Inches(0.85))
        tf_t = tb.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        r1 = p_t.add_run()
        r1.text = title_w + " "
        r1.font.size = Pt(28)
        r1.font.bold = True
        r1.font.color.rgb = WHITE

        r2 = p_t.add_run()
        r2.text = title_c
        r2.font.size = Pt(28)
        r2.font.bold = True
        r2.font.color.rgb = CYAN

        if subtitle:
            p_sub = tf_t.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED
            p_sub.space_before = Pt(3)

    def add_card(slide, left, top, width, height, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)
        return card

    # =========================================================================
    # SLIDE 1: Title (Executive & Clean)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1)

    # Big Title Box
    tb_t = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(7.2), Inches(3.2))
    tf1 = tb_t.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "FINAL YEAR PROJECT DEFENSE • DEEP LEARNING"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    p1 = tf1.add_paragraph()
    r = p1.add_run()
    r.text = "Khmer Handwritten\n"
    r.font.size = Pt(42)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r2 = p1.add_run()
    r2.text = "Line Recognition"
    r2.font.size = Pt(42)
    r2.font.bold = True
    r2.font.color.rgb = CYAN

    p2 = tf1.add_paragraph()
    p2.text = "An Empirical Comparison of Three CTC-Based Deep Learning Approaches"
    p2.font.size = Pt(15)
    p2.font.color.rgb = TEXT_LIGHT
    p2.space_before = Pt(8)

    # Author Card (Left Bottom)
    add_card(s1, Inches(0.8), Inches(5.1), Inches(6.8), Inches(1.7))
    tb_a = s1.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(6.4), Inches(1.5))
    tf_a = tb_a.text_frame
    pa1 = tf_a.paragraphs[0]
    pa1.text = "Hong Than Brathna  |  Software Engineering"
    pa1.font.size = Pt(16)
    pa1.font.bold = True
    pa1.font.color.rgb = WHITE

    pa2 = tf_a.add_paragraph()
    pa2.text = "Advisor / Lecturer: Mr. Soklong HIM  •  2026"
    pa2.font.size = Pt(13)
    pa2.font.color.rgb = CYAN
    pa2.space_before = Pt(3)

    pa3 = tf_a.add_paragraph()
    pa3.text = "Evaluation Protocol: Strict Writer-Disjoint Held-Out Benchmark (W007, W0011)"
    pa3.font.size = Pt(11)
    pa3.font.color.rgb = TEXT_MUTED
    pa3.space_before = Pt(3)

    # Right Side: 3-step pipeline flow cards
    cards_r = [
        ("1. Input Line", TEST_LINE, "Handwritten Line Image", True),
        ("2. Deep Model", None, "VGG / ResNet / Transformer", False),
        ("3. Unicode Text", None, "ព្រឹកនេះ នៅក្នុងបន្ទប់...", False)
    ]
    for i, (tag, img_p, label, is_img) in enumerate(cards_r):
        c_top = Inches(1.8) + i * Inches(1.75)
        add_card(s1, Inches(8.3), c_top, Inches(4.2), Inches(1.55))
        tb = s1.shapes.add_textbox(Inches(8.5), c_top + Inches(0.12), Inches(3.8), Inches(1.3))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = tag
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CYAN

        if is_img and img_p and os.path.exists(img_p):
            s1.shapes.add_picture(img_p, Inches(8.5), c_top + Inches(0.48), width=Inches(3.8))
        else:
            p_val = tf.add_paragraph()
            p_val.text = label
            p_val.font.size = Pt(14)
            p_val.font.bold = True
            p_val.font.color.rgb = WHITE
            p_val.space_before = Pt(8)

    s1.notes_slide.notes_text_frame.text = (
        "Good morning, respected lecturer Mr. Soklong HIM. Today I present my final project: "
        "Khmer Handwritten Line Recognition. We benchmarked three distinct deep learning architectures "
        "on strictly held-out unseen writers using a two-stage curriculum strategy."
    )

    # =========================================================================
    # SLIDE 2: Problem Statement (Visual Infographic)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2)
    add_header(s2, "02  PROBLEM STATEMENT", "Why is Khmer Handwritten OCR", "Uniquely Difficult?",
               "Khmer script violates standard Western OCR linear segmentation assumptions.")

    # Left: Embed custom generated script anatomy visual
    if os.path.exists(ANATOMY_IMG):
        add_card(s2, Inches(0.8), Inches(1.9), Inches(5.8), Inches(5.1), border_color=CARD_BORDER_CYAN)
        s2.shapes.add_picture(ANATOMY_IMG, Inches(0.9), Inches(2.0), width=Inches(5.6))

    # Right: 3 Clean Cards with high whitespace
    r_challenges = [
        ("1. Multi-Tier Vertical Stacking",
         "Characters stack across 3 tiers (Upper Vowel, Base, and Subscript Coeng).\nTraditional linear bounding-box segmentation destroys character meaning."),
        
        ("2. Continuous Script (Zero Spaces)",
         "Words flow seamlessly without whitespace boundaries.\nRequires temporal sequence alignment via CTC loss instead of word slicing."),
        
        ("3. High Inter-Writer Variability",
         "Cursive loop collapse creates visual twins (ព vs យ, ត vs ដ).\nStrict writer-disjoint splitting is required to prevent style memorization.")
    ]
    for i, (title, desc) in enumerate(r_challenges):
        c_top = Inches(1.9) + i * Inches(1.75)
        add_card(s2, Inches(6.9), c_top, Inches(5.633), Inches(1.55))
        tb = s2.shapes.add_textbox(Inches(7.1), c_top + Inches(0.12), Inches(5.2), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = CYAN

        p_b = tf.add_paragraph()
        p_b.text = desc
        p_b.font.size = Pt(10.5)
        p_b.font.color.rgb = TEXT_LIGHT
        p_b.space_before = Pt(3)

    s2.notes_slide.notes_text_frame.text = (
        "As seen in the anatomy diagram on the left, Khmer glyphs stack across three vertical tiers: "
        "upper vowels, base consonants, and subscript Coengs. Because Khmer text has no spaces between words, "
        "traditional character slicing fails completely. An end-to-end CTC sequence model is essential."
    )

    # =========================================================================
    # SLIDE 3: System Pipeline (Flowchart)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3)
    add_header(s3, "03  SYSTEM PIPELINE", "End-to-End", "Recognition Pipeline",
               "Transforms unconstrained handwritten line images into Unicode text.")

    steps_s3 = [
        ("1. Input", "Raw Line Image", "Variable width crop"),
        ("2. Normalize", "Height 48px", "Preserve aspect ratio"),
        ("3. CNN", "Feature Maps", "Asymmetric pool (2,1)"),
        ("4. Sequence", "BiLSTM / Transf.", "Temporal context"),
        ("5. CTC Head", "87 Classes", "Greedy decode"),
        ("6. Text", "Unicode Output", "NFC normalized")
    ]
    w_box = Inches(1.8)
    gap_box = Inches(0.18)
    for i, (num, t_top, t_sub) in enumerate(steps_s3):
        l = Inches(0.8) + i * (w_box + gap_box)
        add_card(s3, l, Inches(2.0), w_box, Inches(2.8))
        tb = s3.shapes.add_textbox(l + Inches(0.1), Inches(2.2), w_box - Inches(0.2), Inches(2.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = num
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = CYAN
        p0.alignment = PP_ALIGN.CENTER

        p1 = tf.add_paragraph()
        p1.text = t_top
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = WHITE
        p1.alignment = PP_ALIGN.CENTER
        p1.space_before = Pt(10)

        p2 = tf.add_paragraph()
        p2.text = t_sub
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(6)

    # 3 Summary Cards Below
    sum_cards = [
        ("Input Normalization", "Grayscale 1-channel, height 48px, horizontal max width 1600px with zero vertical squishing."),
        ("Unicode NFC Vocabulary", "86 total classes (33 consonants, 31 vowels/diacritics, 8 punctuation + Index 0 for CTC <blank>)."),
        ("Orthographic Rule Cleaner", "Deterministic rule cleaner fixes duplicate vowels, invalid coeng stacking, and diacritic ordering.")
    ]
    col_w3 = Inches(3.733)
    for i, (title, desc) in enumerate(sum_cards):
        l = Inches(0.8) + i * (col_w3 + Inches(0.266))
        add_card(s3, l, Inches(5.1), col_w3, Inches(1.9))
        tb = s3.shapes.add_textbox(l + Inches(0.15), Inches(5.25), col_w3 - Inches(0.3), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = CYAN

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_before = Pt(4)

    s3.notes_slide.notes_text_frame.text = (
        "Here is the complete recognition pipeline. Images are normalized to height 48 while preserving natural "
        "stroke aspect ratios. Asymmetric pooling in the CNN shrinks height to 1 while keeping horizontal resolution high. "
        "Sequence features pass to CTC loss for alignment, outputting clean NFC Unicode text."
    )

    # =========================================================================
    # SLIDE 4: Dataset & Writer Split (Big Numbers)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4)
    add_header(s4, "04  DATASET INTEGRITY", "Writer-Disjoint", "Evaluation Protocol",
               "Zero writer data leakage: test writers never appear in training or validation splits.")

    # 3 Big KPI Stat Cards
    kpis = [
        ("Training Set", "177", "Handwritten Lines", "7 Human Writers (W001-W006, W008)"),
        ("Validation Set", "50", "Handwritten Lines", "2 Human Writers (W009, W010)"),
        ("Held-Out Test Set", "50", "Handwritten Lines", "2 Unseen Writers (W007, W0011)")
    ]
    for i, (title, num, sub1, sub2) in enumerate(kpis):
        l = Inches(0.8) + i * Inches(2.9)
        add_card(s4, l, Inches(1.9), Inches(2.7), Inches(2.3))
        tb = s4.shapes.add_textbox(l + Inches(0.15), Inches(2.05), Inches(2.4), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(12)
        p0.font.bold = True
        p0.font.color.rgb = CYAN if i < 2 else GREEN

        p1 = tf.add_paragraph()
        p1.text = num
        p1.font.size = Pt(40)
        p1.font.bold = True
        p1.font.color.rgb = WHITE
        p1.space_before = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = f"{sub1}\n{sub2}"
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED

    # 4th Card: Academic Leakage Protection
    add_card(s4, Inches(9.5), Inches(1.9), Inches(3.033), Inches(2.3), border_color=CARD_BORDER_CYAN)
    tb_lk = s4.shapes.add_textbox(Inches(9.65), Inches(2.05), Inches(2.7), Inches(2.0))
    tf_lk = tb_lk.text_frame
    tf_lk.word_wrap = True
    p = tf_lk.paragraphs[0]
    p.text = "🛡️ Academic Rigor"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pd = tf_lk.add_paragraph()
    pd.text = (
        "● Random splitting causes memorization of pen pressure and style.\n"
        "● Our locked writer-disjoint split evaluates true generalization to unseen authors.\n"
        "● W007 & W0011 were strictly untouched during all training."
    )
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(4)

    # Bottom: 3 Actual Image Crops
    add_card(s4, Inches(0.8), Inches(4.5), Inches(11.733), Inches(2.5))
    tb_c_title = s4.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(11.3), Inches(0.35))
    tb_c_title.text_frame.paragraphs[0].text = "Verified Handwritten Line Crops from Each Split"
    tb_c_title.text_frame.paragraphs[0].font.size = Pt(12)
    tb_c_title.text_frame.paragraphs[0].font.bold = True
    tb_c_title.text_frame.paragraphs[0].font.color.rgb = CYAN

    crops_s4 = [
        (TRAIN_LINE, "Train Line (Writer W008)", Inches(1.0)),
        (VAL_LINE, "Validation Line (Writer W009)", Inches(4.9)),
        (TEST_LINE, "Held-Out Test Line (Writer W0011)", Inches(8.8))
    ]
    for img_p, lbl, left in crops_s4:
        if os.path.exists(img_p):
            s4.shapes.add_picture(img_p, left, Inches(5.1), width=Inches(3.5))
        tb_l = s4.shapes.add_textbox(left, Inches(6.45), Inches(3.5), Inches(0.35))
        p = tb_l.text_frame.paragraphs[0]
        p.text = lbl
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED
        p.alignment = PP_ALIGN.CENTER

    s4.notes_slide.notes_text_frame.text = (
        "To satisfy Mr. Soklong's rubric, we enforced a strict writer-disjoint partition. "
        "177 lines for training, 50 for validation, and 50 held out exclusively for testing from writers W007 and W0011. "
        "Random image splitting causes artificial accuracy inflation; our split tests true handwriting generalization."
    )

    # =========================================================================
    # SLIDE 5: Two-Stage Curriculum Training (Spacious 2-Stage Cards)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5)
    add_header(s5, "05  TRAINING STRATEGY", "Two-Stage", "Curriculum Strategy",
               "Pre-train on 10,000 synthetic samples to prevent CTC blank collapse, then fine-tune on real ink.")

    # Stage 1 Card
    add_card(s5, Inches(0.8), Inches(1.9), Inches(5.7), Inches(3.5))
    tb1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(5.3), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Stage 1: Synthetic Pre-training"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN

    p_stat1 = tf1.add_paragraph()
    p_stat1.text = "10,000 Synthetic Samples"
    p_stat1.font.size = Pt(22)
    p_stat1.font.bold = True
    p_stat1.font.color.rgb = WHITE
    p_stat1.space_before = Pt(4)

    p_d1 = tf1.add_paragraph()
    p_d1.text = (
        "● HarfBuzz OpenType font shaping across 8 distinct Khmer fonts.\n"
        "● Complete coverage of all 86 Unicode characters & stacked ligatures.\n"
        "● Cold-Start Solution: Prevents CTC loss from collapsing into blanks.\n"
        "● Achieved >98% Character Accuracy on synthetic validation."
    )
    p_d1.font.size = Pt(10.5)
    p_d1.font.color.rgb = TEXT_LIGHT
    p_d1.space_before = Pt(8)

    # Stage 2 Card
    add_card(s5, Inches(6.8), Inches(1.9), Inches(5.733), Inches(3.5), border_color=CARD_BORDER_CYAN)
    tb2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.3), Inches(3.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stage 2: Real Handwriting Fine-Tuning"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = GREEN

    p_stat2 = tf2.add_paragraph()
    p_stat2.text = "177 Real Lines + 3,000 Crop Mix"
    p_stat2.font.size = Pt(22)
    p_stat2.font.bold = True
    p_stat2.font.color.rgb = WHITE
    p_stat2.space_before = Pt(4)

    p_d2 = tf2.add_paragraph()
    p_d2.text = (
        "● Multi-scale Elastic & Perspective Distortions simulate natural writing.\n"
        "● 8x Real Ink Oversampling ensures real strokes drive ~40% of every epoch.\n"
        "● Synthetic Anchor Mixing (3,000 samples) prevents catastrophic forgetting.\n"
        "● Cosine Annealing with 3 Warmup Epochs + EMA (decay=0.999)."
    )
    p_d2.font.size = Pt(10.5)
    p_d2.font.color.rgb = TEXT_LIGHT
    p_d2.space_before = Pt(8)

    # Bottom Takeaway Card
    add_card(s5, Inches(0.8), Inches(5.6), Inches(11.733), Inches(1.4))
    tb_take = s5.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.3), Inches(1.2))
    tf_take = tb_take.text_frame
    tf_take.word_wrap = True
    p = tf_take.paragraphs[0]
    p.text = "🎯 Why Curriculum Training Succeeded"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pd = tf_take.add_paragraph()
    pd.text = (
        "Training directly on 177 real lines from scratch yielded 0.14 train loss but 1.42 val loss (severe memorization). "
        "Pre-training on HarfBuzz synthetic fonts established robust character alignment first, allowing fine-tuning to focus purely on stroke adaptation."
    )
    pd.font.size = Pt(11)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(3)

    s5.notes_slide.notes_text_frame.text = (
        "Our two-stage training strategy was critical. Training from scratch on 177 lines caused catastrophic overfitting. "
        "In Stage 1, we pre-trained on 10,000 synthetic samples to master character glyphs. In Stage 2, we fine-tuned on real handwriting "
        "with elastic augmentations, 8x ink oversampling, and anchor mixing to prevent forgetting."
    )

    # =========================================================================
    # SLIDE 6: Approach 1 — Baseline CRNN (Clean Breakdown)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6)
    add_header(s6, "06  APPROACH 1", "Baseline CRNN", "(VGG + BiLSTM + CTC)",
               "Lightweight, sample-efficient baseline with specialized asymmetric downsampling.")

    # 3 Column Cards
    add_card(s6, Inches(0.8), Inches(1.9), Inches(4.3), Inches(4.9))
    tb1_6 = s6.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(3.9), Inches(4.5))
    tf = tb1_6.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CNN Feature Extractor"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    p_b = tf.add_paragraph()
    p_b.text = (
        "● Custom 7-Layer VGG CNN:\n"
        "  - 64 → 128 → 256 → 512 channels\n"
        "  - BatchNorm & ReLU activations\n\n"
        "● Asymmetric MaxPool (2, 1):\n"
        "  - Layers 3 & 4 pool vertical height aggressively from 48 down to 1.\n"
        "  - Preserves horizontal width W' = W/4.\n\n"
        "● Eliminates vertical ambiguity, mapping subscript coengs directly into sequential feature columns."
    )
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = TEXT_LIGHT
    p_b.space_before = Pt(8)

    # Middle Card: Model Size & Recurrence
    add_card(s6, Inches(5.3), Inches(1.9), Inches(3.0), Inches(4.9), border_color=CARD_BORDER_CYAN)
    tb2_6 = s6.shapes.add_textbox(Inches(5.45), Inches(2.1), Inches(2.7), Inches(4.5))
    tf = tb2_6.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Model Capacity"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.alignment = PP_ALIGN.CENTER

    p_num = tf.add_paragraph()
    p_num.text = "8.49M"
    p_num.font.size = Pt(44)
    p_num.font.bold = True
    p_num.font.color.rgb = WHITE
    p_num.alignment = PP_ALIGN.CENTER
    p_num.space_before = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "Trainable Parameters\n\n2-Layer BiLSTM\n(Hidden Size: 256, Dropout: 0.2)\n\nLinear CTC Head (87 Classes)"
    p_sub.font.size = Pt(11)
    p_sub.font.color.rgb = TEXT_LIGHT
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.space_before = Pt(8)

    # Right Card: Strengths
    add_card(s6, Inches(8.5), Inches(1.9), Inches(4.033), Inches(4.9))
    tb3_6 = s6.shapes.add_textbox(Inches(8.7), Inches(2.1), Inches(3.6), Inches(4.5))
    tf = tb3_6.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Why A1 Won 1st Place"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN

    p_b = tf.add_paragraph()
    p_b.text = (
        "✓ Innate Recurrence Inductive Bias:\n"
        "  BiLSTM naturally processes text strictly left-to-right, matching human handwriting order.\n\n"
        "✓ High Sample Efficiency:\n"
        "  Compact parameter footprint avoids overfitting on small real datasets.\n\n"
        "✓ Zero Domain Mismatch:\n"
        "  All 8.49M parameters were trained specifically on Khmer glyphs, not natural photos."
    )
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = TEXT_LIGHT
    p_b.space_before = Pt(8)

    s6.notes_slide.notes_text_frame.text = (
        "Approach 1 is our baseline CRNN. It uses 7 VGG convolution layers with asymmetric pooling to squash height to 1 "
        "while keeping width high. The 2-layer BiLSTM models sequential context. With 8.49M parameters, it achieved "
        "the best balance of sample efficiency and accuracy."
    )

    # =========================================================================
    # SLIDE 7: Approach 2 — Transfer Learning (ResNet-18)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7)
    add_header(s7, "07  APPROACH 2", "ResNet-18", "Transfer Learning",
               "Evaluating pre-trained ImageNet representations with differential learning rates.")

    add_card(s7, Inches(0.8), Inches(1.9), Inches(4.3), Inches(4.9))
    tb1_7 = s7.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(3.9), Inches(4.5))
    tf = tb1_7.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Adaptation"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    p_b = tf.add_paragraph()
    p_b.text = (
        "● Pre-trained ResNet-18:\n"
        "  - Initialized with ImageNet-1K weights.\n"
        "  - Input grayscale replicated to 3 channels.\n\n"
        "● Resolution Modifications:\n"
        "  - Early Conv1 and MaxPool horizontal strides changed from 2 to 1.\n"
        "  - Asymmetric bridge convolution (3, 1) flattens height to 1.\n\n"
        "● Constant Sequence Head:\n"
        "  - Same 2-layer BiLSTM & CTC head as A1 to isolate the CNN backbone variable."
    )
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = TEXT_LIGHT
    p_b.space_before = Pt(8)

    # Middle Card: 30.95M Parameters
    add_card(s7, Inches(5.3), Inches(1.9), Inches(3.0), Inches(4.9))
    tb2_7 = s7.shapes.add_textbox(Inches(5.45), Inches(2.1), Inches(2.7), Inches(4.5))
    tf = tb2_7.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Model Capacity"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.alignment = PP_ALIGN.CENTER

    p_num = tf.add_paragraph()
    p_num.text = "30.95M"
    p_num.font.size = Pt(44)
    p_num.font.bold = True
    p_num.font.color.rgb = WHITE
    p_num.alignment = PP_ALIGN.CENTER
    p_num.space_before = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "Trainable Parameters\n\n(3.6x larger than A1)\n\nHigh capacity creates vulnerability to small-sample overfitting."
    p_sub.font.size = Pt(11)
    p_sub.font.color.rgb = TEXT_LIGHT
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.space_before = Pt(8)

    # Right Card: Training Strategy & Scientific Insight
    add_card(s7, Inches(8.5), Inches(1.9), Inches(4.033), Inches(4.9))
    tb3_7 = s7.shapes.add_textbox(Inches(8.7), Inches(2.1), Inches(3.6), Inches(4.5))
    tf = tb3_7.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Training Strategy & Insights"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    p_b = tf.add_paragraph()
    p_b.text = (
        "● Differential Learning Rates:\n"
        "  - Backbone LR multiplier: 0.25x\n"
        "  - Head LR multiplier: 1.0x\n"
        "  - 3-Epoch initial backbone freeze.\n\n"
        "● Why It Ranked 3rd (25.78% CER):\n"
        "  - Domain Mismatch: ImageNet filters model natural 3D textures, not 1-channel binary ink strokes.\n"
        "  - Parameter Bloat: 31M parameters overfitted the 177 real training lines."
    )
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = TEXT_LIGHT
    p_b.space_before = Pt(8)

    s7.notes_slide.notes_text_frame.text = (
        "In Approach 2, we tested transfer learning with ResNet-18. We adapted horizontal strides and trained with "
        "differential learning rates. It achieved 74.22% character accuracy, but came third because ImageNet natural photo "
        "features had a domain shift compared to binary handwriting, and its 31M parameters were prone to overfitting."
    )

    # =========================================================================
    # SLIDE 8: Approach 3 — Transformer Sequence Encoder (Visual Infographic)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8)
    add_header(s8, "08  APPROACH 3", "Transformer", "Sequence Encoder",
               "Evaluating multi-head self-attention against recurrent sequence modeling.")

    # Left: Embed custom generated BiLSTM vs Transformer visual!
    if os.path.exists(BILSTM_TRANS_IMG):
        add_card(s8, Inches(0.8), Inches(1.9), Inches(5.8), Inches(5.1), border_color=CARD_BORDER_CYAN)
        s8.shapes.add_picture(BILSTM_TRANS_IMG, Inches(0.9), Inches(2.0), width=Inches(5.6))

    # Right: 3 Clean Summary Cards
    trans_cards = [
        ("Architecture Specifications",
         "● CNN Backbone: Identical VGG to A1 (held constant).\n"
         "● 1D Sinusoidal Positional Encodings inject token order.\n"
         "● 4-Layer Transformer Encoder with 8 Multi-Head Attention heads.\n"
         "● Total Parameters: 9.54 Million."),

        ("Theoretical Advantage: Global Context",
         "● Computes all-to-all token attention in O(1) sequential steps.\n"
         "● Direct long-range receptive field across wide line crops.\n"
         "● Highly parallelizable training on GPU hardware."),

        ("Why BiLSTM Won on Real Handwriting",
         "● Inductive Bias Trade-Off: Transformers lack sequential continuity assumptions.\n"
         "● Sample Hunger: Transformers require vast datasets to learn sequence order.\n"
         "● On 177 real lines, BiLSTM's localized recurrence was more sample-efficient.")
    ]
    for i, (title, desc) in enumerate(trans_cards):
        c_top = Inches(1.9) + i * Inches(1.75)
        add_card(s8, Inches(6.9), c_top, Inches(5.633), Inches(1.55))
        tb = s8.shapes.add_textbox(Inches(7.1), c_top + Inches(0.12), Inches(5.2), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = CYAN

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_before = Pt(3)

    s8.notes_slide.notes_text_frame.text = (
        "In Approach 3, we replaced the BiLSTM with a 4-layer Transformer Encoder with 8 attention heads. "
        "As illustrated on the left, self-attention computes an all-to-all attention matrix. While it captures global context, "
        "it lacks the localized recurrence inductive bias that BiLSTMs possess. On a small real dataset, BiLSTM was more sample efficient."
    )

    # =========================================================================
    # SLIDE 9: Results & Benchmark Table (THE MASTER SLIDE)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9)
    add_header(s9, "09  RESULTS & BENCHMARK", "Empirical", "Performance Evaluation",
               "Evaluated on held-out test writers (W007, W0011) under strict writer-disjoint isolation.")

    # 3 Big Winner Metric Cards (Top)
    win_metrics = [
        ("Character Accuracy (A1 Winner)", "79.94%", "(~80% Accuracy on unseen writers)", GREEN),
        ("Word-Level Accuracy (A1 Winner)", "25.85%", "(Word F1 Score: 27.54%)", CYAN),
        ("Lowest Test CER (A1 Winner)", "20.06%", "(80% error reduction vs baseline)", WHITE)
    ]
    for i, (title, val, sub, col) in enumerate(win_metrics):
        l = Inches(0.8) + i * Inches(3.95)
        add_card(s9, l, Inches(1.9), Inches(3.8), Inches(1.5))
        tb = s9.shapes.add_textbox(l + Inches(0.15), Inches(2.0), Inches(3.5), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = TEXT_MUTED

        p1 = tf.add_paragraph()
        p1.text = val
        p1.font.size = Pt(28)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.space_before = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_LIGHT

    # Middle: Real Results Table
    tbl_shape = s9.shapes.add_table(4, 7, Inches(0.8), Inches(3.6), Inches(11.733), Inches(2.0))
    table = tbl_shape.table
    col_widths = [Inches(0.8), Inches(3.4), Inches(1.3), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.733)]
    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    headers = ["Rank", "Model Architecture", "Params", "Test CER ↓", "CharAcc ↑", "WordAcc ↑", "Exact Line ↑"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PILL_BG
        cell.text_frame.paragraphs[0].font.size = Pt(10)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = WHITE

    rows = [
        ("🥇 1st", "A1: Baseline CRNN (VGG + BiLSTM)", "8.49 M", "20.06%", "79.94%", "25.85%", "10.00%"),
        ("🥈 2nd", "A3: Transformer Sequence Encoder", "9.54 M", "22.92%", "77.08%", "19.73%", "2.00%"),
        ("🥉 3rd", "A2: Transfer Learning (ResNet-18)", "30.95 M", "25.78%", "74.22%", "12.24%", "0.00%")
    ]
    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(10)
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(16, 42, 86)
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = CYAN

    # Bottom Prediction Showcase
    add_card(s9, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.3), border_color=CARD_BORDER_CYAN)
    tb_pred = s9.shapes.add_textbox(Inches(1.0), Inches(5.88), Inches(11.3), Inches(1.1))
    tf_p = tb_pred.text_frame
    tf_p.word_wrap = True
    p0 = tf_p.paragraphs[0]
    p0.text = "Unseen Test Writer (W0011) Live Prediction:"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    p1 = tf_p.add_paragraph()
    r = p1.add_run()
    r.text = "Ground Truth & Model Prediction:  "
    r.font.size = Pt(11)
    r.font.color.rgb = WHITE
    r2 = p1.add_run()
    r2.text = "ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់"
    r2.font.size = Pt(14)
    r2.font.bold = True
    r2.font.color.rgb = GREEN

    r3 = p1.add_run()
    r3.text = "   [✓ 100% Exact Character Match!]"
    r3.font.size = Pt(11)
    r3.font.bold = True
    r3.font.color.rgb = CYAN

    s9.notes_slide.notes_text_frame.text = (
        "Here are our official benchmark results on test writers W007 and W0011. "
        "A1 Baseline CRNN achieved first place with 79.94% Character Accuracy, 25.85% Word-Level Accuracy, "
        "and 10% exact whole-line match. A3 Transformer placed second, and A2 ResNet-18 placed third. "
        "Our model accurately read the full unseen test sentence shown at the bottom."
    )

    # =========================================================================
    # SLIDE 10: Performance Analysis & Charts (Visual Side-by-Side)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10)
    add_header(s10, "10  CONVERGENCE & ACCURACY", "Visual", "Learning Dynamics",
               "Validation learning curves and test accuracy comparison charts from Google Drive.")

    # Left: Overlaid Learning Curves from Drive
    if os.path.exists(CURVES_IMG):
        add_card(s10, Inches(0.8), Inches(1.9), Inches(6.8), Inches(3.7))
        s10.shapes.add_picture(CURVES_IMG, Inches(0.9), Inches(2.0), width=Inches(6.6))

    # Right: Comparison Bar Chart from Drive
    if os.path.exists(CHART_IMG):
        add_card(s10, Inches(7.8), Inches(1.9), Inches(4.733), Inches(3.7))
        s10.shapes.add_picture(CHART_IMG, Inches(7.9), Inches(2.0), width=Inches(4.5))

    # Bottom Observations Card
    add_card(s10, Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.3))
    tb_c_obs = s10.shapes.add_textbox(Inches(1.0), Inches(5.88), Inches(11.3), Inches(1.1))
    tf_co = tb_c_obs.text_frame
    tf_co.word_wrap = True
    p = tf_co.paragraphs[0]
    p.text = "Training Stability & Convergence Insights"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pd = tf_co.add_paragraph()
    pd.text = (
        "● Blue Trajectory (A1): Steepest descent and lowest validation floor (10.9% val CER), proving high sample efficiency.\n"
        "● Regularization Success: Zero validation spikes across all 65 epochs thanks to Cosine Warmup and EMA decay (0.999).\n"
        "● Error Reduction: Character Error Rate dropped from 98% (untrained) to ~20%, an 80% error reduction."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(3)

    s10.notes_slide.notes_text_frame.text = (
        "Looking at the validation curves on the left, A1 (blue) descended smoothly to the lowest error floor. "
        "The comparison bar chart on the right confirms A1's superiority in both CER and Word-Level Accuracy. "
        "EMA decay and cosine warmup completely stabilized training across all models."
    )

    # =========================================================================
    # SLIDE 11: Qualitative Error Analysis (3 Clean Cards)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11)
    add_header(s11, "11  QUALITATIVE ANALYSIS", "Recognition Examples &", "Error Modes",
               "Analyzing successful transcriptions and linguistic failure cases on unseen handwriting.")

    col3_w = Inches(3.733)
    # Card 1: Correct
    add_card(s11, Inches(0.8), Inches(1.9), col3_w, Inches(4.2), border_color=GREEN)
    tb1 = s11.shapes.add_textbox(Inches(1.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.9))
    tf = tb1.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "✓ Exact Sentence Match"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN

    pd = tf.add_paragraph()
    pd.text = (
        "Ground Truth:\nព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n\n"
        "Prediction:\nព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n\n"
        "● 100% exact match across all 58 characters.\n"
        "● Successfully recovers subscript Coeng ្រ, upper vowels ឹ, ើ, and dependent vowel ៀ without drift."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(6)

    # Card 2: Minor Coeng Omission
    add_card(s11, Inches(4.8), Inches(1.9), col3_w, Inches(4.2), border_color=AMBER)
    tb2 = s11.shapes.add_textbox(Inches(5.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.9))
    tf = tb2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚠️ Subscript Coeng Omission"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = AMBER

    pd = tf.add_paragraph()
    pd.text = (
        "Ground Truth:  ក្ប (ក + ្ + ប)\n"
        "Prediction:    ក (Subscript omitted)\n\n"
        "● Failure Mechanism:\n"
        "  When writers use faint ink or hurried pen strokes, subscripts lack pixel contrast.\n\n"
        "● Downsampling layer misses the faint stroke, collapsing it into the base consonant."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(6)

    # Card 3: Visual Twins
    add_card(s11, Inches(8.8), Inches(1.9), col3_w, Inches(4.2), border_color=RED)
    tb3 = s11.shapes.add_textbox(Inches(9.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.9))
    tf = tb3.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "✗ Visual Twin Loop Collapse"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RED

    pd = tf.add_paragraph()
    pd.text = (
        "Ground Truth: ព  →  Prediction: យ\n"
        "Ground Truth: នោះ → Prediction: នេះ\n\n"
        "● Failure Mechanism:\n"
        "  Cursive loops collapse into identical oval shapes at 48px height.\n\n"
        "● Solution: Requires external language model for semantic disambiguation."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(6)

    # Bottom Key Observation
    add_card(s11, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.9))
    tb_ko = s11.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.3), Inches(0.8))
    p = tb_ko.text_frame.paragraphs[0]
    p.text = "💡 Linguistic Finding: Errors are concentrated on faint subscript strokes and cursive twin loops, not base character confusion."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    s11.notes_slide.notes_text_frame.text = (
        "Qualitative error analysis revealed two primary failure modes: faint subscript coengs and visual twin characters. "
        "However, on well-written handwriting, the model achieves 100% precision across complete sentences."
    )

    # =========================================================================
    # SLIDE 12: Live Demo & Engineering Deliverables
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12)
    add_header(s12, "12  LIVE DEMO & ARTIFACTS", "Real-Time Inference", "& Production Software",
               "End-to-end Python CLI and production engineering pipeline.")

    # Left: Embed Live Demo Output Image
    if os.path.exists(DEMO_IMG):
        add_card(s12, Inches(0.8), Inches(1.9), Inches(6.8), Inches(2.2))
        s12.shapes.add_picture(DEMO_IMG, Inches(0.9), Inches(2.0), width=Inches(6.6))

    # Command Box Below Image
    add_card(s12, Inches(0.8), Inches(4.3), Inches(6.8), Inches(2.8))
    tb_cmd = s12.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(6.4), Inches(2.5))
    tf_c = tb_cmd.text_frame
    tf_c.word_wrap = True
    p0 = tf_c.paragraphs[0]
    p0.text = "Single-Command Terminal CLI:"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    p1 = tf_c.add_paragraph()
    p1.text = (
        "$ python scripts/demo_predict.py \\\n"
        "    --image data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png \\\n"
        "    --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\\n"
        "    --vocab-path data/metadata/char_to_idx.json \\\n"
        "    --approach baseline\n\n"
        "✓ Output: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់"
    )
    p1.font.size = Pt(10)
    p1.font.name = "Courier New"
    p1.font.color.rgb = GREEN
    p1.space_before = Pt(4)

    # Right: Software Engineering Deliverables
    add_card(s12, Inches(7.8), Inches(1.9), Inches(4.733), Inches(5.2))
    tb_art = s12.shapes.add_textbox(Inches(8.0), Inches(2.1), Inches(4.3), Inches(4.7))
    tf_a = tb_art.text_frame
    tf_a.word_wrap = True
    p0 = tf_a.paragraphs[0]
    p0.text = "Software Engineering Deliverables"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    pd = tf_a.add_paragraph()
    pd.text = (
        "● Modular PyTorch Architecture:\n"
        "  - src/models/ (A1, A2, A3)\n"
        "  - src/dataset.py & dynamic white padding collate_fn\n"
        "  - src/tokenizer.py & postprocessor.py\n\n"
        "● Automated Scanning Pipeline:\n"
        "  - src/scan_pipeline.py with 4-point homography deskewing\n\n"
        "● 100% Passing Unit Test Suite:\n"
        "  - 30 pytest tests passing across macOS and Linux Colab\n\n"
        "● Complete Reproducibility:\n"
        "  - Fixed RNG seeds, checkpoints, and automated Colab runner."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(8)

    s12.notes_slide.notes_text_frame.text = (
        "In addition to deep learning models, we delivered production software engineering artifacts: "
        "a modular codebase, 30 passing unit tests, an automated scanning pipeline, and a live prediction CLI."
    )

    # =========================================================================
    # SLIDE 13: Conclusion & Future Work (Spacious 3-Column)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13)
    add_header(s13, "13  CONCLUSION & HORIZONS", "Conclusion &", "Future Directions",
               "Summary of empirical insights, project limitations, and future research paths.")

    col13_w = Inches(3.733)
    # Col 1: Key Findings
    add_card(s13, Inches(0.8), Inches(1.9), col13_w, Inches(4.0))
    tb1 = s13.shapes.add_textbox(Inches(1.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "🏆 Key Findings"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pd = tf1.add_paragraph()
    pd.text = (
        "● A1 CRNN is the Winner: Reached 79.94% CharAcc and 25.85% WordAcc with only 8.49M params.\n\n"
        "● BiLSTM > Transformer: Sequential recurrence inductive bias provides superior sample efficiency on modest handwriting data.\n\n"
        "● Synthetic Curriculum was Critical: HarfBuzz pre-training eliminated CTC blank collapse."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(8)

    # Col 2: Limitations
    add_card(s13, Inches(4.8), Inches(1.9), col13_w, Inches(4.0))
    tb2 = s13.shapes.add_textbox(Inches(5.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "⚠️ Limitations"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = AMBER

    pd = tf2.add_paragraph()
    pd.text = (
        "● Limited Human Writers: 11 writers; test set does not cover all regional cursive variants.\n\n"
        "● Low Contrast Strokes: Faint pencil strokes can cause subscript deletions.\n\n"
        "● Absence of Language Model: Cannot disambiguate visual twin characters via semantic context."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(8)

    # Col 3: Future Work
    add_card(s13, Inches(8.8), Inches(1.9), col13_w, Inches(4.0))
    tb3 = s13.shapes.add_textbox(Inches(9.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "💡 Future Work"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN

    pd = tf3.add_paragraph()
    pd.text = (
        "● Expanded Multi-Province Dataset: Collect writing samples across high schools nationwide.\n\n"
        "● Khmer RoBERTa Language Model: Integrate beam search decoding for post-OCR semantic correction.\n\n"
        "● Mobile On-Device Deployment: Export model via ONNX/CoreML for real-time offline phone camera scanning."
    )
    pd.font.size = Pt(10.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(8)

    # Bottom Final Takeaway Banner
    add_card(s13, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.1))
    tb_f = s13.shapes.add_textbox(Inches(1.0), Inches(6.12), Inches(11.3), Inches(0.9))
    p = tb_f.text_frame.paragraphs[0]
    p.text = "🎯 Final Takeaway: Baseline CRNN with VGG and BiLSTM achieved ~80% Character Accuracy on unseen human writers, proving that localized recurrence and synthetic curriculum pre-training are the optimal foundation for Khmer handwriting OCR."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    s13.notes_slide.notes_text_frame.text = (
        "To conclude: Baseline CRNN with VGG and BiLSTM provides the most accurate and sample-efficient architecture "
        "for Khmer handwritten line recognition. Thank you very much, Mr. Soklong HIM. I am now ready for your questions!"
    )

    # =========================================================================
    # SLIDE 14: References & Acknowledgements (Clean & Professional)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_bg(s14)
    add_header(s14, "14  REFERENCES & THANKS", "References &", "Acknowledgements",
               "Academic citations and acknowledgement of support.")

    # Left: References
    add_card(s14, Inches(0.8), Inches(1.9), Inches(5.7), Inches(4.0))
    tb_r = s14.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(5.3), Inches(3.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p0 = tf_r.paragraphs[0]
    p0.text = "📖 Academic References"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    pd = tf_r.add_paragraph()
    pd.text = (
        "1. Graves, A., et al. (2006). Connectionist Temporal Classification (CTC): Labelling Unsegmented Sequence Data. ICML.\n\n"
        "2. Shi, B., Bai, X., & Yao, C. (2016). An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN). IEEE TPAMI.\n\n"
        "3. He, K., et al. (2016). Deep Residual Learning for Image Recognition (ResNet). CVPR.\n\n"
        "4. Vaswani, A., et al. (2017). Attention Is All You Need (Transformer). NeurIPS.\n\n"
        "5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR."
    )
    pd.font.size = Pt(9.5)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(6)

    # Right: Acknowledgements
    add_card(s14, Inches(6.8), Inches(1.9), Inches(5.733), Inches(4.0))
    tb_ack = s14.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.3), Inches(3.7))
    tf_ack = tb_ack.text_frame
    tf_ack.word_wrap = True
    p0 = tf_ack.paragraphs[0]
    p0.text = "🤝 Acknowledgements"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = CYAN

    pd = tf_ack.add_paragraph()
    pd.text = (
        "● Academic Guidance:\n"
        "  Sincere gratitude to lecturer Mr. Soklong HIM for continuous mentorship, technical rigor, and feedback.\n\n"
        "● Open-Source Community:\n"
        "  PyTorch team, Google Colab, and HarfBuzz for high-performance deep learning and OpenType shaping engines.\n\n"
        "● Dataset Contributors:\n"
        "  Thank you to all human writers and contributors who provided handwritten samples for this research."
    )
    pd.font.size = Pt(10)
    pd.font.color.rgb = TEXT_LIGHT
    pd.space_before = Pt(6)

    # Bottom Banner: Thank You
    add_card(s14, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.1))
    tb_ty = s14.shapes.add_textbox(Inches(1.0), Inches(6.15), Inches(11.3), Inches(0.8))
    tf_ty = tb_ty.text_frame
    tf_ty.word_wrap = True
    p = tf_ty.paragraphs[0]
    r1 = p.add_run()
    r1.text = "🚀 Thank You!  "
    r1.font.size = Pt(18)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    r2 = p.add_run()
    r2.text = "|  Thank you for your time and attention. I welcome your questions and discussion."
    r2.font.size = Pt(12)
    r2.font.color.rgb = CYAN

    s14.notes_slide.notes_text_frame.text = (
        "Thank you very much, Mr. Soklong HIM. I am now ready for your questions!"
    )

    out1 = "slides/Khmer_Handwritten_OCR_Presentation.pptx"
    out2 = "/Users/macbookpro/Desktop/Khmer_Handwritten_OCR_Presentation.pptx"
    prs.save(out1)
    prs.save(out2)
    print(f"✓ Clean, uncluttered, professional presentation saved to {out1} and {out2}!")

if __name__ == "__main__":
    build_presentation()

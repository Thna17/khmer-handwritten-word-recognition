"""
Generate a professional 16:9 PowerPoint presentation (.pptx) that matches
the exact dark-mode, glowing-cyan aesthetic of the user's reference images
in /Users/macbookpro/Documents/slide (90%+ visual similarity), while injecting
the real empirical results, charts, and Khmer handwriting text from our project.

Output:
    slides/Khmer_Handwritten_OCR_Presentation.pptx
    /Users/macbookpro/Desktop/Khmer_Handwritten_OCR_Presentation.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BG_IMG = "/Users/macbookpro/Documents/slide/bg.png"
CHART_IMG = "slides/assets/comparison_chart.png"
CURVES_IMG = "slides/assets/overlaid_learning_curves.png"
DEMO_IMG = "slides/assets/live_demo_output.png"
TRAIN_LINE = "data/handwritten_external/W008/lines_processed/W008_P002_L006.png"
VAL_LINE = "data/handwritten_external/W009/lines_processed/W009_P002_L008.png"
TEST_LINE = "data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png"

# Color Palette matching reference images
CARD_BG = RGBColor(11, 25, 54)          # Deep translucent navy
CARD_BORDER = RGBColor(30, 64, 175)     # Glowing blue border (Blue 700)
CARD_BORDER_BRIGHT = RGBColor(37, 99, 235) # Bright electric blue
WHITE = RGBColor(255, 255, 255)
CYAN_ACCENT = RGBColor(56, 189, 248)    # Sky 400
TEXT_GRAY = RGBColor(148, 163, 184)     # Slate 400
TEXT_OFFWHITE = RGBColor(203, 213, 225) # Slate 300
PILL_BG = RGBColor(29, 78, 216)         # Blue 700
PILL_BORDER = RGBColor(56, 189, 248)    # Cyan
BADGE_BG = RGBColor(37, 99, 235)        # Electric blue
GREEN_ACCENT = RGBColor(34, 197, 94)    # Green 500
ORANGE_ACCENT = RGBColor(245, 158, 11)  # Amber 500
RED_ACCENT = RGBColor(239, 68, 68)      # Red 500

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_slide_bg(slide):
        if os.path.exists(BG_IMG):
            slide.shapes.add_picture(BG_IMG, 0, 0, width=prs.slide_width, height=prs.slide_height)

    def add_top_badge_and_header(slide, badge_text, title_white, title_cyan, subtitle=""):
        # Top Pill Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(2.6), Inches(0.38))
        badge.fill.solid()
        badge.fill.fore_color.rgb = PILL_BG
        badge.line.color.rgb = PILL_BORDER
        badge.line.width = Pt(1)
        tf_b = badge.text_frame
        tf_b.margin_top = tf_b.margin_bottom = tf_b.margin_left = tf_b.margin_right = 0
        p_b = tf_b.paragraphs[0]
        p_b.text = badge_text
        p_b.font.size = Pt(10)
        p_b.font.bold = True
        p_b.font.color.rgb = WHITE
        p_b.font.name = "Arial"
        p_b.alignment = PP_ALIGN.CENTER

        # Title Textbox
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = title_white + " "
        r1.font.size = Pt(28)
        r1.font.bold = True
        r1.font.color.rgb = WHITE
        r1.font.name = "Arial"

        r2 = p.add_run()
        r2.text = title_cyan
        r2.font.size = Pt(28)
        r2.font.bold = True
        r2.font.color.rgb = CYAN_ACCENT
        r2.font.name = "Arial"

        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_GRAY
            p_sub.font.name = "Arial"
            p_sub.space_before = Pt(4)

    def add_custom_card(slide, left, top, width, height, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)
        return card

    def add_num_badge(slide, left, top, num_str, bg_color=BADGE_BG):
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, Inches(0.52), Inches(0.52))
        badge.fill.solid()
        badge.fill.fore_color.rgb = bg_color
        badge.line.fill.background()
        tf = badge.text_frame
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = num_str
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Arial"
        p.alignment = PP_ALIGN.CENTER
        return badge

    # =========================================================================
    # SLIDE 1: Title Slide (Matches 1.png)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s1)

    # Top pill
    badge1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), Inches(3.2), Inches(0.45))
    badge1.fill.solid()
    badge1.fill.fore_color.rgb = PILL_BG
    badge1.line.color.rgb = PILL_BORDER
    badge1.line.width = Pt(1)
    tf_b1 = badge1.text_frame
    p_b1 = tf_b1.paragraphs[0]
    p_b1.text = "FINAL PROJECT • DEEP LEARNING"
    p_b1.font.size = Pt(11)
    p_b1.font.bold = True
    p_b1.font.color.rgb = WHITE
    p_b1.alignment = PP_ALIGN.CENTER

    # Big Title
    tb_title = s1.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(6.8), Inches(2.5))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    p1 = tf_t.paragraphs[0]
    r1 = p1.add_run()
    r1.text = "Khmer Handwritten\n"
    r1.font.size = Pt(40)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    r2 = p1.add_run()
    r2.text = "Line Recognition"
    r2.font.size = Pt(40)
    r2.font.bold = True
    r2.font.color.rgb = CYAN_ACCENT

    p2 = tf_t.add_paragraph()
    p2.text = "A Comparative Study of Three CTC-Based Deep Learning Approaches"
    p2.font.size = Pt(15)
    p2.font.color.rgb = TEXT_OFFWHITE
    p2.space_before = Pt(12)

    # Accent blue horizontal bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.7), Inches(1.8), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = CYAN_ACCENT
    bar.line.fill.background()

    # Author info
    tb_auth = s1.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(6.8), Inches(1.8))
    tf_a = tb_auth.text_frame
    pa1 = tf_a.paragraphs[0]
    pa1.text = "Hong Than Brathna"
    pa1.font.size = Pt(18)
    pa1.font.bold = True
    pa1.font.color.rgb = WHITE

    pa2 = tf_a.add_paragraph()
    pa2.text = "Software Engineering  |  Lecturer: Mr. Soklong HIM  |  2026"
    pa2.font.size = Pt(13)
    pa2.font.color.rgb = TEXT_GRAY
    pa2.space_before = Pt(4)

    # Right side: 3 cards flow with arrows (Handwritten Line -> Model -> Khmer Unicode)
    c1 = add_custom_card(s1, Inches(7.5), Inches(2.5), Inches(1.7), Inches(2.6))
    if os.path.exists(TEST_LINE):
        s1.shapes.add_picture(TEST_LINE, Inches(7.6), Inches(3.0), width=Inches(1.5))
    tb_c1 = s1.shapes.add_textbox(Inches(7.5), Inches(4.3), Inches(1.7), Inches(0.7))
    tb_c1.text_frame.word_wrap = True
    p = tb_c1.text_frame.paragraphs[0]
    p.text = "Handwritten\nLine Image"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_OFFWHITE
    p.alignment = PP_ALIGN.CENTER

    arr1 = s1.shapes.add_textbox(Inches(9.2), Inches(3.5), Inches(0.4), Inches(0.5))
    arr1.text_frame.paragraphs[0].text = "➔"
    arr1.text_frame.paragraphs[0].font.size = Pt(18)
    arr1.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    c2 = add_custom_card(s1, Inches(9.6), Inches(2.5), Inches(1.7), Inches(2.6))
    tb_c2_icon = s1.shapes.add_textbox(Inches(9.6), Inches(3.0), Inches(1.7), Inches(1.0))
    tb_c2_icon.text_frame.paragraphs[0].text = "🧠 ⚙️"
    tb_c2_icon.text_frame.paragraphs[0].font.size = Pt(28)
    tb_c2_icon.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    tb_c2 = s1.shapes.add_textbox(Inches(9.6), Inches(4.3), Inches(1.7), Inches(0.7))
    tb_c2.text_frame.word_wrap = True
    p = tb_c2.text_frame.paragraphs[0]
    p.text = "Deep Learning\nOCR Model"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_OFFWHITE
    p.alignment = PP_ALIGN.CENTER

    arr2 = s1.shapes.add_textbox(Inches(11.3), Inches(3.5), Inches(0.4), Inches(0.5))
    arr2.text_frame.paragraphs[0].text = "➔"
    arr2.text_frame.paragraphs[0].font.size = Pt(18)
    arr2.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    c3 = add_custom_card(s1, Inches(11.7), Inches(2.5), Inches(1.4), Inches(2.6))
    tb_c3_text = s1.shapes.add_textbox(Inches(11.7), Inches(3.0), Inches(1.4), Inches(1.0))
    tb_c3_text.text_frame.paragraphs[0].text = "ខ្មែរ"
    tb_c3_text.text_frame.paragraphs[0].font.size = Pt(32)
    tb_c3_text.text_frame.paragraphs[0].font.bold = True
    tb_c3_text.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT
    tb_c3_text.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    tb_c3 = s1.shapes.add_textbox(Inches(11.7), Inches(4.3), Inches(1.4), Inches(0.7))
    tb_c3.text_frame.word_wrap = True
    p = tb_c3.text_frame.paragraphs[0]
    p.text = "Khmer Unicode\nText"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_OFFWHITE
    p.alignment = PP_ALIGN.CENTER

    s1.notes_slide.notes_text_frame.text = (
        "Good morning, respected lecturer Mr. Soklong HIM. "
        "Today I present my final project: Khmer Handwritten Line Recognition: A Comparison "
        "of Three CTC-Based Deep Learning Approaches. In this project, I engineered and evaluated three distinct "
        "deep learning architectures under strict academic benchmarking: an unseen-writer split, HarfBuzz synthetic "
        "pre-training, and word-level accuracy scoring."
    )

    # =========================================================================
    # SLIDE 2: Problem Statement (Matches 2.png)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s2)
    add_top_badge_and_header(s2, "02  PROBLEM STATEMENT", "Why is Khmer Handwritten OCR", "Uniquely Difficult?")

    col_w = Inches(3.733)
    col_gap = Inches(0.266)
    left_start = Inches(0.8)

    cards_data = [
        ("01", "Complex Character", "Structure",
         "Khmer is an abugida script with stacked consonants and dependent vowels positioned around the base consonant.\n\n"
         "● Characters span multiple vertical levels (subscript Coeng, base, upper vowel).\n"
         "● Many possible complex ligature combinations.\n"
         "● High density of visually similar diacritics."),

        ("02", "No Word Spaces", "(Continuous Script)",
         "Khmer sentences are written continuously without spaces between individual words.\n\n"
         "● Words cannot be segmented using simple whitespace slicing.\n"
         "● Character boundaries are blurred and continuous.\n"
         "● Requires end-to-end sequence alignment via CTC loss."),

        ("03", "High Inter-Writer", "Variability",
         "Handwriting styles differ significantly across writers, with variations in stroke thickness, cursive loops and slant.\n\n"
         "● The same character looks drastically different across people.\n"
         "● Visually similar character pairs (e.g. ព vs យ, ត vs ដ, ច vs ជ).\n"
         "● Requires strict writer-disjoint evaluation to prevent data leakage.")
    ]

    for i, (num, t1, t2, body) in enumerate(cards_data):
        c_left = left_start + i * (col_w + col_gap)
        add_custom_card(s2, c_left, Inches(1.9), col_w, Inches(5.1))
        add_num_badge(s2, c_left + Inches(0.3), Inches(2.2), num)

        tb = s2.shapes.add_textbox(c_left + Inches(0.95), Inches(2.15), col_w - Inches(1.1), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = t1 + "\n"
        r1.font.size = Pt(14)
        r1.font.bold = True
        r1.font.color.rgb = WHITE
        r2 = p.add_run()
        r2.text = t2
        r2.font.size = Pt(14)
        r2.font.bold = True
        r2.font.color.rgb = CYAN_ACCENT

        tb_body = s2.shapes.add_textbox(c_left + Inches(0.3), Inches(3.1), col_w - Inches(0.6), Inches(3.7))
        tf_b = tb_body.text_frame
        tf_b.word_wrap = True
        p_b = tf_b.paragraphs[0]
        p_b.text = body
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = TEXT_OFFWHITE

    s2.notes_slide.notes_text_frame.text = (
        "Khmer handwriting presents three unique challenges. First, vertical stacking of subscript Coeng consonants. "
        "Second, Khmer words flow without spaces, making naive bounding-box slicing impossible. "
        "Third, high inter-writer variability and visual twins like ព and យ. "
        "Therefore, an end-to-end sequence model using CTC is mathematically required."
    )

    # =========================================================================
    # SLIDE 3: System Pipeline (Matches 3.png)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s3)
    add_top_badge_and_header(s3, "03  SYSTEM PIPELINE", "End-to-End", "Recognition Pipeline",
                             "From handwritten line image to Khmer Unicode text using deep learning.")

    pipe_steps = [
        ("1", "Input Image", "Raw handwritten line crop\n(variable width)"),
        ("2", "Preprocessing", "Resize height 48px\nKeep aspect ratio & pad"),
        ("3", "CNN Feature Ext.", "VGG or ResNet-18\nExtracts visual frames"),
        ("4", "Sequence Model", "BiLSTM or Transformer\nModels sequence context"),
        ("5", "CTC Decoding", "CTC greedy decoding\n(no character boxes)"),
        ("6", "Output Text", "Khmer Unicode text\n(NFC normalized)")
    ]

    p_w = Inches(1.8)
    p_gap = Inches(0.18)
    for i, (num, title, desc) in enumerate(pipe_steps):
        c_left = Inches(0.8) + i * (p_w + p_gap)
        add_custom_card(s3, c_left, Inches(2.0), p_w, Inches(3.1))
        add_num_badge(s3, c_left + Inches(0.15), Inches(2.15), num)

        tb = s3.shapes.add_textbox(c_left + Inches(0.1), Inches(2.75), p_w - Inches(0.2), Inches(0.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

        tb_d = s3.shapes.add_textbox(c_left + Inches(0.1), Inches(3.5), p_w - Inches(0.2), Inches(1.4))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        p_d = tf_d.paragraphs[0]
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = TEXT_OFFWHITE
        p_d.alignment = PP_ALIGN.CENTER

        if i < 5:
            arr = s3.shapes.add_textbox(c_left + p_w, Inches(3.2), p_gap, Inches(0.5))
            arr.text_frame.paragraphs[0].text = "➔"
            arr.text_frame.paragraphs[0].font.size = Pt(14)
            arr.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    # Bottom summary card (Input Normalization, Vocabulary, Post-processing)
    b_card = add_custom_card(s3, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.7))
    tb_sum = s3.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.5))
    tf_sum = tb_sum.text_frame
    tf_sum.word_wrap = True
    p = tf_sum.paragraphs[0]
    p.text = "Key Pipeline Principles"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_body = tf_sum.add_paragraph()
    p_body.text = (
        "● Input Normalization: Grayscale 1-channel, height 48px, horizontal max width 1600px with aspect-ratio preservation.\n"
        "● Vocabulary: 86 Khmer Unicode NFC code-points (33 consonants, 15 independent vowels, 16 dependent vowels, 14 diacritics, 8 digits/punctuation, index 0 reserved for CTC <blank>).\n"
        "● Post-Processing: Orthographic cleaner fixes invalid duplicate vowels and misplaced diacritics."
    )
    p_body.font.size = Pt(10.5)
    p_body.font.color.rgb = TEXT_OFFWHITE
    p_body.space_before = Pt(4)

    s3.notes_slide.notes_text_frame.text = (
        "Here is our complete pipeline. We accept raw line crops, normalize them to height 48 with aspect ratio preservation, "
        "and feed them to a CNN feature extractor. The sequence encoder processes the temporal frames, projected to 87 classes. "
        "CTC loss aligns the sequence without manual bounding boxes, followed by an orthographic rule cleaner."
    )

    # =========================================================================
    # SLIDE 4: Dataset & Writer-Disjoint Split (Matches 4.png)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s4)
    add_top_badge_and_header(s4, "04  DATASET", "Dataset &", "Writer-Disjoint Split",
                             "Training, validation and testing are performed on different writers to evaluate generalization.")

    # 4 Cards in row 1
    d_cards = [
        ("Training Set", "177", "handwritten lines", "7 writers"),
        ("Validation Set", "50", "handwritten lines", "2 writers"),
        ("Test Set", "50", "handwritten lines", "2 writers (W007, W0011)"),
    ]
    for i, (title, num, unit, sub) in enumerate(d_cards):
        c_left = Inches(0.8) + i * Inches(2.8)
        add_custom_card(s4, c_left, Inches(1.9), Inches(2.6), Inches(2.4))
        tb = s4.shapes.add_textbox(c_left + Inches(0.15), Inches(2.05), Inches(2.3), Inches(2.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = CYAN_ACCENT

        p2 = tf.add_paragraph()
        p2.text = num
        p2.font.size = Pt(36)
        p2.font.bold = True
        p2.font.color.rgb = WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = f"{unit}\n{sub}"
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_GRAY

    # Card 4: Writer-Disjoint Partition Graphic
    add_custom_card(s4, Inches(9.2), Inches(1.9), Inches(3.333), Inches(2.4))
    tb_p = s4.shapes.add_textbox(Inches(9.35), Inches(2.05), Inches(3.0), Inches(2.1))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    p1 = tf_p.paragraphs[0]
    p1.text = "Writer-Disjoint Partition"
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = CYAN_ACCENT

    p2 = tf_p.add_paragraph()
    p2.text = "●●●●●●●  ●●  ●●\nTrain (7)     Val (2)  Test (2)"
    p2.font.size = Pt(13)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(8)

    p3 = tf_p.add_paragraph()
    p3.text = "✓ Zero writer leakage: W007 and W0011 were permanently held out for test evaluation."
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = TEXT_OFFWHITE
    p3.space_before = Pt(8)

    # Bottom card: Example Handwritten Lines
    add_custom_card(s4, Inches(0.8), Inches(4.5), Inches(11.733), Inches(2.5))
    tb_ex = s4.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(11.3), Inches(0.4))
    tb_ex.text_frame.paragraphs[0].text = "Example Handwritten Lines from Each Partition"
    tb_ex.text_frame.paragraphs[0].font.size = Pt(13)
    tb_ex.text_frame.paragraphs[0].font.bold = True
    tb_ex.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    # 3 crops
    crops = [
        (TRAIN_LINE, "Training Example (W008)", Inches(1.0)),
        (VAL_LINE, "Validation Example (W009)", Inches(4.9)),
        (TEST_LINE, "Test Example — Unseen Writer (W0011)", Inches(8.8))
    ]
    for img_p, label, left in crops:
        if os.path.exists(img_p):
            s4.shapes.add_picture(img_p, left, Inches(5.1), width=Inches(3.5))
        tb_lbl = s4.shapes.add_textbox(left, Inches(6.4), Inches(3.5), Inches(0.4))
        p = tb_lbl.text_frame.paragraphs[0]
        p.text = label
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TEXT_OFFWHITE
        p.alignment = PP_ALIGN.CENTER

    s4.notes_slide.notes_text_frame.text = (
        "We enforced a strict writer-disjoint split. Writers W007 and W0011 were held out permanently. "
        "Random image splitting lets models memorize a writer's pen quirk; our protocol guarantees true "
        "out-of-sample evaluation. Our vocabulary covers 86 Unicode characters with mandatory round-trip verification."
    )

    # =========================================================================
    # SLIDE 5: Training Strategy (Matches 5.png)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s5)
    add_top_badge_and_header(s5, "05  TRAINING STRATEGY", "Two-Stage", "Training Strategy",
                             "Use synthetic data to learn Khmer characters, then fine-tune on real handwriting.")

    # Stage 1 Card
    add_custom_card(s5, Inches(0.8), Inches(1.9), Inches(5.6), Inches(3.3))
    add_num_badge(s5, Inches(1.0), Inches(2.1), "1")
    tb1_t = s5.shapes.add_textbox(Inches(1.65), Inches(2.1), Inches(4.5), Inches(0.6))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    r1 = p.add_run()
    r1.text = "Stage 1: "
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    r2 = p.add_run()
    r2.text = "Synthetic Pre-training"
    r2.font.size = Pt(14)
    r2.font.bold = True
    r2.font.color.rgb = CYAN_ACCENT

    tb1_b = s5.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(5.2), Inches(2.2))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    p_num = tf1_b.paragraphs[0]
    p_num.text = "10,000 Synthetic Line Samples"
    p_num.font.size = Pt(18)
    p_num.font.bold = True
    p_num.font.color.rgb = WHITE

    p_desc = tf1_b.add_paragraph()
    p_desc.text = (
        "● Generated with HarfBuzz authentic OpenType text shaping.\n"
        "● Rendered across 8 diverse Khmer fonts (serif, sans-serif, round, slanted).\n"
        "● Complete coverage of all 86 Khmer Unicode code-points.\n"
        "● Result: Solves CTC blank-collapse and teaches baseline character geometry."
    )
    p_desc.font.size = Pt(10.5)
    p_desc.font.color.rgb = TEXT_OFFWHITE
    p_desc.space_before = Pt(6)

    # Arrow between stages
    arr_s5 = s5.shapes.add_textbox(Inches(6.45), Inches(3.2), Inches(0.4), Inches(0.5))
    arr_s5.text_frame.paragraphs[0].text = "➔"
    arr_s5.text_frame.paragraphs[0].font.size = Pt(22)
    arr_s5.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    # Stage 2 Card
    add_custom_card(s5, Inches(6.9), Inches(1.9), Inches(5.633), Inches(3.3))
    add_num_badge(s5, Inches(7.1), Inches(2.1), "2", bg_color=GREEN_ACCENT)
    tb2_t = s5.shapes.add_textbox(Inches(7.75), Inches(2.1), Inches(4.5), Inches(0.6))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    r1 = p.add_run()
    r1.text = "Stage 2: "
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = WHITE
    r2 = p.add_run()
    r2.text = "Real Fine-Tuning"
    r2.font.size = Pt(14)
    r2.font.bold = True
    r2.font.color.rgb = GREEN_ACCENT

    tb2_b = s5.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.2), Inches(2.2))
    tf2_b = tb2_b.text_frame
    tf2_b.word_wrap = True
    p_num2 = tf2_b.paragraphs[0]
    p_num2.text = "177 Real Lines + 3,000 Crop Mix"
    p_num2.font.size = Pt(18)
    p_num2.font.bold = True
    p_num2.font.color.rgb = WHITE

    p_desc2 = tf2_b.add_paragraph()
    p_desc2.text = (
        "● Multi-scale Elastic & Perspective Distortion Augmentations.\n"
        "● Real Oversampling (8x): Real ink represents ~40% of each training epoch.\n"
        "● Synthetic Anchor Mixing (3,000 samples) prevents catastrophic forgetting.\n"
        "● Cosine Annealing with 3 Warmup Epochs + EMA (decay=0.999)."
    )
    p_desc2.font.size = Pt(10.5)
    p_desc2.font.color.rgb = TEXT_OFFWHITE
    p_desc2.space_before = Pt(6)

    # Bottom 4 Pillars Card
    b_card5 = add_custom_card(s5, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.6))
    tb_pil = s5.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.4))
    tf_pil = tb_pil.text_frame
    tf_pil.word_wrap = True
    p = tf_pil.paragraphs[0]
    p.text = "Why Curriculum Learning is Essential"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_b5 = tf_pil.add_paragraph()
    p_b5.text = (
        "1. Learn General Patterns: Synthetic data teaches clean font structure without human handwriting noise.\n"
        "2. Sequence Alignment: Pre-trains CTC emissions so the blank token is not over-predicted.\n"
        "3. Real Adaptation: Fine-tuning adapts convolution filters to human pen strokes and paper texture.\n"
        "4. Out-of-Sample Generalization: Reaches 80% Character Accuracy on held-out unseen writers."
    )
    p_b5.font.size = Pt(10.5)
    p_b5.font.color.rgb = TEXT_OFFWHITE
    p_b5.space_before = Pt(4)

    s5.notes_slide.notes_text_frame.text = (
        "Why two stages? Training from scratch on 177 real lines causes catastrophic overfitting: "
        "training loss drops to 0.14 while validation loss is stuck at 1.42. "
        "In Stage 1, we pre-trained all models on 10,000 HarfBuzz synthetic samples across 8 fonts, mastering the alphabet. "
        "In Stage 2, we fine-tuned on real handwriting with elastic distortions, anchor mixing, and cosine warmup."
    )

    # =========================================================================
    # SLIDE 6: Approach 1 — Baseline CRNN (Matches 6.png)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s6)
    add_top_badge_and_header(s6, "06  APPROACH 1", "Baseline CRNN", "(VGG + BiLSTM + CTC)",
                             "A strong and effective end-to-end architecture for handwritten line recognition.")

    # 3 Bottom Cards (Details, Size, Key Points)
    add_custom_card(s6, Inches(0.8), Inches(2.0), Inches(4.3), Inches(4.9))
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.9), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Details"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_d = tf.add_paragraph()
    p_d.text = (
        "● CNN Backbone:\n"
        "  - 7 Convolutional Layers (VGG-style)\n"
        "  - Asymmetric MaxPool (2, 1) in layers 3 & 4: shrinks height to 1 while keeping W' = W/4.\n\n"
        "● Sequence Modeling:\n"
        "  - 2-Layer Bidirectional LSTM\n"
        "  - Hidden dim = 256 (bidirectional total = 512)\n"
        "  - Dropout = 0.2\n\n"
        "● Output Layer:\n"
        "  - Linear(512 → 87 classes) + CTC Loss\n"
        "  - Greedy decoding."
    )
    p_d.font.size = Pt(11)
    p_d.font.color.rgb = TEXT_OFFWHITE
    p_d.space_before = Pt(8)

    # Middle Card: Model Size 8.49M
    add_custom_card(s6, Inches(5.3), Inches(2.0), Inches(2.8), Inches(4.9))
    tb_size = s6.shapes.add_textbox(Inches(5.4), Inches(2.4), Inches(2.6), Inches(4.0))
    tf_s = tb_size.text_frame
    tf_s.word_wrap = True
    p1 = tf_s.paragraphs[0]
    p1.text = "Model Size"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = CYAN_ACCENT
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf_s.add_paragraph()
    p2.text = "8.49M"
    p2.font.size = Pt(44)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    p3 = tf_s.add_paragraph()
    p3.text = "Trainable Parameters\n\nMost compact and sample-efficient model in our study."
    p3.font.size = Pt(11)
    p3.font.color.rgb = TEXT_GRAY
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(8)

    # Right Card: Key Points
    add_custom_card(s6, Inches(8.3), Inches(2.0), Inches(4.233), Inches(4.9))
    tb_kp = s6.shapes.add_textbox(Inches(8.5), Inches(2.2), Inches(3.8), Inches(4.5))
    tf_kp = tb_kp.text_frame
    tf_kp.word_wrap = True
    p = tf_kp.paragraphs[0]
    p.text = "Key Strengths & Role"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_kpd = tf_kp.add_paragraph()
    p_kpd.text = (
        "✓ Winner of our study: Achieved the lowest CER (20.06%) and highest Char Accuracy (79.94%).\n\n"
        "✓ Ideal Inductive Bias: BiLSTM assumes sequential recurrence, matching left-to-right writing perfectly.\n\n"
        "✓ Asymmetric Pooling: Flattens vertical coeng subscripts directly into temporal feature columns.\n\n"
        "✓ Zero Domain Mismatch: Features learned entirely on Khmer glyphs without natural image distortion."
    )
    p_kpd.font.size = Pt(11)
    p_kpd.font.color.rgb = TEXT_OFFWHITE
    p_kpd.space_before = Pt(8)

    s6.notes_slide.notes_text_frame.text = (
        "Approach 1 is our baseline CRNN. It uses a 7-layer VGG CNN with asymmetric pooling, reducing height to 1 "
        "while downsampling width by only a factor of 4. This maintains rich temporal resolution for CTC frames. "
        "A 2-layer BiLSTM captures forward and backward context. It is compact, parameter-efficient at 8.49M params, "
        "and trained from scratch with synthetic initialization."
    )

    # =========================================================================
    # SLIDE 7: Approach 2 — Transfer Learning (Matches 7.png)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s7)
    add_top_badge_and_header(s7, "07  APPROACH 2", "ResNet-18", "Transfer Learning",
                             "Using a pre-trained ResNet-18 backbone with BiLSTM and CTC for Khmer line recognition.")

    add_custom_card(s7, Inches(0.8), Inches(2.0), Inches(4.3), Inches(4.9))
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.9), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Modifications"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_d = tf.add_paragraph()
    p_d.text = (
        "● Backbone:\n"
        "  - Pre-trained ResNet-18 (ImageNet weights)\n"
        "  - 1-channel grayscale replicated to 3 channels.\n\n"
        "● Resolution Preservation:\n"
        "  - Modified Conv1 & MaxPool stride from 2 to 1 horizontally.\n"
        "  - Asymmetric bridge convolution (3, 1) flattens height to 1.\n\n"
        "● Sequence Head (Held Constant):\n"
        "  - Identical 2-layer BiLSTM (hidden=256)\n"
        "  - Linear CTC projection to 87 classes."
    )
    p_d.font.size = Pt(11)
    p_d.font.color.rgb = TEXT_OFFWHITE
    p_d.space_before = Pt(8)

    # Middle Card: Model Size 30.95M
    add_custom_card(s7, Inches(5.3), Inches(2.0), Inches(2.8), Inches(4.9))
    tb_size7 = s7.shapes.add_textbox(Inches(5.4), Inches(2.4), Inches(2.6), Inches(4.0))
    tf_s7 = tb_size7.text_frame
    tf_s7.word_wrap = True
    p1 = tf_s7.paragraphs[0]
    p1.text = "Model Size"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = CYAN_ACCENT
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf_s7.add_paragraph()
    p2.text = "30.95M"
    p2.font.size = Pt(40)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    p3 = tf_s7.add_paragraph()
    p3.text = "Total Parameters\n\n(3.6x larger than A1)\nHigh capacity architecture."
    p3.font.size = Pt(11)
    p3.font.color.rgb = TEXT_GRAY
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(8)

    # Right Card: Key Points
    add_custom_card(s7, Inches(8.3), Inches(2.0), Inches(4.233), Inches(4.9))
    tb_kp7 = s7.shapes.add_textbox(Inches(8.5), Inches(2.2), Inches(3.8), Inches(4.5))
    tf_kp7 = tb_kp7.text_frame
    tf_kp7.word_wrap = True
    p = tf_kp7.paragraphs[0]
    p.text = "Training Strategy: Differential LR"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_kpd7 = tf_kp7.add_paragraph()
    p_kpd7.text = (
        "● Explores Rubric Dimension (B): Training Strategy.\n\n"
        "● Differential Learning Rate:\n"
        "  - Backbone LR multiplier = 0.25x\n"
        "  - Head LR multiplier = 1.0x\n\n"
        "● 3-Epoch Freeze:\n"
        "  - Allows CTC head to adapt before disturbing pre-trained weights.\n\n"
        "● Insight: Domain mismatch (ImageNet natural photos vs binary ink) caused higher test error (25.78% CER)."
    )
    p_kpd7.font.size = Pt(11)
    p_kpd7.font.color.rgb = TEXT_OFFWHITE
    p_kpd7.space_before = Pt(8)

    s7.notes_slide.notes_text_frame.text = (
        "In Approach 2, we evaluate transfer learning using an ImageNet pre-trained ResNet-18. "
        "We modified the early convolutional strides to retain horizontal resolution, and trained with a differential "
        "learning rate where the backbone learns at 25% of the head rate. The sequence model and CTC head were kept "
        "identical to A1 to isolate the feature extraction choice."
    )

    # =========================================================================
    # SLIDE 8: Approach 3 — Transformer Sequence Encoder (Matches 8.png)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s8)
    add_top_badge_and_header(s8, "08  APPROACH 3", "Transformer", "Sequence Encoder",
                             "Using self-attention to model long-range dependencies in handwritten Khmer lines.")

    add_custom_card(s8, Inches(0.8), Inches(2.0), Inches(4.3), Inches(4.9))
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.9), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architecture Details"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_d = tf.add_paragraph()
    p_d.text = (
        "● VGG Backbone: Identical to A1 (held constant).\n\n"
        "● 1D Sinusoidal Positional Encoding:\n"
        "  - Injects temporal ordering into order-agnostic self-attention.\n\n"
        "● 4-Layer Transformer Encoder:\n"
        "  - 8 Attention heads per layer\n"
        "  - FFN: 512 → 1024 → 512 (GELU)\n"
        "  - Pre-LayerNorm & Residual connections.\n\n"
        "● Output Layer: Identical Linear(512 → 87) + CTC."
    )
    p_d.font.size = Pt(11)
    p_d.font.color.rgb = TEXT_OFFWHITE
    p_d.space_before = Pt(8)

    # Middle Card: Model Size 9.54M
    add_custom_card(s8, Inches(5.3), Inches(2.0), Inches(2.8), Inches(4.9))
    tb_size8 = s8.shapes.add_textbox(Inches(5.4), Inches(2.4), Inches(2.6), Inches(4.0))
    tf_s8 = tb_size8.text_frame
    tf_s8.word_wrap = True
    p1 = tf_s8.paragraphs[0]
    p1.text = "Model Size"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = CYAN_ACCENT
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf_s8.add_paragraph()
    p2.text = "9.54M"
    p2.font.size = Pt(44)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(16)

    p3 = tf_s8.add_paragraph()
    p3.text = "Trainable Parameters\n\nSelf-attention architecture with competitive efficiency."
    p3.font.size = Pt(11)
    p3.font.color.rgb = TEXT_GRAY
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(8)

    # Right Card: Key Points
    add_custom_card(s8, Inches(8.3), Inches(2.0), Inches(4.233), Inches(4.9))
    tb_kp8 = s8.shapes.add_textbox(Inches(8.5), Inches(2.2), Inches(3.8), Inches(4.5))
    tf_kp8 = tb_kp8.text_frame
    tf_kp8.word_wrap = True
    p = tf_kp8.paragraphs[0]
    p.text = "Self-Attention vs Recurrence"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_kpd8 = tf_kp8.add_paragraph()
    p_kpd8.text = (
        "● Explores Rubric Dimension (A): Architecture.\n\n"
        "● Pairwise Attention:\n"
        "  - Connects distant character contexts across long lines in O(1) sequential steps.\n\n"
        "● Inductive Bias Trade-Off:\n"
        "  - Transformers lack localized sequential assumptions.\n"
        "  - Ranked 2nd (22.92% CER, 77.08% CharAcc) behind BiLSTM on limited real handwriting data."
    )
    p_kpd8.font.size = Pt(11)
    p_kpd8.font.color.rgb = TEXT_OFFWHITE
    p_kpd8.space_before = Pt(8)

    s8.notes_slide.notes_text_frame.text = (
        "In Approach 3, we isolate the sequence encoder. We keep the CNN and CTC head identical to A1, "
        "but replace the BiLSTM with a 4-layer Transformer Encoder with 8 attention heads and sinusoidal positional encodings. "
        "This compares multi-head self-attention against recurrent inductive bias directly on identical data."
    )

    # =========================================================================
    # SLIDE 9: Results & Evaluation (Matches 9.png / 9(1).png)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s9)
    add_top_badge_and_header(s9, "09  RESULTS & EVALUATION", "Experimental", "Results",
                             "Evaluated on an unseen handwritten test set with multiple writers and writing styles.")

    # Left Top: 3 Metric Cards for Best Model (A1)
    add_custom_card(s9, Inches(0.8), Inches(1.9), Inches(5.6), Inches(1.5))
    tb_m1 = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(1.6), Inches(1.2))
    tb_m1.text_frame.paragraphs[0].text = "Character Accuracy\n"
    tb_m1.text_frame.paragraphs[0].font.size = Pt(11)
    tb_m1.text_frame.paragraphs[0].font.color.rgb = TEXT_GRAY
    r = tb_m1.text_frame.paragraphs[0].add_run()
    r.text = "79.94%"
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = GREEN_ACCENT

    tb_m2 = s9.shapes.add_textbox(Inches(2.8), Inches(2.0), Inches(1.6), Inches(1.2))
    tb_m2.text_frame.paragraphs[0].text = "Word Accuracy\n"
    tb_m2.text_frame.paragraphs[0].font.size = Pt(11)
    tb_m2.text_frame.paragraphs[0].font.color.rgb = TEXT_GRAY
    r = tb_m2.text_frame.paragraphs[0].add_run()
    r.text = "25.85%"
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = CYAN_ACCENT

    tb_m3 = s9.shapes.add_textbox(Inches(4.6), Inches(2.0), Inches(1.6), Inches(1.2))
    tb_m3.text_frame.paragraphs[0].text = "Lowest CER\n"
    tb_m3.text_frame.paragraphs[0].font.size = Pt(11)
    tb_m3.text_frame.paragraphs[0].font.color.rgb = TEXT_GRAY
    r = tb_m3.text_frame.paragraphs[0].add_run()
    r.text = "20.06%"
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = WHITE

    # Left Bottom: Full Comparison Table
    add_custom_card(s9, Inches(0.8), Inches(3.6), Inches(5.6), Inches(3.3))
    tb_tbl_title = s9.shapes.add_textbox(Inches(1.0), Inches(3.7), Inches(5.2), Inches(0.4))
    tb_tbl_title.text_frame.paragraphs[0].text = "Comparison with Other Models"
    tb_tbl_title.text_frame.paragraphs[0].font.size = Pt(13)
    tb_tbl_title.text_frame.paragraphs[0].font.bold = True
    tb_tbl_title.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    tbl_s9_shape = s9.shapes.add_table(4, 5, Inches(1.0), Inches(4.2), Inches(5.2), Inches(2.4))
    table9 = tbl_s9_shape.table
    w_list = [Inches(1.8), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85)]
    for j, w in enumerate(w_list):
        table9.columns[j].width = w

    h_list = ["Model", "CharAcc ↑", "WordAcc ↑", "CER ↓", "LineAcc ↑"]
    for j, h in enumerate(h_list):
        cell = table9.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PILL_BG
        cell.text_frame.paragraphs[0].font.size = Pt(9.5)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = WHITE

    rows_data = [
        ("A1 CRNN (Winner)", "79.94%", "25.85%", "20.06%", "10.00%"),
        ("A3 Transformer", "77.08%", "19.73%", "22.92%", "2.00%"),
        ("A2 Transfer ResNet", "74.22%", "12.24%", "25.78%", "0.00%")
    ]
    for i, r in enumerate(rows_data):
        for j, val in enumerate(r):
            cell = table9.cell(i + 1, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(9.5)
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(16, 42, 86)
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    # Right Card: Real Sample Predictions (Unseen Writer W0011)
    add_custom_card(s9, Inches(6.8), Inches(1.9), Inches(5.733), Inches(5.0))
    tb_pred_title = s9.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.3), Inches(0.4))
    tb_pred_title.text_frame.paragraphs[0].text = "Sample Predictions (Held-Out Writer W0011)"
    tb_pred_title.text_frame.paragraphs[0].font.size = Pt(13)
    tb_pred_title.text_frame.paragraphs[0].font.bold = True
    tb_pred_title.text_frame.paragraphs[0].font.color.rgb = CYAN_ACCENT

    # Real prediction sample 1
    if os.path.exists(TEST_LINE):
        s9.shapes.add_picture(TEST_LINE, Inches(7.0), Inches(2.6), width=Inches(3.2))

    tb_pred1 = s9.shapes.add_textbox(Inches(7.0), Inches(3.2), Inches(5.3), Inches(1.1))
    tf_p1 = tb_pred1.text_frame
    tf_p1.word_wrap = True
    p1 = tf_p1.paragraphs[0]
    p1.text = "Ground Truth:\nព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់"
    p1.font.size = Pt(11)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf_p1.add_paragraph()
    p2.text = "Model Prediction: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់  [✓ Correct]"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = GREEN_ACCENT
    p2.space_before = Pt(4)

    # Key takeaways bullet below
    tb_p_note = s9.shapes.add_textbox(Inches(7.0), Inches(4.5), Inches(5.3), Inches(2.2))
    tf_pn = tb_p_note.text_frame
    tf_pn.word_wrap = True
    p = tf_pn.paragraphs[0]
    p.text = "Qualitative Highlights"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_body = tf_pn.add_paragraph()
    p_body.text = (
        "● Successfully recognizes full 58-character handwritten sentences with zero character errors.\n"
        "● Accurate recovery of complex subscript Coengs (្រ, ៀ, ើ) without horizontal alignment drift.\n"
        "● 10% of all lines in the held-out test set matched with 100% exact character perfection!"
    )
    p_body.font.size = Pt(10.5)
    p_body.font.color.rgb = TEXT_OFFWHITE
    p_body.space_before = Pt(4)

    s9.notes_slide.notes_text_frame.text = (
        "Here are our empirical results on test writers W007 and W0011. "
        "Approach 1 (Baseline CRNN) won first place with 79.94% Character Accuracy and 25.85% Word-Level Accuracy, "
        "transcribing 10% of full lines perfectly. "
        "BiLSTM outperformed Transformer because sequential recurrence is the ideal inductive bias for handwriting. "
        "Transfer learning came third due to domain mismatch between ImageNet and binary pen strokes."
    )

    # =========================================================================
    # SLIDE 10: Model Performance Charts (Matches 9(1).png)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s10)
    add_top_badge_and_header(s10, "10  PERFORMANCE ANALYSIS", "Model Performance &", "Learning Curves",
                             "Overlaid validation learning curves and error rates across all three architectures.")

    # Left: Embed overlaid learning curves
    if os.path.exists(CURVES_IMG):
        s10.shapes.add_picture(CURVES_IMG, Inches(0.8), Inches(1.9), width=Inches(6.6))

    # Right: Embed comparison bar chart
    if os.path.exists(CHART_IMG):
        s10.shapes.add_picture(CHART_IMG, Inches(7.6), Inches(1.9), width=Inches(4.9))

    # Bottom summary card
    add_custom_card(s10, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.7))
    tb_c_obs = s10.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.5))
    tf_co = tb_c_obs.text_frame
    tf_co.word_wrap = True
    p = tf_co.paragraphs[0]
    p.text = "Key Scientific Observations"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_d = tf_co.add_paragraph()
    p_d.text = (
        "● Blue Curve (A1 CRNN): Rapidly drops from 46% to 10.9% validation CER, achieving the lowest error floor.\n"
        "● Green Curve (A3 Transformer): Stably converges around 14.7% validation CER, performing well but trailing BiLSTM.\n"
        "● Orange Curve (A2 ResNet-18): Plateaus around 18.7% validation CER due to capacity bloat and domain shift.\n"
        "● Regularization Success: Cosine warmup and EMA (0.999) completely eliminated validation divergence spikes."
    )
    p_d.font.size = Pt(10.5)
    p_d.font.color.rgb = TEXT_OFFWHITE
    p_d.space_before = Pt(4)

    s10.notes_slide.notes_text_frame.text = (
        "Looking at the overlaid validation learning curves, we observe the training dynamics. "
        "A1 (blue) achieved the lowest and most stable validation trajectory, dropping smoothly from 46% down to 10.9%. "
        "A3 Transformer (green) converged reliably but plateaued higher. "
        "Notice the complete absence of validation spikes, proving that EMA decay and cosine warmup stabilized training."
    )

    # =========================================================================
    # SLIDE 11: Qualitative Error Analysis (Matches 10.png & 11.png)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s11)
    add_top_badge_and_header(s11, "11  QUALITATIVE ANALYSIS", "Recognition Examples &", "Error Analysis",
                             "Examples from unseen handwritten lines showing correct predictions and common error cases.")

    col3_w = Inches(3.733)
    # Card 1: Correct Example
    add_custom_card(s11, Inches(0.8), Inches(1.9), col3_w, Inches(4.0))
    tb_e1 = s11.shapes.add_textbox(Inches(1.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.7))
    tf_e1 = tb_e1.text_frame
    tf_e1.word_wrap = True
    p = tf_e1.paragraphs[0]
    p.text = "✓ Correct Recognition Example"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    p_b1 = tf_e1.add_paragraph()
    p_b1.text = (
        "Ground Truth:\nព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n\n"
        "Model Prediction:\nព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់\n\n"
        "● Outcome: 100% exact match across all 58 characters.\n"
        "● Recovers subscript coeng ្រ, upper vowels ឹ, ើ, and dependent vowel ៀ without drift."
    )
    p_b1.font.size = Pt(10.5)
    p_b1.font.color.rgb = TEXT_OFFWHITE
    p_b1.space_before = Pt(8)

    # Card 2: Minor Error
    add_custom_card(s11, Inches(4.8), Inches(1.9), col3_w, Inches(4.0))
    tb_e2 = s11.shapes.add_textbox(Inches(5.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.7))
    tf_e2 = tb_e2.text_frame
    tf_e2.word_wrap = True
    p = tf_e2.paragraphs[0]
    p.text = "⚠️ Minor Recognition Error"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ORANGE_ACCENT

    p_b2 = tf_e2.add_paragraph()
    p_b2.text = (
        "Ground Truth:\nក្បាលម៉ាស៊ីនរថភ្លើង\n\n"
        "Model Prediction:\nកាលម៉ាស៊ីនរថភ្លើង\n\n"
        "● Error: Subscript Coeng ្ + ប omitted (ក្ប → ក).\n"
        "● Cause: In faint cursive handwriting, small subscript strokes lack contrast, causing temporal collapse."
    )
    p_b2.font.size = Pt(10.5)
    p_b2.font.color.rgb = TEXT_OFFWHITE
    p_b2.space_before = Pt(8)

    # Card 3: Challenging Example
    add_custom_card(s11, Inches(8.8), Inches(1.9), col3_w, Inches(4.0))
    tb_e3 = s11.shapes.add_textbox(Inches(9.0), Inches(2.05), col3_w - Inches(0.4), Inches(3.7))
    tf_e3 = tb_e3.text_frame
    tf_e3.word_wrap = True
    p = tf_e3.paragraphs[0]
    p.text = "✗ Challenging Visual Twin Example"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT

    p_b3 = tf_e3.add_paragraph()
    p_b3.text = (
        "Ground Truth: ព → Prediction: យ\n"
        "Ground Truth: នោះ → Prediction: នេះ\n\n"
        "● Cause: Visual twin loops collapse in quick cursive handwriting, making them indistinguishable without language context.\n"
        "● Solution: Requires a language model for semantic disambiguation."
    )
    p_b3.font.size = Pt(10.5)
    p_b3.font.color.rgb = TEXT_OFFWHITE
    p_b3.space_before = Pt(8)

    # Bottom Observations Card
    add_custom_card(s11, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.1))
    tb_obs11 = s11.shapes.add_textbox(Inches(1.0), Inches(6.1), Inches(11.3), Inches(0.9))
    tf_o11 = tb_obs11.text_frame
    tf_o11.word_wrap = True
    p = tf_o11.paragraphs[0]
    p.text = "Key Takeaway: The model recognizes clear and consistent handwriting accurately. Errors concentrate on faint subscripts and visual twins."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    s11.notes_slide.notes_text_frame.text = (
        "Qualitative error analysis revealed two primary failure modes. "
        "First, faint subscript coengs. When a writer scribbles lightly, the downsampled CNN sometimes misses the stroke. "
        "Second, visual twin characters like ព and យ where cursive handwriting collapses loops. "
        "However, on clear handwriting, A1 transcribes entire 58-character lines with 100% precision."
    )

    # =========================================================================
    # SLIDE 12: Live Inference Demo & Production Artifacts (Matches 12.png)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s12)
    add_top_badge_and_header(s12, "12  LIVE INFERENCE DEMO", "Live Inference Demo &", "Production Artifacts",
                             "Real-time prediction CLI comparing baseline and transfer models on held-out lines.")

    # Left: Embed live demo output
    if os.path.exists(DEMO_IMG):
        s12.shapes.add_picture(DEMO_IMG, Inches(0.8), Inches(1.9), width=Inches(6.8))

    # Terminal card below demo image
    add_custom_card(s12, Inches(0.8), Inches(4.0), Inches(6.8), Inches(3.0))
    tb_cmd = s12.shapes.add_textbox(Inches(1.0), Inches(4.15), Inches(6.4), Inches(2.6))
    tf_cmd = tb_cmd.text_frame
    tf_cmd.word_wrap = True
    p = tf_cmd.paragraphs[0]
    p.text = "Terminal Inference CLI Command:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_cmd = tf_cmd.add_paragraph()
    p_cmd.text = (
        "$ python scripts/demo_predict.py \\\n"
        "    --image data/handwritten_external/W0011/lines_raw/W0011_P001_L001.png \\\n"
        "    --checkpoint checkpoints/a1_baseline_finetuned_best.pt \\\n"
        "    --vocab-path data/metadata/char_to_idx.json \\\n"
        "    --approach baseline\n\n"
        "✓ Output: ព្រឹកនេះ នៅក្នុងបន្ទប់រៀនយើងមានសិស្សជាច្រើននាក់"
    )
    p_cmd.font.size = Pt(10)
    p_cmd.font.name = "Courier New"
    p_cmd.font.color.rgb = GREEN_ACCENT
    p_cmd.space_before = Pt(4)

    # Right Card: Software Engineering Deliverables
    add_custom_card(s12, Inches(7.8), Inches(1.9), Inches(4.733), Inches(5.1))
    tb_art = s12.shapes.add_textbox(Inches(8.0), Inches(2.1), Inches(4.3), Inches(4.7))
    tf_art = tb_art.text_frame
    tf_art.word_wrap = True
    p = tf_art.paragraphs[0]
    p.text = "Software Engineering Deliverables"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_art = tf_art.add_paragraph()
    p_art.text = (
        "● Modular PyTorch Architecture:\n"
        "  - src/models/ (A1 Baseline, A2 Transfer, A3 Transformer)\n"
        "  - src/dataset.py & collate_fn with dynamic white padding\n"
        "  - src/tokenizer.py & postprocessor.py\n\n"
        "● Automated Sheet Cropping Pipeline:\n"
        "  - src/scan_pipeline.py with 4-point homography deskewing\n\n"
        "● 100% Passing Unit Test Suite:\n"
        "  - 30 pytest unit tests passing across macOS and Linux Colab\n\n"
        "● Complete Reproducibility:\n"
        "  - Fixed RNG seeds, checkpoints, and automated Colab runner."
    )
    p_art.font.size = Pt(11)
    p_art.font.color.rgb = TEXT_OFFWHITE
    p_art.space_before = Pt(8)

    s12.notes_slide.notes_text_frame.text = (
        "In addition to deep learning modeling, this project delivered clean software engineering artifacts. "
        "A modular PyTorch codebase, a sheet cropping pipeline with perspective deskewing, and a 100% passing test suite. "
        "Our live demo CLI predicts unseen test lines in real time."
    )

    # =========================================================================
    # SLIDE 13: Conclusion & Future Work (Matches 13.png)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s13)
    add_top_badge_and_header(s13, "13  CONCLUSION & FUTURE WORK", "Conclusion &", "Future Work",
                             "Summary of findings, limitations, and directions for future improvement.")

    col13_w = Inches(3.733)
    # Col 1: Key Findings
    add_custom_card(s13, Inches(0.8), Inches(1.9), col13_w, Inches(4.0))
    tb_kf = s13.shapes.add_textbox(Inches(1.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf_kf = tb_kf.text_frame
    tf_kf.word_wrap = True
    p = tf_kf.paragraphs[0]
    p.text = "🏆 Key Findings"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_kfb = tf_kf.add_paragraph()
    p_kfb.text = (
        "● A1 CRNN Performed Best: Lowest CER (20.06%) and highest word accuracy (25.85%) on unseen writers.\n\n"
        "● BiLSTM > Transformer: Innate sequential recurrence inductive bias gave BiLSTM superior sample efficiency on modest data.\n\n"
        "● Synthetic Curriculum was Critical: HarfBuzz pre-training eliminated CTC blank collapse."
    )
    p_kfb.font.size = Pt(10.5)
    p_kfb.font.color.rgb = TEXT_OFFWHITE
    p_kfb.space_before = Pt(8)

    # Col 2: Limitations
    add_custom_card(s13, Inches(4.8), Inches(1.9), col13_w, Inches(4.0))
    tb_lim = s13.shapes.add_textbox(Inches(5.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf_lim = tb_lim.text_frame
    tf_lim.word_wrap = True
    p = tf_lim.paragraphs[0]
    p.text = "⚠️ Limitations"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ORANGE_ACCENT

    p_limb = tf_lim.add_paragraph()
    p_limb.text = (
        "● Limited Human Writers: 11 writers total; test set may not cover all Cambodian regional cursive quirks.\n\n"
        "● Low Contrast Ink: Faint pencil or pen strokes can cause subscript omissions.\n\n"
        "● No Integrated Language Model: Visual twins cannot be disambiguated by semantic context."
    )
    p_limb.font.size = Pt(10.5)
    p_limb.font.color.rgb = TEXT_OFFWHITE
    p_limb.space_before = Pt(8)

    # Col 3: Future Work
    add_custom_card(s13, Inches(8.8), Inches(1.9), col13_w, Inches(4.0))
    tb_fw = s13.shapes.add_textbox(Inches(9.0), Inches(2.05), col13_w - Inches(0.4), Inches(3.7))
    tf_fw = tb_fw.text_frame
    tf_fw.word_wrap = True
    p = tf_fw.paragraphs[0]
    p.text = "💡 Future Work"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    p_fwb = tf_fw.add_paragraph()
    p_fwb.text = (
        "● Larger Multi-Province Corpus: Collect student handwriting across multiple provinces.\n\n"
        "● Khmer Language Model: Integrate pre-trained Khmer RoBERTa for post-OCR beam search decoding.\n\n"
        "● Mobile Deployment: Export A1 via ONNX / CoreML for instant offline phone camera scanning."
    )
    p_fwb.font.size = Pt(10.5)
    p_fwb.font.color.rgb = TEXT_OFFWHITE
    p_fwb.space_before = Pt(8)

    # Bottom Banner: Final Takeaway
    add_custom_card(s13, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.1))
    tb_ft = s13.shapes.add_textbox(Inches(1.0), Inches(6.1), Inches(11.3), Inches(0.9))
    tf_ft = tb_ft.text_frame
    tf_ft.word_wrap = True
    p = tf_ft.paragraphs[0]
    p.text = "🎯 Final Takeaway: A1 Baseline CRNN achieved the best performance with ~80% character accuracy, proving that well-regularized recurrence and synthetic curriculum pre-training are the optimal foundation for Khmer OCR."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    s13.notes_slide.notes_text_frame.text = (
        "To conclude: our comparative study demonstrated that Baseline CRNN with VGG and BiLSTM provides the most "
        "accurate and sample-efficient architecture for Khmer handwritten line recognition, reaching 80% character accuracy. "
        "Synthetic pre-training was essential to make real fine-tuning viable. "
        "Thank you very much, Mr. Soklong. I am now ready for your questions!"
    )

    # =========================================================================
    # SLIDE 14: References & Acknowledgements (Matches 14.png)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_bg(s14)
    add_top_badge_and_header(s14, "14  REFERENCES & ACKNOWLEDGEMENTS", "References &", "Acknowledgements",
                             "Key academic resources, tools, and mentors who supported this project.")

    # Left: References
    add_custom_card(s14, Inches(0.8), Inches(1.9), Inches(5.7), Inches(4.0))
    tb_ref = s14.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(5.3), Inches(3.7))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True
    p = tf_ref.paragraphs[0]
    p.text = "📖 Academic References"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_r = tf_ref.add_paragraph()
    p_r.text = (
        "1. Graves, A., et al. (2006). Connectionist Temporal Classification (CTC): Labelling Unsegmented Sequence Data. ICML.\n"
        "2. Shi, B., Bai, X., & Yao, C. (2016). An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN). IEEE TPAMI.\n"
        "3. He, K., et al. (2016). Deep Residual Learning for Image Recognition (ResNet). CVPR.\n"
        "4. Vaswani, A., et al. (2017). Attention Is All You Need (Transformer). NeurIPS.\n"
        "5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR."
    )
    p_r.font.size = Pt(9.5)
    p_r.font.color.rgb = TEXT_OFFWHITE
    p_r.space_before = Pt(6)

    # Right: Acknowledgements
    add_custom_card(s14, Inches(6.8), Inches(1.9), Inches(5.733), Inches(4.0))
    tb_ack = s14.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.3), Inches(3.7))
    tf_ack = tb_ack.text_frame
    tf_ack.word_wrap = True
    p = tf_ack.paragraphs[0]
    p.text = "🤝 Acknowledgements"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_a = tf_ack.add_paragraph()
    p_a.text = (
        "● Academic Guidance:\n"
        "  Sincere gratitude to our lecturer Mr. Soklong HIM for continuous mentorship, technical rigor, and feedback.\n\n"
        "● Open-Source Community:\n"
        "  PyTorch team, Google Colab, and HarfBuzz for high-performance deep learning and OpenType shaping engines.\n\n"
        "● Dataset Contributors:\n"
        "  Thank you to all human writers and contributors who provided real handwritten samples for this research."
    )
    p_a.font.size = Pt(10)
    p_a.font.color.rgb = TEXT_OFFWHITE
    p_a.space_before = Pt(6)

    # Bottom Banner: Thank You
    add_custom_card(s14, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.1))
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
    r2.text = "|  Thank you for your time and attention. I look forward to your questions and discussion."
    r2.font.size = Pt(12)
    r2.font.color.rgb = CYAN_ACCENT

    s14.notes_slide.notes_text_frame.text = (
        "Thank you very much, Mr. Soklong HIM. I am now ready for your questions!"
    )

    out_file1 = "slides/Khmer_Handwritten_OCR_Presentation.pptx"
    out_file2 = "/Users/macbookpro/Desktop/Khmer_Handwritten_OCR_Presentation.pptx"
    prs.save(out_file1)
    prs.save(out_file2)
    print(f"✓ Beautiful deck saved to {out_file1} and {out_file2}!")

if __name__ == "__main__":
    create_deck()

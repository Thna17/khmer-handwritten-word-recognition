"""
Day 1 Educational Demonstration Script: The 4 Pillars of CRNN + CTC for Khmer OCR
Run this script:
    .venv/bin/python tests/demo_day1_concepts.py
"""

import torch
import torch.nn as nn
import pandas as pd

def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# -------------------------------------------------------------
# Pillar 1: The Role of the CNN
# -------------------------------------------------------------
def demo_cnn_role():
    print_banner("PILLAR 1: THE ROLE OF THE CNN")
    print("Concept: The CNN converts a 2D image into a 1D sequence of visual slices.")
    
    # Simulate a single grayscale word image of height 48, width 128
    # Batch=1, Channel=1, Height=48, Width=128
    dummy_image = torch.randn(1, 1, 48, 128)
    print(f"1. Input Image Tensor: {tuple(dummy_image.shape)}")
    print("   [Batch=1, Channels=1 (Grayscale), Height=48 px, Width=128 px]")

    # A miniature CNN illustrating how height collapses while width becomes time steps
    cnn = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),          # 48x128 -> 24x64
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),          # 24x64  -> 12x32
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 1)),          # 12x32  -> 6x32 (height reduced, width preserved!)
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(6, 1)),          # 6x32   -> 1x32 (height collapsed to 1!)
    )

    feature_map = cnn(dummy_image)
    print(f"\n2. CNN Feature Map Output: {tuple(feature_map.shape)}")
    print("   [Batch=1, Features=512, Height=1, Time_Steps=32]")

    # Squeeze height: [1, 512, 1, 32] -> [1, 512, 32]
    # Permute to time-major sequence: [Time_Steps=32, Batch=1, Features=512]
    sequence = feature_map.squeeze(2).permute(2, 0, 1)
    print(f"\n3. Reshaped Sequence for RNN: {tuple(sequence.shape)}")
    print("   [Time_Steps=32, Batch=1, Features=512]")
    print("   -> Each of the 32 time steps represents one vertical slice of the word!")

# -------------------------------------------------------------
# Pillar 2: Why Bidirectional LSTM?
# -------------------------------------------------------------
def demo_bilstm():
    print_banner("PILLAR 2: WHY BIDIRECTIONAL LSTM?")
    print("Concept: Khmer characters have complex vertical and horizontal stacking.")
    print("Forward LSTM sees past strokes (left); Backward LSTM sees future strokes (right).")
    
    T, B, feat_dim = 32, 1, 512
    sequence = torch.randn(T, B, feat_dim)
    
    hidden_size = 256
    bilstm = nn.LSTM(
        input_size=feat_dim,
        hidden_size=hidden_size,
        num_layers=2,
        bidirectional=True
    )
    
    lstm_out, _ = bilstm(sequence)
    print(f"\n1. BiLSTM Input Shape:  {tuple(sequence.shape)} [T=32, B=1, Feat=512]")
    print(f"2. BiLSTM Output Shape: {tuple(lstm_out.shape)} [T=32, B=1, 2 x 256 = 512]")
    print("   -> At every time step t, the model has combined info from both directions:")
    print("      [forward_hidden (256) | backward_hidden (256)] = 512 dimensions.")

# -------------------------------------------------------------
# Pillar 3: Why CTC is Necessary
# -------------------------------------------------------------
def demo_ctc():
    print_banner("PILLAR 3: WHY CTC IS NECESSARY & HOW IT DECODES")
    print("Concept: Visual time steps (e.g. 14) are longer than target characters (5).")
    print("Word: ខ្មែរ (Unicode code points: ខ + ្ + ម + ែ + រ)")
    
    # Raw frame-by-frame predictions from the network (with '-' as blank token)
    raw_predictions = ['-', '-', 'ខ', 'ខ', '-', '្', '-', 'ម', 'ម', 'ែ', '-', 'រ', 'រ', '-']
    print(f"\n1. Raw Network Output ({len(raw_predictions)} time slices):")
    print("   " + " ".join(raw_predictions))
    
    # Step 1: Collapse consecutive identical characters
    collapsed = []
    prev = None
    for char in raw_predictions:
        if char != prev:
            collapsed.append(char)
            prev = char
    print("\n2. Step 1 - Collapse consecutive duplicates:")
    print("   " + " ".join(collapsed))
    
    # Step 2: Remove blank tokens '-'
    final_chars = [c for c in collapsed if c != '-']
    decoded_text = "".join(final_chars)
    print("\n3. Step 2 - Remove blank tokens ('-'):")
    print(f"   Final Khmer Unicode Output: '{decoded_text}'")
    print("   Match with ground truth: " + ("PASS" if decoded_text == "ខ្មែរ" else "FAIL"))

# -------------------------------------------------------------
# Pillar 4: Writer-Disjoint Split (No Writer Leakage)
# -------------------------------------------------------------
def demo_writer_split():
    print_banner("PILLAR 4: WRITER-DISJOINT SPLIT (NO WRITER LEAKAGE)")
    print("Concept: Writers in the test set must NEVER appear in the training set.")
    
    # Simulated metadata table
    records = [
        {"image": "001.png", "label": "សាលា", "writer_id": "W001"},
        {"image": "002.png", "label": "ខ្មែរ", "writer_id": "W001"},
        {"image": "003.png", "label": "សាលា", "writer_id": "W002"},
        {"image": "004.png", "label": "កម្ពុជា", "writer_id": "W002"},
        {"image": "005.png", "label": "សាលា", "writer_id": "W003"},
        {"image": "006.png", "label": "ខ្មែរ", "writer_id": "W003"},
        {"image": "007.png", "label": "សាលា", "writer_id": "W004"},
        {"image": "008.png", "label": "កម្ពុជា", "writer_id": "W004"},
    ]
    df = pd.DataFrame(records)
    print("\n1. Sample Metadata Records:")
    print(df.to_string(index=False))
    
    train_writers = {"W001", "W002"}
    val_writers   = {"W003"}
    test_writers  = {"W004"}
    
    train_df = df[df["writer_id"].isin(train_writers)]
    val_df   = df[df["writer_id"].isin(val_writers)]
    test_df  = df[df["writer_id"].isin(test_writers)]
    
    # Verify zero overlap
    train_set = set(train_df["writer_id"])
    test_set  = set(test_df["writer_id"])
    overlap   = train_set.intersection(test_set)
    
    print(f"\n2. Train Writers: {sorted(list(train_set))}")
    print(f"   Val Writers:   {sorted(list(set(val_df['writer_id'])))}")
    print(f"   Test Writers:  {sorted(list(test_set))}")
    print(f"3. Writer Overlap between Train & Test: {overlap} (Must be empty set)")
    assert len(overlap) == 0, "ERROR: Writer leakage detected!"
    print("   Verification: PASSED - Zero writer leakage guaranteed!")

if __name__ == "__main__":
    demo_cnn_role()
    demo_bilstm()
    demo_ctc()
    demo_writer_split()
    print("\n" + "=" * 65)
    print("  ALL 4 PILLARS SUCCESSFULLY DEMONSTRATED AND VERIFIED!")
    print("=" * 65 + "\n")

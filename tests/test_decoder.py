"""
CTC greedy decoder verification.

Run:
    .venv/bin/pytest tests/test_decoder.py -v
"""

import torch

from src.decoder import collapse_ctc_sequence, greedy_decode
from src.tokenizer import BLANK_INDEX, KhmerTokenizer


def test_collapse_matches_the_hand_worked_ctc_example():
    # Same example as tests/demo_day1_concepts.py's Pillar 3, now as a
    # real formal test instead of a printed demonstration.
    tokenizer = KhmerTokenizer.build_from_labels(["ខ្មែរ"])
    ids = {ch: tokenizer.char_to_idx[ch] for ch in "ខ្មែរ"}
    B = BLANK_INDEX

    raw_predictions = [B, B, ids["ខ"], ids["ខ"], B, ids["្"], B, ids["ម"], ids["ម"], ids["ែ"], B, ids["រ"], ids["រ"], B]
    clean = collapse_ctc_sequence(raw_predictions)

    assert tokenizer.decode(clean) == "ខ្មែរ"


def test_collapse_removes_all_blanks_even_with_no_repeats():
    assert collapse_ctc_sequence([BLANK_INDEX, 5, BLANK_INDEX, 7, BLANK_INDEX]) == [5, 7]


def test_collapse_merges_adjacent_repeats_but_not_repeats_separated_by_blank():
    # 5,5,5 -> one 5. But 5, blank, 5 means the SAME character occurred
    # twice on purpose (e.g. doubled letters) -- blank is what lets CTC
    # express "two of the same character in a row" instead of merging them.
    assert collapse_ctc_sequence([5, 5, 5]) == [5]
    assert collapse_ctc_sequence([5, BLANK_INDEX, 5]) == [5, 5]


def test_collapse_of_all_blanks_is_empty():
    assert collapse_ctc_sequence([BLANK_INDEX, BLANK_INDEX, BLANK_INDEX]) == []


def _one_hot_log_probs(sequences: list[list[int]], num_classes: int, confidence: float = 20.0) -> torch.Tensor:
    """Build a [T, B, num_classes] log_probs tensor where argmax at each
    time step is guaranteed to equal the given id (a large logit spike),
    so greedy_decode's behavior can be tested deterministically."""
    t = max(len(seq) for seq in sequences)
    b = len(sequences)
    logits = torch.zeros(t, b, num_classes)
    for i, seq in enumerate(sequences):
        for time_step, token_id in enumerate(seq):
            logits[time_step, i, token_id] = confidence
    return torch.log_softmax(logits, dim=2)


def test_greedy_decode_on_synthetic_logits():
    tokenizer = KhmerTokenizer.build_from_labels(["សាលា", "ខ្មែរ"])
    B = BLANK_INDEX

    sala_ids = tokenizer.encode("សាលា")   # ស, ា, ល, ា
    khmer_ids = tokenizer.encode("ខ្មែរ")  # ខ, ្, ម, ែ, រ

    # Deliberately include repeats and blanks the way a real model would.
    seq_sala = [sala_ids[0], sala_ids[0], B, sala_ids[1], sala_ids[2], sala_ids[2], B, sala_ids[3]]
    seq_khmer = [B, khmer_ids[0], khmer_ids[1], B, khmer_ids[2], khmer_ids[2], khmer_ids[3], khmer_ids[4], B]

    log_probs = _one_hot_log_probs([seq_sala, seq_khmer], num_classes=tokenizer.vocab_size)
    decoded = greedy_decode(log_probs, tokenizer)

    assert decoded == ["សាលា", "ខ្មែរ"]


def test_greedy_decode_truncates_to_input_lengths_ignoring_padding_garbage():
    tokenizer = KhmerTokenizer.build_from_labels(["ទឹក", "ភ្លើង"])
    tuk_ids = tokenizer.encode("ទឹក")

    # After the real content, stuff in garbage predictions that would
    # decode to something else entirely -- this simulates predictions
    # made over white-padding columns that must be ignored.
    garbage = [tuk_ids[0]] * 5
    full_sequence = tuk_ids + garbage

    log_probs = _one_hot_log_probs([full_sequence], num_classes=tokenizer.vocab_size)
    input_lengths = torch.tensor([len(tuk_ids)])

    decoded = greedy_decode(log_probs, tokenizer, input_lengths=input_lengths)
    assert decoded == ["ទឹក"]

    # Without truncation, the garbage tail changes nothing here (same
    # repeated character collapses to one more of the same char) --
    # so use a garbage tail with a DIFFERENT character to prove it matters.
    other_word_ids = tokenizer.encode("ភ្លើង")
    mixed_sequence = tuk_ids + other_word_ids
    log_probs2 = _one_hot_log_probs([mixed_sequence], num_classes=tokenizer.vocab_size)

    decoded_full = greedy_decode(log_probs2, tokenizer)  # no truncation
    decoded_truncated = greedy_decode(log_probs2, tokenizer, input_lengths=torch.tensor([len(tuk_ids)]))

    assert decoded_full == ["ទឹកភ្លើង"]
    assert decoded_truncated == ["ទឹក"]

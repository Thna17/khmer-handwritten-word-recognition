"""
CER / WER / Word Accuracy verification.

Run:
    .venv/bin/pytest tests/test_metrics.py -v
"""

import pytest

from src.metrics import character_error_rate, evaluate_predictions, word_accuracy, word_error_rate


def test_perfect_predictions_give_zero_error_and_full_accuracy():
    refs = ["សាលា", "ខ្មែរ", "កម្ពុជា"]
    hyps = ["សាលា", "ខ្មែរ", "កម្ពុជា"]
    assert character_error_rate(refs, hyps) == 0.0
    assert word_error_rate(refs, hyps) == 0.0
    assert word_accuracy(refs, hyps) == 1.0


def test_cer_counts_edits_over_total_reference_characters():
    # ខ្មែរ (5 codepoints) predicted as ខ្មរ (missing the ែ vowel: 1 deletion).
    # សាលា (4 codepoints) predicted perfectly.
    # Total reference chars = 5 + 4 = 9, total edits = 1 -> CER = 1/9.
    refs = ["ខ្មែរ", "សាលា"]
    hyps = ["ខ្មរ", "សាលា"]
    cer = character_error_rate(refs, hyps)
    assert cer == pytest.approx(1 / 9, abs=1e-6)


def test_word_accuracy_counts_exact_matches_only():
    refs = ["ទឹក", "ភ្លើង", "ខ្យល់", "ថ្ងៃ"]
    hyps = ["ទឹក", "ភ្លើង", "ខ្យល", "ផ្ទះ"]  # 2 correct, 2 wrong (one off by a diacritic)
    assert word_accuracy(refs, hyps) == pytest.approx(0.5)


def test_wer_equals_one_minus_word_accuracy_for_single_word_labels():
    # Every label in this project is exactly one word with no internal
    # spaces, so WER (word-sequence edit distance) must mathematically
    # coincide with 1 - word_accuracy. This is verified, not assumed.
    refs = ["ទឹក", "ភ្លើង", "ខ្យល់", "ថ្ងៃ", "ខែ"]
    hyps = ["ទឹក", "ភ្លើងx", "ខ្យល", "ថ្ងៃ", "khmer"]
    assert word_error_rate(refs, hyps) == pytest.approx(1 - word_accuracy(refs, hyps))


def test_evaluate_predictions_bundle_matches_individual_calls():
    refs = ["សាលា", "គ្រូ", "សិស្ស"]
    hyps = ["សាលា", "គ្រូ", "សិស"]
    bundle = evaluate_predictions(refs, hyps)
    assert bundle["cer"] == character_error_rate(refs, hyps)
    assert bundle["wer"] == word_error_rate(refs, hyps)
    assert bundle["word_accuracy"] == word_accuracy(refs, hyps)


def test_empty_input_raises_instead_of_returning_a_meaningless_number():
    with pytest.raises(ValueError):
        character_error_rate([], [])
    with pytest.raises(ValueError):
        word_accuracy([], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        word_accuracy(["ក", "ខ"], ["ក"])

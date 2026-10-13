"""Evaluation metrics for Khmer handwriting recognition: CER, WER, and
both whole-line and word-level accuracy.

CER and WER are computed with jiwer — this project's locked metrics
library (see PROJECT_CONTEXT.md) — rather than a hand-rolled edit-distance
implementation, since jiwer already implements this correctly and
reimplementing it would just duplicate an approved dependency.

TWO DIFFERENT "WORD ACCURACY" NUMBERS LIVE HERE, AND THE DIFFERENCE MATTERS
---------------------------------------------------------------------------
`word_accuracy` is EXACT WHOLE-SEQUENCE match: the fraction of samples
whose prediction equals the target character for character. That was the
right headline metric when every label was a single Khmer word, which is
what the synthetic word dataset contains. The real handwriting dataset is
not that: its labels are full LINES, a median of 58 characters and 3.7
space-separated words each. A line is scored correct only if all ~58
characters are right, so a genuinely useful model reading 7 of every 10
words still reports 0.00% here. That is exactly what the fine-tuning runs
reported, and it is a property of the metric, not only of the model.

`word_level_metrics` is the metric that matches the task name. It splits
both strings into word tokens, aligns the two token sequences, and reports
the share of reference words recovered, plus precision/recall/F1 over the
token multiset. Report BOTH: the exact-line number is the honest "can it
read a whole line perfectly" figure, and the word-level number is the
honest "what fraction of words does it get right" figure.

Punctuation is excluded from word-level scoring. 69 of the 811 tokens in
the real labels (8.5%) are bare punctuation, overwhelmingly ។ (the Khmer
full stop), and those are trivial to predict — counting them as recognized
"words" inflated the measured accuracy with no reading ability behind it.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

import jiwer


def _validate_inputs(references: list[str], hypotheses: list[str]) -> None:
    if len(references) == 0:
        raise ValueError("Cannot compute metrics over an empty set of samples.")
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must be the same length, got "
            f"{len(references)} vs {len(hypotheses)}."
        )


def character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level CER: total character edits / total reference characters."""
    _validate_inputs(references, hypotheses)
    return jiwer.cer(references, hypotheses)


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER over whole-word "sentences" (see module docstring)."""
    _validate_inputs(references, hypotheses)
    return jiwer.wer(references, hypotheses)


def word_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """Fraction of samples where the prediction exactly matches the target."""
    _validate_inputs(references, hypotheses)
    correct = sum(ref == hyp for ref, hyp in zip(references, hypotheses))
    return correct / len(references)


# Standalone punctuation: each of these ends or separates a word rather than
# belonging to one, so each is split off as its own token and then dropped.
# U+17D7 (ៗ, the repetition sign) is deliberately NOT here — it is part of the
# word form it follows, as in គ្រប់ៗ, and splitting it off would turn one
# correct word into one correct and one spurious token.
_PUNCTUATION = "។៖៕​.,:;!?\"'«»“”()[]-–—=/\\"
_SPLIT_PATTERN = re.compile(rf"[\s{re.escape(_PUNCTUATION)}]+")


def tokenize_words(text: str) -> list[str]:
    """Split a Khmer line into scoreable word tokens.

    Khmer does not put spaces between every word, so a space-separated
    token is really a phrase. That is fine for this comparison as long as
    reference and hypothesis are tokenized identically — what is measured is
    "did the model reproduce this run of text exactly", which is a strictly
    harder bar than per-word scoring with a Khmer word segmenter would be.
    """
    if not text:
        return []
    return [token for token in _SPLIT_PATTERN.split(text) if token]


def word_level_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Word-level recovery rate, precision, recall and F1 over word tokens.

    `word_level_accuracy` aligns the two token sequences with difflib and
    counts reference tokens that survive the alignment, so a single inserted
    or dropped word shifts the alignment rather than destroying every
    position after it — the failure mode a naive positional zip would have.
    """
    _validate_inputs(references, hypotheses)

    reference_total = 0
    hypothesis_total = 0
    aligned_matches = 0
    multiset_matches = 0

    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = tokenize_words(reference)
        hypothesis_tokens = tokenize_words(hypothesis)
        reference_total += len(reference_tokens)
        hypothesis_total += len(hypothesis_tokens)

        matcher = difflib.SequenceMatcher(None, reference_tokens, hypothesis_tokens, autojunk=False)
        aligned_matches += sum(size for _, _, size in matcher.get_matching_blocks())

        hypothesis_counts = Counter(hypothesis_tokens)
        for token, count in Counter(reference_tokens).items():
            multiset_matches += min(count, hypothesis_counts.get(token, 0))

    if reference_total == 0:
        raise ValueError("no scoreable word tokens in the references — every label was punctuation or empty.")

    precision = multiset_matches / hypothesis_total if hypothesis_total else 0.0
    recall = multiset_matches / reference_total
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "word_level_accuracy": aligned_matches / reference_total,
        "word_precision": precision,
        "word_recall": recall,
        "word_f1": f1,
        "reference_word_count": float(reference_total),
        "hypothesis_word_count": float(hypothesis_total),
    }


def evaluate_predictions(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Every metric for one run: character, whole-line, and word level.

    `word_accuracy` (exact whole line) and `word_level_accuracy` (share of
    word tokens recovered) are both present on purpose — see the module
    docstring for why reporting only the first reads as 0.00% on line data.
    """
    return {
        "cer": character_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
        "word_accuracy": word_accuracy(references, hypotheses),
        **word_level_metrics(references, hypotheses),
    }

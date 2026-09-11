"""Evaluation metrics for Khmer word recognition: CER, WER, Word Accuracy.

CER and WER are computed with jiwer — this project's locked metrics
library (see PROJECT_CONTEXT.md) — rather than a hand-rolled edit-distance
implementation, since jiwer already implements this correctly and
reimplementing it would just duplicate an approved dependency.

Note on WER for this task: every label here is exactly ONE Khmer word
(no spaces inside it), so at the word-sequence level each reference and
hypothesis is a "sentence" of length 1. That means WER mathematically
reduces to exactly `1 - word_accuracy` on this dataset — both are still
reported (via jiwer, not by formula substitution) because the course
rubric and OCR literature both expect WER by name, and computing it
properly keeps this file correct even if a future experiment ever
compares multi-word phrases.
"""

from __future__ import annotations

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


def evaluate_predictions(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Convenience bundle of all three metrics, for evaluate.py to call once per run."""
    return {
        "cer": character_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
        "word_accuracy": word_accuracy(references, hypotheses),
    }

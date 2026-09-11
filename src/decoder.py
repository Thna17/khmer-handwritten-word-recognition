"""CTC greedy decoding: turns a model's raw per-time-step predictions into
clean Khmer text.

Two cleanup steps, always in this order (see Phase 5 / Gate #3 — this is
the "why CTC needs decoding" concept, now implemented for real):
  1. Collapse consecutive repeated predictions — the model naturally
     predicts the same character across several adjacent time steps.
  2. Remove every <blank> token — CTC's "nothing new here" placeholder.

This is NOT the same job as tokenizer.decode(), which assumes its input
is already a clean list of character IDs with no repeats or blanks —
that assumption only holds AFTER this collapsing step has run.
"""

from __future__ import annotations

import torch

from src.tokenizer import BLANK_INDEX, KhmerTokenizer


def collapse_ctc_sequence(token_ids: list[int]) -> list[int]:
    """Step 1 + 2 of CTC greedy decoding on ONE already-argmax'd sequence
    of raw per-time-step predictions."""
    collapsed = []
    previous = None
    for token_id in token_ids:
        if token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    return [token_id for token_id in collapsed if token_id != BLANK_INDEX]


def greedy_decode(
    log_probs: torch.Tensor,
    tokenizer: KhmerTokenizer,
    input_lengths: torch.Tensor | None = None,
) -> list[str]:
    """log_probs: [T, B, num_classes] (a model's raw output). Returns one
    decoded Khmer string per batch item.

    If `input_lengths` is given, each sample's predictions are truncated
    to its own true (pre-padding) length first — predictions made over
    white-padding columns aren't meaningful and must be discarded, not
    decoded as if they were real content.
    """
    t, b, _ = log_probs.shape
    predicted_ids = log_probs.argmax(dim=2)  # [T, B] — greedy: most likely class per time step

    decoded_texts = []
    for i in range(b):
        length = int(input_lengths[i]) if input_lengths is not None else t
        sequence = predicted_ids[:length, i].tolist()
        clean_ids = collapse_ctc_sequence(sequence)
        decoded_texts.append(tokenizer.decode(clean_ids))
    return decoded_texts

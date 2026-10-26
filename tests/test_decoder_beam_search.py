"""Unit tests for CTC Prefix Beam Search & N-Gram Language Model."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.decoder_beam_search import (
    KhmerNGramLM,
    build_default_lm,
    ctc_prefix_beam_search_single,
)
from src.tokenizer import BLANK_INDEX, KhmerTokenizer


@pytest.fixture
def dummy_tokenizer():
    # 0 is blank, 1: ក, 2: ា, 3: រ, 4: ប, 5: ្ល, 6: ្រ
    char_to_idx = {"ក": 1, "ា": 2, "រ": 3, "ប": 4, "្ល": 5, "្រ": 6}
    return KhmerTokenizer(char_to_idx)


def test_ngram_lm_training_scoring_persistence():
    texts = ["ការ", "ការងារ", "ប្រធាន", "ការសិក្សា"]
    lm = KhmerNGramLM(n=3)
    lm.train(texts)

    assert lm.total_chars > 0
    assert "ក" in lm.vocab
    assert "រ" in lm.vocab

    # Score should be a negative log-prob float
    score = lm.score_char("កា", "រ")
    assert isinstance(score, float)
    assert score <= 0.0

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_lm.json"
        lm.save(path)
        loaded = KhmerNGramLM.load(path)
        assert loaded.n == lm.n
        assert loaded.total_chars == lm.total_chars
        assert loaded.score_char("កា", "រ") == pytest.approx(score)


def test_beam_search_deterministic_emissions(dummy_tokenizer):
    # Sequence length 4, vocab size 7
    # Time 0: emit 'ក' (idx 1)
    # Time 1: emit blank (idx 0)
    # Time 2: emit 'ា' (idx 2)
    # Time 3: emit 'រ' (idx 3)
    t_steps = 4
    num_classes = dummy_tokenizer.vocab_size
    logits = torch.full((t_steps, num_classes), -10.0)

    logits[0, 1] = 5.0  # ក
    logits[1, 0] = 5.0  # blank
    logits[2, 2] = 5.0  # ា
    logits[3, 3] = 5.0  # រ

    text, score = ctc_prefix_beam_search_single(
        logits, tokenizer=dummy_tokenizer, lm=None, beam_width=5
    )
    assert text == "ការ"
    assert isinstance(score, float)


def test_beam_search_repeated_character_with_blank(dummy_tokenizer):
    # Two consecutive identical tokens separated by a blank should not collapse
    # Time 0: 'ក' (1)
    # Time 1: blank (0)
    # Time 2: 'ក' (1)
    logits = torch.full((3, dummy_tokenizer.vocab_size), -10.0)
    logits[0, 1] = 6.0
    logits[1, 0] = 6.0
    logits[2, 1] = 6.0

    text, _ = ctc_prefix_beam_search_single(
        logits, tokenizer=dummy_tokenizer, lm=None, beam_width=5
    )
    assert text == "កក"


def test_beam_search_repeated_character_without_blank(dummy_tokenizer):
    # Two consecutive identical tokens WITHOUT a blank should collapse
    # Time 0: 'ក' (1)
    # Time 1: 'ក' (1)
    # Time 2: 'ា' (2)
    logits = torch.full((3, dummy_tokenizer.vocab_size), -10.0)
    logits[0, 1] = 6.0
    logits[1, 1] = 6.0
    logits[2, 2] = 6.0

    text, _ = ctc_prefix_beam_search_single(
        logits, tokenizer=dummy_tokenizer, lm=None, beam_width=5
    )
    assert text == "កា"

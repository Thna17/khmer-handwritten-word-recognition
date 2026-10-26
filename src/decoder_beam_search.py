"""CTC Prefix Beam Search Decoder with N-Gram Language Model Scoring for Khmer OCR.

Addresses visual twin confusions (e.g. ប្រ vs ប្ល, ច vs ថ, ព vs យ) by combining
CTC acoustic/visual probabilities with an n-gram character language model:

    TotalScore(prefix) = log P_CTC(prefix | X) + alpha * log P_LM(prefix) + beta * |prefix|

Includes:
1. `KhmerNGramLM`: Lightweight, dependency-free character/sub-syllable n-gram model
   with backoff smoothing, fast scoring, and JSON persistence.
2. `ctc_prefix_beam_search`: Prefix beam search decoder with blank-collapse and
   LM extension scoring.
3. `build_default_lm`: Pre-trains and caches an LM from project corpora.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer import BLANK_INDEX, KhmerTokenizer


# =============================================================================
# 1. LIGHTWEIGHT N-GRAM CHARACTER LANGUAGE MODEL
# =============================================================================
class KhmerNGramLM:
    """Character-level N-gram Language Model with backoff smoothing."""

    def __init__(self, n: int = 4, alpha: float = 0.4):
        self.n = n
        self.alpha = alpha  # backoff weight
        self.counts: dict[int, Counter[str]] = {i: Counter() for i in range(1, n + 1)}
        self.contexts: dict[int, Counter[str]] = {i: Counter() for i in range(1, n + 1)}
        self.vocab: set[str] = set()
        self.total_chars = 0

    def train(self, texts: list[str]) -> "KhmerNGramLM":
        """Trains n-gram counts on a list of Khmer strings."""
        for raw_text in texts:
            text = unicodedata.normalize("NFC", raw_text)
            if not text:
                continue
            self.total_chars += len(text)
            self.vocab.update(text)

            # Pad start with start-of-sequence token '^'
            padded = ("^" * (self.n - 1)) + text + "$"

            for order in range(1, self.n + 1):
                for i in range(len(padded) - order + 1):
                    ngram = padded[i : i + order]
                    context = ngram[:-1]
                    self.counts[order][ngram] += 1
                    if context:
                        self.contexts[order][context] += 1

        return self

    def score_char(self, context: str, char: str) -> float:
        """Computes log P(char | context) with backoff smoothing."""
        # Trim context to max (n - 1) characters
        ctx = ("^" * (self.n - 1) + context)[-(self.n - 1) :]

        for order in range(self.n, 1, -1):
            sub_ctx = ctx[-(order - 1) :]
            ngram = sub_ctx + char
            ctx_count = self.contexts[order].get(sub_ctx, 0)
            if ctx_count > 0:
                ngram_count = self.counts[order].get(ngram, 0)
                # Smoothed probability
                p = (ngram_count + 0.1) / (ctx_count + 0.1 * (len(self.vocab) + 1))
                return math.log(max(p, 1e-7))

        # Fallback to unigram
        unigram_count = self.counts[1].get(char, 0)
        p_uni = (unigram_count + 0.1) / (self.total_chars + 0.1 * (len(self.vocab) + 1))
        return math.log(max(p_uni, 1e-7))

    def score_sequence(self, text: str) -> float:
        """Sums log-probabilities across an entire string."""
        total_log_p = 0.0
        for i, ch in enumerate(text):
            ctx = text[:i]
            total_log_p += self.score_char(ctx, ch)
        return total_log_p

    def save(self, path: Path | str) -> None:
        """Saves n-gram counts and vocabulary to JSON."""
        data = {
            "n": self.n,
            "vocab": sorted(list(self.vocab)),
            "total_chars": self.total_chars,
            "counts": {str(k): dict(v) for k, v in self.counts.items()},
            "contexts": {str(k): dict(v) for k, v in self.contexts.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "KhmerNGramLM":
        """Loads a saved n-gram model from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lm = cls(n=data["n"])
        lm.vocab = set(data["vocab"])
        lm.total_chars = data["total_chars"]
        lm.counts = {int(k): Counter(v) for k, v in data["counts"].items()}
        lm.contexts = {int(k): Counter(v) for k, v in data["contexts"].items()}
        return lm


def build_default_lm(
    corpus_paths: list[Path] | None = None,
    save_path: Path | None = REPO_ROOT / "data" / "metadata" / "khmer_ngram_lm.json",
    order: int = 4,
) -> KhmerNGramLM:
    """Trains a default 4-gram LM from project datasets and caches it."""
    if save_path and save_path.is_file():
        try:
            return KhmerNGramLM.load(save_path)
        except Exception:
            pass

    if corpus_paths is None:
        corpus_paths = [
            REPO_ROOT / "data" / "metadata" / "labels.csv",
            REPO_ROOT / "data" / "synthetic" / "synthetic_labels.csv",
            REPO_ROOT / "data" / "soyvitou" / "soyvitou_labels.csv",
        ]

    texts: list[str] = []
    for cp in corpus_paths:
        if not cp.is_file():
            continue
        try:
            with open(cp, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[1].strip():
                        texts.append(row[1].strip())
        except Exception:
            pass

    lm = KhmerNGramLM(n=order).train(texts)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        lm.save(save_path)

    return lm


# =============================================================================
# 2. CTC PREFIX BEAM SEARCH DECODER
# =============================================================================
def ctc_prefix_beam_search_single(
    log_probs_t: torch.Tensor,
    tokenizer: KhmerTokenizer,
    lm: KhmerNGramLM | None = None,
    beam_width: int = 25,
    alpha: float = 0.6,
    beta: float = 1.0,
    blank_index: int = BLANK_INDEX,
) -> tuple[str, float]:
    """Prefix beam search for a single sequence [T, num_classes].

    Returns:
        (decoded_text, best_score)
    """
    t_steps, num_classes = log_probs_t.shape

    # Beam state: prefix_tuple -> (prob_blank, prob_non_blank, lm_log_p)
    # Prefix is a tuple of token IDs
    beam: dict[tuple[int, ...], tuple[float, float, float]] = {
        (): (1.0, 0.0, 0.0)
    }

    # Softmax probabilities to avoid underflow
    probs = torch.softmax(log_probs_t, dim=-1).cpu().numpy()

    for t in range(t_steps):
        p_t = probs[t]
        new_beam: dict[tuple[int, ...], tuple[float, float, float]] = defaultdict(
            lambda: (0.0, 0.0, 0.0)
        )

        # Only evaluate tokens with probability > 1e-4 to speed up search
        valid_indices = np.where(p_t > 1e-4)[0]

        for prefix, (p_b, p_nb, lm_score) in beam.items():
            p_total = p_b + p_nb

            # 1. Blank extension: prefix remains unchanged
            p_blank = float(p_t[blank_index])
            cur_pb, cur_pnb, cur_lm = new_beam[prefix]
            new_beam[prefix] = (cur_pb + p_total * p_blank, cur_pnb, lm_score)

            # 2. Character extension
            last_token = prefix[-1] if prefix else None
            for c in valid_indices:
                if c == blank_index:
                    continue

                prob_c = float(p_t[c])

                if c == last_token:
                    # Repeated token without blank in between stays in same prefix
                    pb, pnb, lms = new_beam[prefix]
                    new_beam[prefix] = (pb, pnb + p_nb * prob_c, lms)

                    # Repeated token WITH blank creates new extended prefix
                    ext_prefix = prefix + (int(c),)
                    ext_pb, ext_pnb, _ = new_beam[ext_prefix]

                    new_lm_score = lm_score
                    if lm is not None:
                        char_str = tokenizer.idx_to_char.get(int(c), "")
                        prefix_str = tokenizer.decode(list(prefix))
                        new_lm_score += lm.score_char(prefix_str, char_str)

                    new_beam[ext_prefix] = (ext_pb, ext_pnb + p_b * prob_c, new_lm_score)
                else:
                    ext_prefix = prefix + (int(c),)
                    ext_pb, ext_pnb, _ = new_beam[ext_prefix]

                    new_lm_score = lm_score
                    if lm is not None:
                        char_str = tokenizer.idx_to_char.get(int(c), "")
                        prefix_str = tokenizer.decode(list(prefix))
                        new_lm_score += lm.score_char(prefix_str, char_str)

                    new_beam[ext_prefix] = (ext_pb, ext_pnb + p_total * prob_c, new_lm_score)

        # Score and prune beam to top beam_width
        def score_prefix(item: tuple[tuple[int, ...], tuple[float, float, float]]) -> float:
            pref, (pb, pnb, lms) = item
            ctc_prob = max(pb + pnb, 1e-12)
            length_bonus = beta * len(pref)
            return math.log(ctc_prob) + alpha * lms + length_bonus

        sorted_items = sorted(new_beam.items(), key=score_prefix, reverse=True)
        beam = dict(sorted_items[:beam_width])

    # Pick the best overall hypothesis
    best_prefix, (best_pb, best_pnb, best_lm) = max(beam.items(), key=score_prefix)
    best_score = score_prefix((best_prefix, (best_pb, best_pnb, best_lm)))
    decoded_text = tokenizer.decode(list(best_prefix))
    return decoded_text, best_score


def beam_search_decode(
    log_probs: torch.Tensor,
    tokenizer: KhmerTokenizer,
    lm: KhmerNGramLM | None = None,
    beam_width: int = 25,
    alpha: float = 0.6,
    beta: float = 1.0,
    input_lengths: torch.Tensor | None = None,
) -> list[str]:
    """Decodes a batch of log_probs [T, B, num_classes] using CTC Prefix Beam Search.

    Args:
        log_probs: Model output tensor of shape [T, B, C].
        tokenizer: KhmerTokenizer instance.
        lm: Optional KhmerNGramLM language model.
        beam_width: Beam width for candidate pruning.
        alpha: Weight of Language Model score.
        beta: Sequence length bonus.
        input_lengths: Optional valid lengths [B] for variable-width padding truncation.

    Returns:
        List of decoded Khmer strings for each batch item.
    """
    t, b, _ = log_probs.shape
    decoded_texts: list[str] = []

    for i in range(b):
        length = int(input_lengths[i]) if input_lengths is not None else t
        item_log_probs = log_probs[:length, i, :]
        text, _ = ctc_prefix_beam_search_single(
            item_log_probs,
            tokenizer=tokenizer,
            lm=lm,
            beam_width=beam_width,
            alpha=alpha,
            beta=beta,
        )
        decoded_texts.append(text)

    return decoded_texts

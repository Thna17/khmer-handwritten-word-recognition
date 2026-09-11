"""Khmer Unicode tokenizer and vocabulary manager for CTC-based OCR.

Normalization policy: Unicode NFC (unicodedata.normalize("NFC", text)),
applied consistently to every label this tokenizer ever sees.

One token = one NFC-normalized Unicode code point (NOT one visible glyph —
a Khmer glyph is often a base consonant + a subscript sign (U+17D2) +
a vowel, i.e. multiple code points rendered as what looks like one shape).

Index 0 is reserved for the CTC <blank> token and is never assigned to a
real character. This tokenizer only handles Khmer text <-> token ID
conversion; collapsing repeated predictions and removing blanks from a
model's raw output is a separate concern, handled by decoder.py.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

BLANK_TOKEN = "<blank>"
BLANK_INDEX = 0


class KhmerTokenizer:
    def __init__(self, char_to_idx: dict[str, int]):
        if any(idx == BLANK_INDEX for idx in char_to_idx.values()):
            raise ValueError(f"Index {BLANK_INDEX} is reserved for {BLANK_TOKEN!r} and cannot be assigned to a character.")
        self.char_to_idx: dict[str, int] = dict(char_to_idx)
        self.idx_to_char: dict[int, str] = {idx: ch for ch, idx in self.char_to_idx.items()}

    @property
    def vocab_size(self) -> int:
        """Number of real characters plus 1 for the blank token."""
        return len(self.char_to_idx) + 1

    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @classmethod
    def build_from_labels(cls, labels: list[str]) -> "KhmerTokenizer":
        """Deterministically build a vocabulary from a list of label strings.

        Vocabulary order is sorted by Unicode code point value, NOT by
        first appearance in the data — so the same set of labels always
        produces the exact same char_to_idx mapping, regardless of the
        order rows happen to be read from a CSV.
        """
        unique_chars: set[str] = set()
        for label in labels:
            unique_chars.update(cls.normalize(label))

        sorted_chars = sorted(unique_chars)
        char_to_idx = {ch: idx + 1 for idx, ch in enumerate(sorted_chars)}  # 0 reserved for blank
        return cls(char_to_idx)

    def encode(self, text: str) -> list[int]:
        """Khmer text -> list of token IDs. Raises on any unknown character
        instead of silently dropping it (an unknown character means the
        vocabulary is stale and must be rebuilt, not ignored)."""
        normalized = self.normalize(text)
        ids = []
        for ch in normalized:
            if ch not in self.char_to_idx:
                raise ValueError(
                    f"Character {ch!r} (U+{ord(ch):04X}) in {text!r} is not in the "
                    "vocabulary. Rebuild the vocabulary with build_from_labels() "
                    "over the full dataset instead of skipping this character."
                )
            ids.append(self.char_to_idx[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        """List of token IDs -> Khmer text. `ids` must not contain the
        blank index or repeated-run artifacts from a raw model output —
        see decoder.py for turning raw CTC predictions into clean IDs."""
        chars = []
        for token_id in ids:
            if token_id not in self.idx_to_char:
                raise ValueError(f"Token id {token_id} is not in the vocabulary (blank index is {BLANK_INDEX}).")
            chars.append(self.idx_to_char[token_id])
        return "".join(chars)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "char_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(directory / "idx_to_char.json", "w", encoding="utf-8") as f:
            json.dump(
                {str(idx): ch for idx, ch in sorted(self.idx_to_char.items())},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "KhmerTokenizer":
        directory = Path(directory)
        with open(directory / "char_to_idx.json", encoding="utf-8") as f:
            char_to_idx = json.load(f)
        return cls(char_to_idx)

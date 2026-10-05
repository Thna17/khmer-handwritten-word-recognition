"""Khmer Orthographic Post-Processor.

Cleans up CTC decoding artifacts and enforces Khmer Unicode orthographic rules:
1. Collapses duplicated subscript (Coeng) signs (e.g., `្្` -> `្`).
2. Removes orphaned Coeng signs (at string start, after spaces, before spaces, or at string end).
3. Resolves conflicting consecutive dependent vowels.
4. Removes floating/orphaned diacritics.
5. Standardizes whitespace and ensures Unicode NFC normalization.
"""

from __future__ import annotations

import re
import unicodedata

# Character sets
KHMER_CONSONANTS = set("កខគឃងចឆជឈញដឋឌណតថទធនបផពភមយរលវសហឡអឥឩឧឳឬឬឯឰឱឲ")
KHMER_COENG = "្"  # U+17D2
KHMER_VOWELS = set("ាិីឹឺុូួើឿៀេែៃោៅ")
KHMER_DIACRITICS = set("ំះៈ៉៊់៌៍៎៏័")
KHMER_PUNCTUATION = set("។៕៖ៗ«»“”\"'.?:-=")


def clean_khmer_orthography(text: str) -> str:
    """Applies rule-based Khmer orthographic cleaning to a predicted string."""
    if not text:
        return ""

    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Collapse multiple consecutive Coeng signs
    text = re.sub(r"្+", "្", text)

    # Step 3: Remove orphaned Coeng at string start or after whitespace/punctuation
    text = re.sub(r"^្+", "", text)
    text = re.sub(r"([\s\.\,\?\!\«\»\“\”\៖\។\:\-\=])្+", r"\1", text)

    # Step 4: Remove orphaned Coeng at string end or before whitespace/punctuation
    text = re.sub(r"្+$", "", text)
    text = re.sub(r"្+([\s\.\,\?\!\«\»\“\”\៖\។\:\-\=])", r"\1", text)

    # Step 5: Remove Coeng that is NOT followed by a valid Khmer consonant or independent vowel
    # Coeng must be immediately followed by a consonant
    valid_sub_consonants = set("កខគឃងចឆជឈញដឋឌណតថទធនបផពភមយរលវសហឡអ")
    cleaned_chars = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == KHMER_COENG:
            # Lookahead: is next character a valid consonant?
            if i + 1 < n and text[i + 1] in valid_sub_consonants:
                cleaned_chars.append(c)
            # else skip this orphaned Coeng
        else:
            cleaned_chars.append(c)
        i += 1

    text = "".join(cleaned_chars)

    # Step 6: Remove floating diacritics preceded by space, punctuation, or start of string
    text = re.sub(r"^[ំះៈ៉៊់៌៍៎៏័]+", "", text)
    text = re.sub(r"([\s\.\,\?\!\«\»\“\”\៖\។\:\-\=])[ំះៈ៉៊់៌៍៎៏័]+", r"\1", text)

    # Step 7: Collapse duplicate consecutive vowels (keep the first valid one)
    # E.g. two consecutive vowels like ើា or ាិ -> keep first
    cleaned_chars = []
    prev_was_vowel = False
    for c in text:
        if c in KHMER_VOWELS:
            if not prev_was_vowel:
                cleaned_chars.append(c)
                prev_was_vowel = True
        else:
            cleaned_chars.append(c)
            prev_was_vowel = False

    text = "".join(cleaned_chars)

    # Step 8: Collapse multiple consecutive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def postprocess_predictions(predictions: list[str]) -> list[str]:
    """Convenience batch processor."""
    return [clean_khmer_orthography(p) for p in predictions]

"""Unicode diagnostics and normalization for Khmer script."""

import unicodedata
import re

# Khmer Unicode Block: U+1780 - U+17FF
KHMER_COENG = '\u17d2'  # ្

def normalize_khmer(text: str) -> str:
    """
    Normalizes Khmer text string according to standard Unicode NFC and Khmer canonical order.
    
    Policy:
    1. Unicode NFC normalization.
    2. Strip extraneous zero-width spaces (\u200b, \u200c) at borders while preserving internal semantics.
    3. Canonical reordering of Khmer combining marks if inverted:
       Base -> Coeng Sequence -> Dependent Vowel -> Diacritic.
    """
    if not text:
        return ""
    
    # Step 1: Standard NFC
    norm = unicodedata.normalize('NFC', text).strip()
    
    # Step 2: Remove leading/trailing zero-width spaces/joiners
    norm = re.sub(r'^[\u200b\u200c\u200d]+|[\u200b\u200c\u200d]+$', '', norm)
    
    # Step 3: Khmer specific canonical order adjustment for common misplaced pairs:
    # E.g. Nikahit (\u17c6) before Vowel Aa (\u17b6) -> should be \u17b6\u17c6 (ាំ)
    norm = norm.replace('\u17c6\u17b6', '\u17b6\u17c6')
    
    # Re-normalize NFC after fixes
    norm = unicodedata.normalize('NFC', norm)
    return norm

def get_unicode_codepoints(text: str) -> str:
    """Returns space-separated hex codepoint representation (e.g. U+179F U+17B6)."""
    return " ".join(f"U+{ord(c):04X}" for c in text)

def get_codepoint_details(text: str) -> list[dict]:
    """Returns detailed breakdown of each character in the string."""
    details = []
    for c in text:
        try:
            name = unicodedata.name(c)
        except ValueError:
            name = "UNKNOWN"
        details.append({
            "char": c,
            "hex": f"U+{ord(c):04X}",
            "name": name,
            "category": unicodedata.category(c)
        })
    return details

def format_diagnostic_report(word: str) -> str:
    """Generates a formatted diagnostic string for inspecting a Khmer word."""
    raw_cps = get_unicode_codepoints(word)
    norm_word = normalize_khmer(word)
    norm_cps = get_unicode_codepoints(norm_word)
    
    lines = [
        f"Raw Word:           {word}",
        f"Raw Code Points:    {raw_cps}",
        f"Normalized Word:    {norm_word}",
        f"Normalized Points:  {norm_cps}",
        "Character Breakdown:"
    ]
    for d in get_codepoint_details(word):
        lines.append(f"  {d['char']}  {d['hex']}  ({d['category']})  {d['name']}")
    return "\n".join(lines)

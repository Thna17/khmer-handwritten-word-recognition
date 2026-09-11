"""
Gate #4 verification: the Khmer tokenizer must round-trip every label
exactly: original Khmer -> encode -> token IDs -> decode -> same Khmer.

Run:
    .venv/bin/pytest tests/test_tokenizer.py -v
"""

import unicodedata

import pytest

from src.tokenizer import BLANK_INDEX, KhmerTokenizer

# Draft word list under review (see TASKS.md Week 1) — used here only to
# exercise the tokenizer, not yet the final locked project vocabulary.
SAMPLE_LABELS = [
    "ទឹក", "ភ្លើង", "ខ្យល់", "ថ្ងៃ", "ខែ", "ឆ្នាំ", "ផ្ទះ", "សាលា",
    "គ្រូ", "សៀវភៅ", "ខ្មៅដៃ", "តុ", "កៅអី", "ឆ្កែ", "ឆ្មា", "មាន់",
    "ត្រី", "ផ្កា", "ជ្រូក", "គោ", "ខ្មែរ", "កម្ពុជា", "គ្រួសារ", "មិត្ត",
    "ស្រី", "ប្រុស", "ចិត្ត", "សុខ", "ក្តៅ", "ធំ", "តូច",
]


@pytest.fixture(scope="module")
def tokenizer() -> KhmerTokenizer:
    return KhmerTokenizer.build_from_labels(SAMPLE_LABELS)


def test_blank_index_is_reserved_and_unassigned(tokenizer):
    assert BLANK_INDEX == 0
    assert BLANK_INDEX not in tokenizer.idx_to_char


def test_vocab_size_accounts_for_blank(tokenizer):
    assert tokenizer.vocab_size == len(tokenizer.char_to_idx) + 1


def test_vocab_build_is_deterministic_regardless_of_input_order():
    forward = KhmerTokenizer.build_from_labels(SAMPLE_LABELS)
    shuffled = KhmerTokenizer.build_from_labels(list(reversed(SAMPLE_LABELS)))
    assert forward.char_to_idx == shuffled.char_to_idx


@pytest.mark.parametrize("word", SAMPLE_LABELS)
def test_roundtrip_encode_decode(tokenizer, word):
    ids = tokenizer.encode(word)
    decoded = tokenizer.decode(ids)
    assert decoded == unicodedata.normalize("NFC", word)


def test_encode_rejects_unknown_character_instead_of_dropping_it(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode("Z")


def test_decode_rejects_unknown_token_id(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.decode([999999])


def test_visible_glyph_count_is_not_codepoint_count(tokenizer):
    # ខ្មែរ looks like ~3-4 visible glyphs but is 5 Unicode code points:
    # ខ (base) + ្ (U+17D2 subscript sign) + ម (subscript consonant) + ែ (vowel) + រ (base)
    word = "ខ្មែរ"
    normalized = unicodedata.normalize("NFC", word)
    assert len(normalized) == 5
    ids = tokenizer.encode(word)
    assert len(ids) == 5
    assert tokenizer.decode(ids) == normalized


def test_save_and_load_produce_identical_tokenizer(tmp_path, tokenizer):
    tokenizer.save(tmp_path)
    assert (tmp_path / "char_to_idx.json").exists()
    assert (tmp_path / "idx_to_char.json").exists()

    loaded = KhmerTokenizer.load(tmp_path)
    assert loaded.char_to_idx == tokenizer.char_to_idx

    for word in SAMPLE_LABELS:
        assert loaded.decode(loaded.encode(word)) == unicodedata.normalize("NFC", word)


def test_constructor_rejects_blank_index_collision():
    with pytest.raises(ValueError):
        KhmerTokenizer(char_to_idx={"ក": 0})

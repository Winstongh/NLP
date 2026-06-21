from __future__ import annotations

import re
from typing import Iterable, List


_EN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")


def tokenize_en(text: str) -> List[str]:
    """Tokenize English text into lowercase words, numbers, and punctuation."""

    return _EN_PATTERN.findall(normalize_text(text).lower())


def tokenize_zh(text: str) -> List[str]:
    """Tokenize Chinese text at character level, preserving ASCII word runs."""

    normalized = normalize_text(text)
    tokens: List[str] = []
    i = 0
    while i < len(normalized):
        ch = normalized[i]
        if ch.isspace():
            i += 1
            continue
        if _is_ascii_word_char(ch):
            j = i + 1
            while j < len(normalized) and _is_ascii_word_char(normalized[j]):
                j += 1
            tokens.append(normalized[i:j].lower())
            i = j
            continue
        tokens.append(ch)
        i += 1
    return tokens


def detokenize_zh(tokens: Iterable[str]) -> str:
    """Join Chinese character tokens into a readable sentence."""

    return "".join(tokens).strip()


def detokenize_en(tokens: Iterable[str]) -> str:
    """A simple English detokenizer for debugging and reports."""

    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    return text.strip()


def whitespace_tokenize(text: str) -> List[str]:
    """For data that is already tokenized (space-separated), e.g. cmn-eng-simple."""

    return str(text).split()


def join_zh_words(tokens: Iterable[str]) -> str:
    """Join word-segmented Chinese tokens into a sentence (no spaces)."""

    return "".join(tokens).strip()


def join_spaced(tokens: Iterable[str]) -> str:
    return " ".join(t for t in tokens if t).strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing language-specific characters."""

    return re.sub(r"\s+", " ", str(text).replace("\u3000", " ")).strip()


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in {"_", "-", "'"})

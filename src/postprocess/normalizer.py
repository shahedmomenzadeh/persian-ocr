"""Persian Unicode normalization, ZWNJ hygiene, and digit policy handling."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

DigitPolicy = Literal["persian", "latin", "preserve"]

ZWNJ = "\u200c"

# Arabic -> Persian canonical character mappings
_CHAR_MAP = str.maketrans(
    {
        "\u064a": "\u06cc",  # Arabic Yeh (ي) -> Persian Yeh (ی)
        "\u0649": "\u06cc",  # Alef Maksura (ى) -> Persian Yeh (ی)
        "\u0643": "\u06a9",  # Arabic Kaf (ك) -> Persian Keheh (ک)
        "\u0629": "\u0647",  # Teh Marbuta (ة) -> Heh (ه)
        "\u06c0": "\u0647\u0654",  # Heh with Yeh above (ۀ) -> هٔ
        "\u0626": "\u0626",  # Keep Yeh with Hamza above (ئ)
        "\u200f": "",  # Strip stray RLM control chars if injected
        "\u200e": "",  # Strip stray LRM control chars if injected
    }
)

_ASCII_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_ARABIC_INDIC_TO_PERSIAN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")
_ALL_TO_LATIN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Tokens that should keep ASCII digits even when digit_policy == "persian"
_LTR_PROTECTED_TOKEN_RE = re.compile(
    r"(?:"
    r"https?://\S+"
    r"|www\.\S+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|[A-Za-z_][A-Za-z0-9_./:-]*\d[A-Za-z0-9_./:-]*"
    r"|(?<![A-Za-z0-9])\d+(?:\.\d+)+(?:[A-Za-z0-9._-]*)"  # version numbers like 3.11 or 0.2.1
    r")"
)

_MULTI_ZWNJ_RE = re.compile(r"\u200c{2,}")
_SPACE_AROUND_ZWNJ_RE = re.compile(r"[ \t]*\u200c[ \t]*")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _convert_digits_persian_preserving_ltr(text: str) -> str:
    """Convert digits to Persian ۰-۹ while preserving digits inside Latin/URL/version tokens."""
    # First normalize Arabic-Indic digits everywhere outside protected tokens
    out_parts: list[str] = []
    last_end = 0

    for match in _LTR_PROTECTED_TOKEN_RE.finditer(text):
        start, end = match.span()
        if start > last_end:
            segment = text[last_end:start]
            segment = segment.translate(_ARABIC_INDIC_TO_PERSIAN_DIGITS)
            segment = segment.translate(_ASCII_TO_PERSIAN_DIGITS)
            out_parts.append(segment)
        out_parts.append(match.group(0))
        last_end = end

    if last_end < len(text):
        tail = text[last_end:]
        tail = tail.translate(_ARABIC_INDIC_TO_PERSIAN_DIGITS)
        tail = tail.translate(_ASCII_TO_PERSIAN_DIGITS)
        out_parts.append(tail)

    return "".join(out_parts)


def normalize_persian(text: str, digit_policy: DigitPolicy = "persian") -> str:
    """Normalize Persian text: NFC, Arabic->Persian chars, ZWNJ hygiene, and digit policy."""
    if not text:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Unify Arabic letters to Persian equivalents
    normalized = normalized.translate(_CHAR_MAP)

    # 3. Clean ZWNJ noise without deleting valid internal ZWNJ
    normalized = _MULTI_ZWNJ_RE.sub(ZWNJ, normalized)
    normalized = _SPACE_AROUND_ZWNJ_RE.sub(ZWNJ, normalized)

    # 4. Normalize whitespace per line while preserving line breaks
    lines: list[str] = []
    for raw_line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _MULTI_SPACE_RE.sub(" ", raw_line).strip(" \t\u200c")
        lines.append(line)
    normalized = "\n".join(lines).strip("\n")

    # 5. Apply digit policy
    if digit_policy == "persian":
        normalized = _convert_digits_persian_preserving_ltr(normalized)
    elif digit_policy == "latin":
        normalized = normalized.translate(_ALL_TO_LATIN_DIGITS)
    elif digit_policy == "preserve":
        pass
    else:
        raise ValueError(f"Unsupported digit_policy: {digit_policy!r}")

    return normalized

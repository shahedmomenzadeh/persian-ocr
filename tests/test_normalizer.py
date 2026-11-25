"""Unit tests for Persian Unicode normalization, ZWNJ hygiene, and digit policy."""

from __future__ import annotations

from src.postprocess.normalizer import ZWNJ, normalize_persian


def test_arabic_to_persian_char_unification() -> None:
    raw = "كتاب عربي و مدرسهٔ جميلة و علي"
    out = normalize_persian(raw, digit_policy="preserve")
    assert out == "کتاب عربی و مدرسهٔ جمیله و علی"
    assert "ي" not in out
    assert "ك" not in out
    assert "ة" not in out


def test_zwnj_preserved_and_surrounding_spaces_cleaned() -> None:
    # Valid internal ZWNJ must survive, while accidental spaces around ZWNJ or repeated ZWNJs collapse
    raw = f"می {ZWNJ} خواهم   کتاب{ZWNJ}{ZWNJ} ها را بخوانم و نیم{ZWNJ}فاصله حفظ شود"
    out = normalize_persian(raw, digit_policy="preserve")
    assert out == f"می{ZWNJ}خواهم کتاب{ZWNJ}ها را بخوانم و نیم{ZWNJ}فاصله حفظ شود"


def test_double_spaces_and_newlines_cleaned() -> None:
    raw = "  خط   اول  \n  خط    دوم  "
    out = normalize_persian(raw, digit_policy="preserve")
    assert out == "خط اول\nخط دوم"


def test_digit_policies_with_mixed_rtl_ltr() -> None:
    sample = "نسخه Python 3.11 در سال 1404 و شماره ٤٢ منتشر شد."

    persian = normalize_persian(sample, digit_policy="persian")
    assert "Python 3.11" in persian
    assert "۱۴۰۴" in persian
    assert "۴۲" in persian

    latin = normalize_persian("سال ۱۴۰۴ و رقم ٤٢", digit_policy="latin")
    assert latin == "سال 1404 و رقم 42"

    preserved = normalize_persian("سال ۱۴۰۴ و 2025", digit_policy="preserve")
    assert preserved == "سال ۱۴۰۴ و 2025"


def test_urls_and_emails_not_corrupted_under_persian_digits() -> None:
    sample = "آدرس https://example.com/v2/page3 و ایمیل dev99@test.ir در تاریخ 2026/01/05 ثبت شد"
    out = normalize_persian(sample, digit_policy="persian")
    assert "https://example.com/v2/page3" in out
    assert "dev99@test.ir" in out
    assert "۲۰۲۶/۰۱/۰۵" in out

"""Unit tests for Phase 1 Persian OCR-weighted lexicon spellchecker."""

from __future__ import annotations

from src.postprocess.normalizer import ZWNJ
from src.postprocess.spellcheck import (
    PersianLexicon,
    PersianSpellchecker,
    get_spellchecker,
    weighted_ocr_distance,
)


def test_weighted_ocr_distance_prefers_visual_dot_swaps() -> None:
    # گ↔ک visual dot confusion costs 0.25 vs generic substitution 1.0
    assert weighted_ocr_distance("گشور", "کشور") == 0.25
    assert weighted_ocr_distance("ازدبیل", "اردبیل") == 0.25
    assert weighted_ocr_distance("مشهد", "تهران") > 1.5


def test_spellcheck_fixes_real_bina02_visual_errors() -> None:
    checker = get_spellchecker("lexicon")
    assert checker is not None

    samples = {
        "سازمان ثبت اسناد و املاگ گشور": "سازمان ثبت اسناد و املاک کشور",
        "انحاقیه دفترچه": "الحاقیه دفترچه",
        "استان ازدبیل": "استان اردبیل",
        "سهمیه ۵ درصد ابثارگران": "سهمیه ۵ درصد ایثارگران",
        "سهمیه ۳ درصد معلوان": "سهمیه ۳ درصد معلولان",
    }
    for raw, expected in samples.items():
        fixed, corrs = checker.correct_text(raw, page_number=1)
        assert fixed == expected
        assert len(corrs) >= 1


def test_spellcheck_zwnj_and_adjacent_split_join() -> None:
    checker = get_spellchecker("lexicon")
    assert checker is not None

    fixed_zwnj, corrs_zwnj = checker.correct_text(f"ظرفیت شغلمحل{ZWNJ}ها اضافه شد", page_number=2)
    assert fixed_zwnj == f"ظرفیت شغل{ZWNJ}محل{ZWNJ}ها اضافه شد"
    assert any(c.rule == "zwnj_join" for c in corrs_zwnj)

    fixed_split, corrs_split = checker.correct_text("کارشناس نقشه برد اری", page_number=4)
    assert fixed_split == f"کارشناس نقشه{ZWNJ}برداری"
    assert any(c.rule == "split_join" for c in corrs_split)


def test_spellcheck_safety_guards_preserve_numbers_urls_cities_and_punct() -> None:
    checker = get_spellchecker("lexicon")
    assert checker is not None

    protected_lines = [
        "شماره ۲۶۰۸ مورخ ۱۴۰۰/۰۱/۲۳ در تاریخ ۱۴۰۳/۰۳/۱۸",
        "سایت https://hrtc.ir/v2 و نسخه Python 3.11",
        "شهرستان‌های کوهبنان و فشافویه و بوانات و بجنورد و ایذه و طبس",
        "کارشناس (قراردادی) و (پیمانی) از به در با",
    ]
    for line in protected_lines:
        fixed, corrs = checker.correct_text(line, page_number=1)
        assert fixed == line
        assert corrs == []


def test_spellcheck_off_returns_none() -> None:
    assert get_spellchecker("off") is None


def test_kenlm_mode_gracefully_falls_back_to_lexicon_when_missing() -> None:
    checker = get_spellchecker("kenlm", kenlm_path="/nonexistent/model.klm")
    assert isinstance(checker, PersianSpellchecker)
    fixed, _ = checker.correct_text("گشور")
    assert fixed == "کشور"

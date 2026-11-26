"""Unit and integration tests for Phase 1 & Phase 2 Persian OCR spellchecker."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_kenlm_real_word_rescoring_respects_confidence_and_margin() -> None:
    lexicon = PersianLexicon.build()

    class _MockKenLM:
        def __init__(self, gain: float = 4.0) -> None:
            self.gain = gain

        def BeginSentenceWrite(self, state) -> None:  # noqa: N802
            return None

        def BaseScore(self, prev_state, word: str, out_state) -> float:  # noqa: N802
            if word == "اسناد":
                return -1.0
            if word == "استاد":
                return -1.0 - self.gain
            return -2.0

    checker = PersianSpellchecker(
        lexicon=lexicon,
        mode="kenlm",
        kenlm_model=_MockKenLM(gain=4.0),
        real_word_conf_threshold=0.94,
        real_word_lm_margin=2.0,
    )

    fixed_low, corrs_low = checker.correct_text("سازمان ثبت استاد", page_number=1, line_confidence=0.90)
    assert fixed_low == "سازمان ثبت اسناد"
    assert any(c.rule == "kenlm_context" and c.original == "استاد" and c.corrected == "اسناد" for c in corrs_low)

    fixed_high, corrs_high = checker.correct_text("سازمان ثبت استاد", page_number=1, line_confidence=0.98)
    assert fixed_high == "سازمان ثبت استاد"
    assert corrs_high == []

    weak_checker = PersianSpellchecker(
        lexicon=lexicon,
        mode="kenlm",
        kenlm_model=_MockKenLM(gain=1.0),  # Below 2.0 margin
        real_word_conf_threshold=0.94,
        real_word_lm_margin=2.0,
    )
    fixed_weak, corrs_weak = weak_checker.correct_text("سازمان ثبت استاد", page_number=1, line_confidence=0.90)
    assert fixed_weak == "سازمان ثبت استاد"
    assert corrs_weak == []


@pytest.mark.skipif(
    not Path("models/kenlm/persian_5gram_4bit.binary").is_file(),
    reason="Persian 5-gram KenLM binary not downloaded",
)
def test_real_persian_kenlm_fixes_contextual_and_oov_errors() -> None:
    checker = get_spellchecker(
        "kenlm",
        kenlm_path="models/kenlm/persian_5gram_4bit.binary",
        real_word_conf_threshold=0.94,
    )
    assert checker is not None

    fixed1, corrs1 = checker.correct_text("سازمان ثبت استاد", page_number=1, line_confidence=0.90)
    assert fixed1 == "سازمان ثبت اسناد"
    assert any(c.rule == "kenlm_context" for c in corrs1)

    fixed2, _ = checker.correct_text("و املاگ گشور", page_number=1, line_confidence=0.90)
    assert fixed2 == "و املاک کشور"

    clean = "زمان اعلام نتایج آزمون اوایل تیرماه خواهد بود"
    fixed3, corrs3 = checker.correct_text(clean, page_number=2, line_confidence=0.90)
    assert fixed3 == clean
    assert corrs3 == []

    fixed4, corrs4 = checker.correct_text("ارزباب گمرک", page_number=7, line_confidence=0.80)
    assert fixed4 == "ارزیاب گمرک"
    assert any(c.original == "ارزباب" and c.corrected == "ارزیاب" for c in corrs4)

    fixed5, corrs5 = checker.correct_text("نسبت به موارد ذیل معترض هستید", page_number=1, line_confidence=0.80)
    assert fixed5 == "نسبت به موارد ذیل معترض هستید"
    assert corrs5 == []


def test_spellcheck_preserves_tanvin_and_fixes_adjacent_typo() -> None:
    lexicon = PersianLexicon.build()
    for mode in ("lexicon", "kenlm"):
        if mode == "kenlm" and not Path("models/kenlm/persian_5gram_4bit.binary").is_file():
            continue
        checker = get_spellchecker(mode, kenlm_path="models/kenlm/persian_5gram_4bit.binary")
        assert checker is not None

        tanvin_line = "ضمناً اکیداً صرفاً لطفاً معمولاً"
        fixed_clean, corrs_clean = checker.correct_text(tanvin_line, page_number=1, line_confidence=0.88)
        assert fixed_clean == tanvin_line
        assert corrs_clean == []

        fixed_mixed, corrs_mixed = checker.correct_text("ضمناً گشور", page_number=1, line_confidence=0.88)
        assert fixed_mixed == "ضمناً کشور"
        assert [c.original for c in corrs_mixed] == ["گشور"]

    assert lexicon.is_known("ضمناً")
    assert lexicon.is_known("نمایندگان")


def test_spellcheck_preserves_iranian_geography_domain_terms_and_morphology() -> None:
    checker = get_spellchecker("lexicon")
    assert checker is not None

    clean_lines = [
        "گمرک سومار و شهرستان خنداب و مرزن آباد",
        "ارزیاب گمرک و مداد پاکن و پیجر",
        "مهندسی صنایع دخانی و امتیاز حافظین قرآن",
        "بندپایان و جوندگان و بیماری‌های منتقله",
        "مراجعین و ناظرین و نمایندگان",
    ]
    for line in clean_lines:
        fixed, corrs = checker.correct_text(line, page_number=1)
        assert fixed == line
        assert corrs == []

    fixed_err, corrs_err = checker.correct_text("ارزباب گمرک", page_number=7)
    assert fixed_err == "ارزیاب گمرک"
    assert len(corrs_err) == 1
    assert corrs_err[0].corrected == "ارزیاب"

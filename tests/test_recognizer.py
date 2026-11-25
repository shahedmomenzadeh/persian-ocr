"""Unit tests for pluggable recognizer dispatch, Bina-0.2 helpers, and Nemotron prompt routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.settings import Settings
from src.inference import recognizer as rec_mod
from src.inference.recognizer import (
    BinaNemotronRecognizer,
    _bucket_by_resized_width,
    _patch_nemotron_cache_compat,
    _prepare_crop_orientation,
    _resolve_bina_paddle_dir,
    _split_wide_line,
    create_recognizer,
)


def test_prepare_crop_orientation_rotates_tall_crops_only() -> None:
    tall = np.full((120, 30, 3), 255, dtype=np.uint8)
    rotated = _prepare_crop_orientation(tall)
    assert rotated.shape[:2] == (30, 120)

    wide = np.full((48, 300, 3), 255, dtype=np.uint8)
    untouched = _prepare_crop_orientation(wide)
    assert untouched.shape[:2] == (48, 300)


def test_split_wide_line_splits_at_whitespace_in_rtl_order() -> None:
    narrow = np.full((48, 400, 3), 255, dtype=np.uint8)
    assert len(_split_wide_line(narrow, max_wh_ratio=14.0)) == 1

    wide = np.full((48, 1400, 3), 255, dtype=np.uint8)
    wide[10:38, 40:680] = (180, 20, 20)
    wide[10:38, 720:1360] = (20, 20, 180)
    segments = _split_wide_line(wide, max_wh_ratio=14.0, target_chunk_ratio=12.0)
    assert len(segments) == 3

    assert segments[0][:, :, 2].mean() > segments[0][:, :, 0].mean()
    assert segments[-1][:, :, 0].mean() > segments[-1][:, :, 2].mean()


def test_bucket_by_resized_width_groups_uniform_and_isolates_outliers() -> None:
    segs = [
        np.full((48, 200, 3), 255, dtype=np.uint8),
        np.full((48, 600, 3), 255, dtype=np.uint8),
        np.full((48, 750, 3), 255, dtype=np.uint8),
        np.full((48, 1400, 3), 255, dtype=np.uint8),
    ]
    buckets_bs4 = _bucket_by_resized_width(segs, batch_size=4)
    assert buckets_bs4 == [[0, 1, 2], [3]]

    buckets_bs1 = _bucket_by_resized_width(segs, batch_size=1)
    assert buckets_bs1 == [[0], [1], [2], [3]]


def test_resolve_bina_paddle_dir_selects_variant() -> None:
    s1 = Settings(rec_model_variant="bina-0.2-rizehpizeh", bina_model_dir=Path("./models/bina-0.2"))
    d1, lbl1 = _resolve_bina_paddle_dir(s1)
    assert lbl1 == "Bina-0.2-RizehPizeh"
    assert d1.name == "bina-0.2"

    s2 = Settings(rec_model_variant="bina-0.2-rizeh", bina_rizeh_model_dir=Path("./models/bina-0.2-rizeh"))
    d2, lbl2 = _resolve_bina_paddle_dir(s2)
    assert lbl2 == "Bina-0.2-Rizeh"
    assert d2.name == "bina-0.2-rizeh"


def test_create_recognizer_dispatches_by_variant(monkeypatch) -> None:
    seen: list[str] = []

    class _FakePaddle:
        @classmethod
        def get_instance(cls, settings: Settings):
            seen.append(f"paddle:{settings.rec_model_variant}")
            return cls()

    class _FakeNemotron:
        @classmethod
        def get_instance(cls, settings: Settings):
            seen.append(f"nemotron:{settings.rec_model_variant}")
            return cls()

    monkeypatch.setattr(rec_mod, "BinaRecognizer", _FakePaddle)
    monkeypatch.setattr(rec_mod, "BinaNemotronRecognizer", _FakeNemotron)

    create_recognizer(Settings(rec_model_variant="bina-0.2-rizehpizeh"))
    create_recognizer(Settings(rec_model_variant="bina-0.2-rizeh"))
    create_recognizer(Settings(rec_model_variant="bina-0.4-koochik"))

    assert seen == [
        "paddle:bina-0.2-rizehpizeh",
        "paddle:bina-0.2-rizeh",
        "nemotron:bina-0.4-koochik",
    ]


def test_nemotron_prompt_selection_and_recognize_region() -> None:
    assert BinaNemotronRecognizer.select_prompt("table") == "</s><s><output_markdown><predict_text_in_pic>"
    assert BinaNemotronRecognizer.select_prompt("table_cell") == "</s><s><predict_text_in_pic>"
    assert BinaNemotronRecognizer.select_prompt("text") == "</s><s><predict_text_in_pic>"

    # Bypass __init__ to unit-test recognize_region logic cleanly without weights
    instance = BinaNemotronRecognizer.__new__(BinaNemotronRecognizer)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    class _FakeTorch:
        def inference_mode(self):
            return _NoGrad()

    captured_prompts: list[str] = []
    captured_max_tokens: list[int] = []

    def _fake_generate(model, proc, pil_img, prompt, gen_config):
        captured_prompts.append(prompt)
        captured_max_tokens.append(gen_config["max_new_tokens"])
        return {
            "selected": {
                "ocr_prediction": "متن آزمایشی",
                "ended_with_eos": True,
                "capped": False,
                "signal": {"flagged": False},
            }
        }

    instance._torch = _FakeTorch()
    instance._model = object()
    instance._processor = object()
    instance._gen_config = {"max_new_tokens": 256}
    instance._generate_repaired = _fake_generate

    crop = np.full((64, 200, 3), 255, dtype=np.uint8)
    txt, conf = instance.recognize_region(crop, block_type="table")
    assert txt == "متن آزمایشی"
    assert conf == 1.0
    assert captured_prompts == ["</s><s><output_markdown><predict_text_in_pic>"]
    assert captured_max_tokens == [256]

    txt2, conf2 = instance.recognize_region(crop, block_type="table_cell", max_new_tokens=128)
    assert txt2 == "متن آزمایشی"
    assert conf2 == 1.0
    assert captured_prompts[-1] == "</s><s><predict_text_in_pic>"
    assert captured_max_tokens[-1] == 128


def test_patch_nemotron_cache_compat_forwards_cache_kwarg() -> None:
    seen: list[tuple[object, object, object]] = []

    class _DummyModel:
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
            seen.append((input_ids, past_key_values, kwargs.get("pixel_values")))
            return {"input_ids": input_ids, "past_key_values": past_key_values}

    model = _DummyModel()
    _patch_nemotron_cache_compat(model)

    fake_cache = object()
    explicit_pkv = object()

    res1 = model.prepare_inputs_for_generation("ids1", cache=fake_cache, pixel_values="px")
    assert res1["past_key_values"] is fake_cache

    res2 = model.prepare_inputs_for_generation(
        "ids2", past_key_values=explicit_pkv, cache=fake_cache, pixel_values=None
    )
    assert res2["past_key_values"] is explicit_pkv
    assert seen == [("ids1", fake_cache, "px"), ("ids2", explicit_pkv, None)]

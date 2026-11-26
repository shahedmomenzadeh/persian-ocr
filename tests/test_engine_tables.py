"""Unit tests for SLANet_plus table-cell recognition in region mode and ink filtering."""

from __future__ import annotations

import numpy as np

from config.settings import Settings
from src.inference.detector import DetectedLine
from src.inference.engine import (
    InferenceEngine,
    _cell_has_ink,
    _clean_vlm_cell_text,
    _is_valid_vlm_table_markdown,
    _prepare_vlm_cell_crop,
)
from src.inference.layout import LayoutBox, TableStructureOutput


def test_cell_has_ink_filters_blank_and_border_only_cells() -> None:
    blank_white = np.full((60, 80, 3), 255, dtype=np.uint8)
    assert _cell_has_ink(blank_white) is False

    blank_gray = np.full((60, 80, 3), 220, dtype=np.uint8)
    assert _cell_has_ink(blank_gray) is False

    blank_gold = np.full((60, 80, 3), (40, 160, 200), dtype=np.uint8)
    assert _cell_has_ink(blank_gold) is False

    # Gridlines only on the outer 2px border should be ignored
    border_only = np.full((60, 80, 3), 245, dtype=np.uint8)
    border_only[:2, :] = 10
    border_only[-2:, :] = 10
    border_only[:, :2] = 10
    border_only[:, -2:] = 10
    assert _cell_has_ink(border_only) is False

    # Tiny single-digit glyph inside interior must be preserved
    single_digit = np.full((60, 80, 3), 245, dtype=np.uint8)
    single_digit[24:36, 38:41] = 20
    assert _cell_has_ink(single_digit) is True


def test_region_mode_uses_slanet_cells_skips_blank_and_flips_rtl() -> None:
    # 2x2 table at y=[0..100], x=[0..200]; 3 cells have ink, bottom-left (col 0 row 1) is blank
    img = np.full((120, 220, 3), 250, dtype=np.uint8)
    img[15:35, 30:70] = 20    # row 0, visual left (col_idx 0 -> RTL col 1)
    img[15:35, 130:170] = 20  # row 0, visual right (col_idx 1 -> RTL col 0)
    img[65:85, 130:170] = 20  # row 1, visual right (col_idx 1 -> RTL col 0)

    class _FakeLayoutAnalyzer:
        loaded_model_names = ["PP-DocLayout_plus-L", "SLANet_plus"]

        def analyze_table(self, table_crop: np.ndarray) -> TableStructureOutput:
            tokens = [
                "<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
                "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>",
            ]
            bboxes = [
                (0.0, 0.0, 100.0, 50.0),
                (100.0, 0.0, 200.0, 50.0),
                (0.0, 50.0, 100.0, 100.0),
                (100.0, 50.0, 200.0, 100.0),
            ]
            return TableStructureOutput(structure_tokens=tokens, cell_bboxes=bboxes)

    calls: list[tuple[str, int | None]] = []
    cell_texts = iter(["ستون دوم", "ستون اول", "مقدار"])

    class _FakeRegionRecognizer:
        model_label = "Bina-0.4-Koochik"
        granularity = "region"

        def recognize_region(
            self,
            crop: np.ndarray,
            block_type: str = "text",
            max_new_tokens: int | None = None,
        ) -> tuple[str, float]:
            calls.append((block_type, max_new_tokens))
            return (next(cell_texts), 1.0)

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.settings = Settings(rec_model_variant="bina-0.4-koochik", device="cpu", enable_tables=True)
    engine.layout_analyzer = _FakeLayoutAnalyzer()  # type: ignore[assignment]
    engine.recognizer = _FakeRegionRecognizer()  # type: ignore[assignment]
    engine.text_detector = None

    layout_boxes = [
        LayoutBox(type="table", raw_label="table", bbox=(4.0, 4.0, 204.0, 104.0), confidence=0.95)
    ]
    blocks, timings = engine._process_region_mode(1, img, layout_boxes)

    assert len(blocks) == 1
    blk = blocks[0]
    # Only 3 inked cells should be sent to Koochik (the blank cell is skipped)
    assert calls == [("table_cell", 128), ("table_cell", 128), ("table_cell", 128)]
    assert len(blk.cells) == 4
    assert blk.html is not None and '<table dir="rtl">' in blk.html
    assert "ستون اول" in blk.html and "ستون دوم" in blk.html and "مقدار" in blk.html

    # Visual-right cell (x=100..200) must now be RTL col_start=0
    row0_cols = {c.col_start: c.text for c in blk.cells if c.row_start == 0}
    assert row0_cols == {0: "ستون اول", 1: "ستون دوم"}
    assert timings["table_sec"] >= 0.0


def test_vlm_cell_crop_hygiene_and_text_cleanup() -> None:
    # 20x60 cell with thick dark outer gridlines on the 2px rim and blank white interior
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    img[2:22, 2:62] = 255
    img[2:4, 2:62] = 10
    img[20:22, 2:62] = 10
    img[2:22, 2:4] = 10
    img[2:22, 60:62] = 10

    assert _prepare_vlm_cell_crop(img, (2.0, 2.0, 62.0, 22.0)) is None

    # Add interior ink glyph inside the trimmed region and confirm padding wraps it cleanly
    img[9:15, 25:35] = 20
    prepared = _prepare_vlm_cell_crop(img, (2.0, 2.0, 62.0, 22.0))
    assert prepared is not None
    assert prepared.shape == (28, 68, 3)  # (20-4+12, 60-4+12)
    assert int(prepared[0, 0].min()) > 240  # uniform padded background on rim

    dirty = "| کارشناس | نقشه برداری | --- | </chart> \\mbox{} \\text{۱۳۷} ______ |"
    assert _clean_vlm_cell_text(dirty) == "کارشناس نقشه برداری ۱۳۷"


def test_hybrid_mode_routes_tables_to_vlm_and_simple_text_to_ctc() -> None:
    img = np.full((320, 400, 3), 250, dtype=np.uint8)
    # Ink in text block (top) and 2x2 table (bottom)
    img[20:40, 220:360] = 20
    img[160:180, 60:140] = 20
    img[160:180, 260:340] = 20
    img[230:250, 260:340] = 20

    class _FakeHybridLayout:
        loaded_model_names = ["PP-DocLayout_plus-L", "SLANet_plus"]

        def analyze_table(self, table_crop: np.ndarray) -> TableStructureOutput:
            tokens = [
                "<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
                "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>",
            ]
            bboxes = [
                (0.0, 0.0, 180.0, 60.0),
                (180.0, 0.0, 360.0, 60.0),
                (0.0, 60.0, 180.0, 120.0),
                (180.0, 60.0, 360.0, 120.0),
            ]
            return TableStructureOutput(structure_tokens=tokens, cell_bboxes=bboxes)

    class _FakeDetector:
        def detect(self, bgr_img: np.ndarray) -> list[DetectedLine]:
            return [
                # Simple text line inside block 0
                DetectedLine(bbox=(210.0, 18.0, 370.0, 42.0), polygon=[[210, 18], [370, 18], [370, 42], [210, 42]], score=0.95),
                # Table line inside block 1 (must be ignored by CTC/unassigned fallback!)
                DetectedLine(bbox=(250.0, 158.0, 350.0, 182.0), polygon=[[250, 158], [350, 158], [350, 182], [250, 182]], score=0.95),
            ]

        def extract_crop(self, bgr_img: np.ndarray, polygon, bbox) -> np.ndarray:
            return bgr_img[18:42, 210:370]

    ctc_calls: list[int] = []
    vlm_calls: list[str] = []

    class _FakeCTC:
        model_label = "Bina-0.2-Rizeh"
        granularity = "line"

        def recognize_batch(self, crops, batch_size=None):
            ctc_calls.append(len(crops))
            return [("سلام ایران", 0.96) for _ in crops]

    vlm_calls = 0

    class _FakeVLM:
        model_label = "Bina-0.4-Koochik"
        granularity = "region"

        def recognize_region(self, crop, block_type="text", max_new_tokens=None):
            nonlocal vlm_calls
            vlm_calls += 1
            return (
                "| ردیف | عنوان |\n| --- | --- |\n| ۱ | کارشناس |",
                1.0,
            )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.settings = Settings(
        rec_model_variant="hybrid",
        device="cpu",
        enable_tables=True,
        hybrid_vlm_max_table_rows=8,
        hybrid_vlm_max_table_height_px=550,
    )
    engine.is_hybrid = True
    engine.layout_analyzer = _FakeHybridLayout()  # type: ignore[assignment]
    engine.text_detector = _FakeDetector()  # type: ignore[assignment]
    engine.recognizer = _FakeCTC()  # type: ignore[assignment]
    engine._vlm_recognizer = _FakeVLM()  # type: ignore[assignment]

    layout_boxes = [
        LayoutBox(type="text", raw_label="text", bbox=(200.0, 10.0, 380.0, 60.0), confidence=0.95),
        LayoutBox(type="table", raw_label="table", bbox=(24.0, 154.0, 384.0, 274.0), confidence=0.96),
    ]
    blocks, timings = engine._process_hybrid_mode(1, img, layout_boxes)

    assert len(blocks) == 2
    assert ctc_calls == [1]  # Only the text block line; compact VLM table lines skipped
    assert vlm_calls == 1
    assert blocks[0].recognizer == "Bina-0.2-Rizeh"
    assert blocks[0].routing_reason == "ctc_simple"
    assert blocks[0].text == "سلام ایران"
    assert blocks[1].recognizer == "Bina-0.4-Koochik"
    assert blocks[1].routing_reason == "table_vlm_compact"
    assert blocks[1].html is not None and "کارشناس" in blocks[1].html
    # Logical RTL col 0 must stay col 0 without double flipping
    hdr = {c.col_start: c.text for c in blocks[1].cells if c.row_start == 0}
    assert hdr == {0: "ردیف", 1: "عنوان"}
    assert timings["ctc_blocks"] == 1
    assert timings["vlm_blocks"] == 1


def test_ctc_cell_ink_recovery_and_hybrid_tall_and_boilerplate_fallbacks() -> None:
    # Image with ink in: 2x2 table (top-left cell only detected; bottom-right digit missed by det) + missed footer
    img = np.full((340, 400, 3), 250, dtype=np.uint8)
    img[30:55, 220:340] = 20   # table row 0, visual right
    img[95:115, 60:110] = 20   # table row 1, visual left (missed by det -> recovered via cell_ink)
    img[300:320, 190:210] = 20 # footer (missed by det -> recovered via block_ink)

    class _TableLayout:
        def __init__(self, rows: int) -> None:
            self.rows = rows

        def analyze_table(self, table_crop: np.ndarray) -> TableStructureOutput:
            if self.rows == 2:
                tokens = ["<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
                          "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>"]
                bboxes = [(0.0, 0.0, 180.0, 60.0), (180.0, 0.0, 360.0, 60.0),
                          (0.0, 60.0, 180.0, 120.0), (180.0, 60.0, 360.0, 120.0)]
            else:
                tokens = ["<thead>", "<tr>", "<td>", "</td>", "</tr>", "</thead>", "<tbody>"]
                bboxes = [(0.0, 0.0, 360.0, 12.0)]
                for r in range(1, self.rows):
                    tokens.extend(["<tr>", "<td>", "</td>", "</tr>"])
                    bboxes.append((0.0, float(r * 12), 360.0, float((r + 1) * 12)))
                tokens.append("</tbody>")
            return TableStructureOutput(structure_tokens=tokens, cell_bboxes=bboxes)

    class _SparseDetector:
        def detect(self, bgr_img: np.ndarray) -> list[DetectedLine]:
            return [
                DetectedLine(bbox=(210.0, 28.0, 350.0, 55.0), polygon=[[210, 28], [350, 28], [350, 55], [210, 55]], score=0.95),
                DetectedLine(bbox=(212.0, 30.0, 348.0, 53.0), polygon=[[212, 30], [348, 30], [348, 53], [212, 53]], score=0.95),
            ]

        def extract_crop(self, bgr_img: np.ndarray, polygon, bbox) -> np.ndarray:
            return bgr_img[28:55, 210:350]

    class _BatchCTC:
        model_label = "Bina-0.2-Rizeh"
        granularity = "line"

        def recognize_batch(self, crops, batch_size=None):
            out = []
            for idx, _ in enumerate(crops):
                out.append((f"ک{idx}", 0.96))
            return out

    # 1. Pure line mode recovers missed table cell + missed footer inside one recognize_batch call
    eng_line = InferenceEngine.__new__(InferenceEngine)
    eng_line.settings = Settings(rec_model_variant="bina-0.2-rizeh", device="cpu", enable_tables=True, enable_cell_ink_recovery=True)
    eng_line.is_hybrid = False
    eng_line.layout_analyzer = _TableLayout(rows=2)  # type: ignore[assignment]
    eng_line.text_detector = _SparseDetector()  # type: ignore[assignment]
    eng_line.recognizer = _BatchCTC()  # type: ignore[assignment]
    eng_line._vlm_recognizer = None

    boxes = [
        LayoutBox(type="table", raw_label="table", bbox=(24.0, 20.0, 384.0, 140.0), confidence=0.95),
        LayoutBox(type="footer", raw_label="footer", bbox=(180.0, 290.0, 220.0, 330.0), confidence=0.95),
    ]
    line_blocks, _ = eng_line._process_line_mode(1, img, list(boxes))
    assert line_blocks[0].html is not None
    assert "ک0" in line_blocks[0].html and "ک2" in line_blocks[0].html  # ক0,ک1 detected + ک2 recovered cell
    assert line_blocks[1].text == "ک3"  # footer recovered in the same batch

    # 2. Markdown validator rejects synthetic CSS boilerplate and preserves clean Markdown
    assert _is_valid_vlm_table_markdown("<!doctype html><style>.page{}</style>", 1.0, (0, 0, 100, 100)) == []
    valid_cells = _is_valid_vlm_table_markdown("| الف | ب |\n| --- | --- |\n| ۱ | ۲ |", 1.0, (0, 0, 100, 100))
    assert len(valid_cells) == 4 and valid_cells[0].text == "الف"

    # 3. Hybrid mode rejects CSS boilerplate from compact table VLM -> falls back to CTC table
    class _CssBoilerplateVLM:
        model_label = "Bina-0.4-Koochik"
        granularity = "region"

        def recognize_region(self, crop, block_type="text", max_new_tokens=None):
            return ("<!doctype html><style>.page{width:1800px}</style>", 1.0)

    eng_hyb = InferenceEngine.__new__(InferenceEngine)
    eng_hyb.settings = Settings(rec_model_variant="hybrid", device="cpu", enable_tables=True, hybrid_vlm_max_table_rows=8)
    eng_hyb.is_hybrid = True
    eng_hyb.layout_analyzer = _TableLayout(rows=2)  # type: ignore[assignment]
    eng_hyb.text_detector = _SparseDetector()  # type: ignore[assignment]
    eng_hyb.recognizer = _BatchCTC()  # type: ignore[assignment]
    eng_hyb._vlm_recognizer = _CssBoilerplateVLM()  # type: ignore[assignment]
    hyb_fallback_blocks, _ = eng_hyb._process_hybrid_mode(1, img, [boxes[0]])
    assert hyb_fallback_blocks[0].routing_reason == "table_ctc_fallback"
    assert hyb_fallback_blocks[0].recognizer == "Bina-0.2-Rizeh"

    # 4. Hybrid mode routes tall table (>8 rows) directly to CTC without ever touching VLM
    eng_hyb.layout_analyzer = _TableLayout(rows=12)  # type: ignore[assignment]
    eng_hyb._vlm_recognizer = None
    eng_hyb._get_vlm_recognizer = lambda: (_ for _ in ()).throw(AssertionError("VLM must not run on tall tables"))  # type: ignore[method-assign]
    hyb_tall_blocks, _ = eng_hyb._process_hybrid_mode(1, img, [boxes[0]])
    assert hyb_tall_blocks[0].routing_reason == "table_ctc_tall"
    assert hyb_tall_blocks[0].recognizer == "Bina-0.2-Rizeh"


def test_hybrid_mode_skips_lazy_vlm_on_simple_page_and_falls_back_on_low_conf() -> None:
    img = np.full((120, 300, 3), 250, dtype=np.uint8)
    img[20:40, 20:280] = 20

    class _FakeDetector:
        def detect(self, bgr_img: np.ndarray) -> list[DetectedLine]:
            return [
                DetectedLine(bbox=(20.0, 20.0, 280.0, 45.0), polygon=[[20, 20], [280, 20], [280, 45], [20, 45]], score=0.95),
            ]

        def extract_crop(self, bgr_img: np.ndarray, polygon, bbox) -> np.ndarray:
            return bgr_img[20:45, 20:280]

    class _FakeCTC:
        def __init__(self, conf: float) -> None:
            self.conf = conf
            self.model_label = "Bina-0.2-Rizeh"
            self.granularity = "line"

        def recognize_batch(self, crops, batch_size=None):
            return [("متن نامطمئن", self.conf) for _ in crops]

    # 1. High-conf simple page must never load or call VLM
    engine_hi = InferenceEngine.__new__(InferenceEngine)
    engine_hi.settings = Settings(rec_model_variant="hybrid", device="cpu", hybrid_ctc_conf_threshold=0.85)
    engine_hi.is_hybrid = True
    engine_hi.layout_analyzer = object()  # type: ignore[assignment]
    engine_hi.text_detector = _FakeDetector()  # type: ignore[assignment]
    engine_hi.recognizer = _FakeCTC(conf=0.95)  # type: ignore[assignment]
    engine_hi._vlm_recognizer = None
    vlm_init_calls = 0

    def _fail_if_called():
        nonlocal vlm_init_calls
        vlm_init_calls += 1
        raise AssertionError("VLM should not be initialized on simple high-confidence pages")

    engine_hi._get_vlm_recognizer = _fail_if_called  # type: ignore[method-assign]
    boxes = [LayoutBox(type="text", raw_label="text", bbox=(10.0, 10.0, 290.0, 60.0), confidence=0.95)]
    blocks_hi, timings_hi = engine_hi._process_hybrid_mode(1, img, boxes)
    assert vlm_init_calls == 0
    assert timings_hi["ctc_blocks"] == 1 and timings_hi["vlm_blocks"] == 0
    assert blocks_hi[0].routing_reason == "ctc_simple"

    # 2. Low-conf CTC triggers VLM fallback (accepted if conf >= 0.65, rejected if flagged 0.30)
    class _FakeVLM:
        def __init__(self, txt: str, conf: float) -> None:
            self.txt = txt
            self.conf = conf
            self.model_label = "Bina-0.4-Koochik"
            self.granularity = "region"

        def recognize_region(self, crop, block_type="text", max_new_tokens=None):
            return (self.txt, self.conf)

    engine_lo = InferenceEngine.__new__(InferenceEngine)
    engine_lo.settings = Settings(rec_model_variant="hybrid", device="cpu", hybrid_ctc_conf_threshold=0.85)
    engine_lo.is_hybrid = True
    engine_lo.layout_analyzer = object()  # type: ignore[assignment]
    engine_lo.text_detector = _FakeDetector()  # type: ignore[assignment]
    engine_lo.recognizer = _FakeCTC(conf=0.60)  # type: ignore[assignment]
    engine_lo._vlm_recognizer = _FakeVLM("متن اصلاح‌شده با مدل بینا", 1.0)  # type: ignore[assignment]

    blocks_lo, timings_lo = engine_lo._process_hybrid_mode(1, img, boxes)
    assert blocks_lo[0].recognizer == "Bina-0.4-Koochik"
    assert blocks_lo[0].routing_reason == "ctc_low_conf"
    assert blocks_lo[0].text == "متن اصلاح‌شده با مدل بینا"
    assert timings_lo["vlm_blocks"] == 1

    # Flagged low-conf VLM loop keeps CTC text instead
    engine_lo._vlm_recognizer = _FakeVLM("تکرار تکرار تکرار", 0.30)  # type: ignore[assignment]
    blocks_flagged, timings_flagged = engine_lo._process_hybrid_mode(1, img, boxes)
    assert blocks_flagged[0].recognizer == "Bina-0.2-Rizeh"
    assert blocks_flagged[0].routing_reason == "ctc_simple"
    assert blocks_flagged[0].text == "متن نامطمئن"
    assert timings_flagged["ctc_blocks"] == 1 and timings_flagged["vlm_blocks"] == 0

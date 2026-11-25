"""Unit tests for SLANet_plus table-cell recognition in region mode and ink filtering."""

from __future__ import annotations

import numpy as np

from config.settings import Settings
from src.inference.engine import InferenceEngine, _cell_has_ink
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

"""Unit tests for RTL reading-order sorting (blocks and lines)."""

from __future__ import annotations

from src.postprocess.rtl_order import sort_blocks_rtl, sort_lines_rtl
from src.schemas.document import LayoutBlock, TextLine


def _block(
    bid: str,
    btype: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    text: str = "",
) -> LayoutBlock:
    return LayoutBlock(
        block_id=bid,
        type=btype,  # type: ignore[arg-type]
        bbox=(x1, y1, x2, y2),
        text=text or bid,
    )


def test_two_column_rtl_with_spanning_title_header_footer() -> None:
    blocks = [
        _block("footer", "footer", 100, 940, 900, 980, "صفحه ۱"),
        _block("left_top", "text", 50, 200, 450, 400, "ستون چپ بالا"),
        _block("right_bottom", "text", 550, 420, 950, 620, "ستون راست پایین"),
        _block("title", "title", 80, 90, 920, 160, "عنوان اصلی"),
        _block("right_top", "text", 550, 200, 950, 400, "ستون راست بالا"),
        _block("left_bottom", "text", 50, 420, 450, 620, "ستون چپ پایین"),
        _block("header", "header", 100, 20, 900, 60, "سربرگ"),
    ]

    ordered = sort_blocks_rtl(blocks, page_width=1000)
    ids = [b.block_id for b in ordered]
    assert ids == [
        "header",
        "title",
        "right_top",
        "right_bottom",
        "left_top",
        "left_bottom",
        "footer",
    ]
    assert [b.reading_order for b in ordered] == list(range(len(ordered)))


def test_three_column_rtl_band_splitting() -> None:
    # Band 1: 3 columns, then a wide table, then Band 2: 2 columns
    blocks = [
        _block("b1_left", "text", 40, 100, 300, 250),
        _block("b1_mid", "text", 350, 100, 650, 250),
        _block("b1_right", "text", 700, 100, 960, 250),
        _block("wide_table", "table", 60, 300, 940, 500),
        _block("b2_left", "text", 60, 550, 460, 750),
        _block("b2_right", "text", 540, 550, 940, 750),
    ]

    ordered = sort_blocks_rtl(blocks, page_width=1000)
    assert [b.block_id for b in ordered] == [
        "b1_right",
        "b1_mid",
        "b1_left",
        "wide_table",
        "b2_right",
        "b2_left",
    ]


def test_sort_lines_rtl_same_row_and_multiple_rows() -> None:
    lines = [
        TextLine(text="چپ بالا", bbox=(50, 10, 200, 40), confidence=0.9),
        TextLine(text="راست بالا", bbox=(300, 12, 480, 42), confidence=0.9),
        TextLine(text="چپ پایین", bbox=(50, 70, 200, 100), confidence=0.9),
        TextLine(text="راست پایین", bbox=(300, 68, 480, 98), confidence=0.9),
    ]

    sorted_lines = sort_lines_rtl(lines)
    assert [ln.text for ln in sorted_lines] == [
        "راست بالا",
        "چپ بالا",
        "راست پایین",
        "چپ پایین",
    ]

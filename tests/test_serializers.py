"""Unit tests for JSON, Markdown, RTL HTML, and plain text serializers."""

from __future__ import annotations

import json

import pytest

from src.postprocess.serializers import (
    parse_formats,
    to_html,
    to_json,
    to_markdown,
    to_text,
)
from src.schemas.document import (
    DocumentMetrics,
    DocumentResult,
    LayoutBlock,
    PageMetrics,
    PageResult,
    RunMetadata,
    StageTimings,
    TableCell,
)


def _sample_doc(with_metadata: bool = False) -> DocumentResult:
    cells = [
        TableCell(row_start=0, row_end=1, col_start=0, col_end=1, bbox=(0, 0, 100, 30), text="نام", is_header=True),
        TableCell(row_start=0, row_end=1, col_start=1, col_end=2, bbox=(100, 0, 200, 30), text="نسخه", is_header=True),
        TableCell(row_start=1, row_end=2, col_start=0, col_end=1, bbox=(0, 30, 100, 60), text="مدل بینا"),
        TableCell(row_start=1, row_end=2, col_start=1, col_end=2, bbox=(100, 30, 200, 60), text="v0.2"),
    ]
    blocks = [
        LayoutBlock(
            block_id="b0",
            type="title",
            bbox=(50, 20, 750, 70),
            reading_order=0,
            text="گزارش فنی نیم‌فاصله",
        ),
        LayoutBlock(
            block_id="b1",
            type="text",
            bbox=(50, 90, 750, 160),
            reading_order=1,
            text="استفاده از Python 3.11 در پردازش متن فارسی.",
        ),
        LayoutBlock(
            block_id="b2",
            type="table",
            bbox=(50, 200, 750, 320),
            reading_order=2,
            cells=cells,
        ),
    ]
    metadata = (
        RunMetadata(
            created_at="2026-11-25T12:00:00+00:00",
            rec_model_variant="bina-0.2-rizehpizeh",
            rec_model_label="Bina-0.2-RizehPizeh",
            granularity="line",
            layout_model="PP-DocLayout_plus-L",
            det_model="PP-OCRv6_medium_det",
            table_model="SLANet_plus",
            device="gpu",
            gpu_name="RTX 4060",
            dpi=200,
            digit_policy="persian",
        )
        if with_metadata
        else None
    )
    metrics = (
        DocumentMetrics(
            total_sec=1.25,
            mean_page_latency_sec=1.2,
            peak_vram_mb=505.4,
            peak_rss_ram_mb=2400.0,
            total_blocks=3,
            total_lines=2,
            total_tables=1,
            total_chars=60,
        )
        if with_metadata
        else None
    )
    page_metrics = (
        PageMetrics(
            latency_sec=1.2,
            peak_vram_mb=505.4,
            rss_ram_mb=2400.0,
            stages=StageTimings(layout_sec=0.3, detect_sec=0.2, recognize_sec=0.5),
            num_blocks=3,
        )
        if with_metadata
        else None
    )
    return DocumentResult(
        doc_id="doc-1",
        source_filename="sample.pdf",
        page_count=1,
        pages=[PageResult(page_number=1, width=800, height=1000, blocks=blocks, metrics=page_metrics)],
        processing_time_sec=1.25,
        metadata=metadata,
        metrics=metrics,
    )


def test_parse_formats_handles_commas_spaces_aliases_and_errors() -> None:
    assert parse_formats("html md txt") == ["html", "md", "txt"]
    assert parse_formats(["html,markdown", "text", "html"]) == ["html", "md", "txt"]
    assert parse_formats(None, default=("json", "html")) == ["json", "html"]
    with pytest.raises(ValueError, match="Unsupported export formats"):
        parse_formats("html pdf")


def test_to_json_preserves_utf8_persian() -> None:
    doc = _sample_doc()
    raw = to_json(doc)
    parsed = json.loads(raw)
    assert parsed["doc_id"] == "doc-1"
    assert parsed["pages"][0]["blocks"][0]["text"] == "گزارش فنی نیم‌فاصله"


def test_to_markdown_renders_headings_tables_and_frontmatter() -> None:
    doc = _sample_doc(with_metadata=True)
    md = to_markdown(doc)
    assert md.startswith("---\n")
    assert 'model: "Bina-0.2-RizehPizeh"' in md
    assert "## گزارش فنی نیم‌فاصله" in md
    assert "استفاده از Python 3.11 در پردازش متن فارسی." in md
    assert "| نام | نسخه |" in md
    assert "| مدل بینا | v0.2 |" in md


def test_to_text_renders_clean_rtl_text_and_tsv_tables() -> None:
    doc = _sample_doc()
    txt = to_text(doc)
    assert "گزارش فنی نیم‌فاصله" in txt
    assert "نام\tنسخه" in txt
    assert "مدل بینا\tv0.2" in txt


def test_to_html_sets_rtl_isolates_ltr_and_includes_provenance() -> None:
    doc = _sample_doc(with_metadata=True)
    html_out = to_html(doc)
    assert '<html lang="fa" dir="rtl">' in html_out
    assert '<meta name="ocr-model" content="Bina-0.2-RizehPizeh">' in html_out
    assert '<meta name="ocr-model-variant" content="bina-0.2-rizehpizeh">' in html_out
    assert '<header class="run-meta" dir="rtl">' in html_out
    assert '<div class="page-meta">' in html_out
    assert '<h2 dir="rtl">گزارش فنی نیم‌فاصله</h2>' in html_out
    assert '<span dir="ltr">Python 3.11</span>' in html_out
    assert '<table dir="rtl">' in html_out
    assert '<span dir="ltr">v0.2</span>' in html_out


def test_html_sanitizes_nested_region_html_and_converts_markdown_tables() -> None:
    nested_html = (
        "<!doctype html><html><head><style>body{}</style></head>"
        "<body><table><tr><td>سلول</td></tr></table></body></html>"
    )
    md_table = "| ستون اول | ستون دوم |\n| --- | --- |\n| الف | ب |"
    doc = DocumentResult(
        doc_id="doc-2",
        source_filename="tables.png",
        page_count=1,
        pages=[
            PageResult(
                page_number=1,
                width=500,
                height=500,
                blocks=[
                    LayoutBlock(
                        block_id="b0",
                        type="table",
                        bbox=(0, 0, 500, 200),
                        reading_order=0,
                        text=nested_html,
                        html=nested_html,
                    ),
                    LayoutBlock(
                        block_id="b1",
                        type="table",
                        bbox=(0, 220, 500, 450),
                        reading_order=1,
                        text=md_table,
                    ),
                ],
            )
        ],
    )
    html_out = to_html(doc)
    assert html_out.count("<!DOCTYPE html>") == 1
    assert "<table><tr><td>سلول</td></tr></table>" in html_out
    assert "<th>ستون اول</th><th>ستون دوم</th>" in html_out
    assert "<td>الف</td><td>ب</td>" in html_out

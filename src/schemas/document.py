"""Canonical output schemas for documents, pages, layout blocks, lines, and tables."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
BlockType = Literal["title", "text", "table", "figure", "header", "footer"]


class TextLine(BaseModel):
    """Single detected and recognized text line."""

    text: str = ""
    bbox: BBox
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.0


class TableCell(BaseModel):
    """Structured table cell mapped from table structure + OCR text lines."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    bbox: BBox
    text: str = ""
    is_header: bool = False
    confidence: float = 0.0


class LayoutBlock(BaseModel):
    """Semantic document region in RTL reading order."""

    block_id: str
    type: BlockType
    bbox: BBox
    reading_order: int = 0
    text: str = ""
    html: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    confidence: float = 0.0
    lines: list[TextLine] = Field(default_factory=list)


class StageTimings(BaseModel):
    """Per-stage wall-clock timings in seconds."""

    preprocess_sec: float = 0.0
    layout_sec: float = 0.0
    detect_sec: float = 0.0
    table_sec: float = 0.0
    recognize_sec: float = 0.0
    postprocess_sec: float = 0.0
    spellcheck_sec: float = 0.0


class SpellcheckCorrection(BaseModel):
    """Single OCR spellcheck rewrite recorded for auditability."""

    page_number: int = 0
    original: str
    corrected: str
    cost: float = 0.0
    rule: str = "visual_lexicon"


class PageMetrics(BaseModel):
    """Per-page latency, stage breakdown, counts, and memory telemetry."""

    latency_sec: float
    peak_vram_mb: float | None = None
    rss_ram_mb: float | None = None
    stages: StageTimings = Field(default_factory=StageTimings)
    num_blocks: int = 0
    num_lines: int = 0
    num_tables: int = 0
    char_count: int = 0
    corrections_count: int = 0


class PageResult(BaseModel):
    """OCR result for a single document page."""

    page_number: int
    width: int
    height: int
    blocks: list[LayoutBlock] = Field(default_factory=list)
    metrics: PageMetrics | None = None


class RunMetadata(BaseModel):
    """Provenance and runtime configuration for an OCR document run."""

    created_at: str
    rec_model_variant: str
    rec_model_label: str
    granularity: Literal["line", "region", "native"] = "line"
    layout_model: str
    det_model: str | None = None
    table_model: str | None = None
    device: str
    gpu_name: str | None = None
    dpi: int = 200
    digit_policy: str = "persian"
    spellcheck_mode: str = "off"
    pipeline_version: str = "0.1.0"


class DocumentMetrics(BaseModel):
    """Document-level aggregated latency, memory, stage timings, and counts."""

    total_sec: float = 0.0
    mean_page_latency_sec: float = 0.0
    peak_vram_mb: float | None = None
    peak_rss_ram_mb: float | None = None
    total_blocks: int = 0
    total_lines: int = 0
    total_tables: int = 0
    total_chars: int = 0
    total_corrections: int = 0
    corrections: list[SpellcheckCorrection] = Field(default_factory=list)
    stage_totals: StageTimings = Field(default_factory=StageTimings)


class DocumentResult(BaseModel):
    """Canonical top-level OCR output for an entire document."""

    doc_id: str
    source_filename: str
    page_count: int
    pages: list[PageResult] = Field(default_factory=list)
    processing_time_sec: float = 0.0
    metadata: RunMetadata | None = None
    metrics: DocumentMetrics | None = None


class RunManifest(BaseModel):
    """Structured run manifest saved alongside generated output artifacts."""

    doc_id: str
    source_filename: str
    page_count: int
    metadata: RunMetadata | None = None
    metrics: DocumentMetrics | None = None
    outputs: dict[str, str] = Field(default_factory=dict)

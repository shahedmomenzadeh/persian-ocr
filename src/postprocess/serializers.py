"""Serializers converting canonical DocumentResult to JSON, Markdown, RTL HTML, and plain text."""

from __future__ import annotations

import html
import re
from typing import Sequence

from src.schemas.document import DocumentResult, LayoutBlock, PageResult, TableCell

VALID_FORMATS: tuple[str, ...] = ("json", "md", "html", "txt")
FORMAT_ALIASES: dict[str, str] = {"markdown": "md", "text": "txt"}

# Matches LTR runs (Latin words, URLs, emails, ASCII/version numbers with optional separators)
# while avoiding matching pure punctuation alone.
_LTR_RUN_RE = re.compile(
    r"(?:"
    r"https?://[^\s<>]+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|[A-Za-z][A-Za-z0-9]*(?:[\s._:/-]+[A-Za-z0-9]+)*"
    r"|[0-9]+(?:[./:-][0-9]+)*[A-Za-z]*"
    r")"
)
_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")


def parse_formats(
    formats: str | Sequence[str] | None,
    default: Sequence[str] = ("json",),
) -> list[str]:
    """Parse comma/space-separated or sequence format specs into canonical ordered names."""
    if formats is None:
        raw_tokens = list(default)
    elif isinstance(formats, str):
        raw_tokens = re.split(r"[\s,]+", formats.strip())
    else:
        raw_tokens = []
        for item in formats:
            raw_tokens.extend(re.split(r"[\s,]+", str(item).strip()))

    resolved: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []

    for tok in raw_tokens:
        cleaned = tok.strip().lower()
        if not cleaned:
            continue
        cleaned = FORMAT_ALIASES.get(cleaned, cleaned)
        if cleaned not in VALID_FORMATS:
            invalid.append(tok.strip())
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            resolved.append(cleaned)

    if invalid:
        raise ValueError(
            f"Unsupported export formats: {sorted(set(invalid))}. "
            f"Allowed: {list(VALID_FORMATS)}"
        )
    if not resolved:
        return parse_formats(list(default), default=default)
    return resolved


def _extract_html_table(raw_html: str) -> str:
    """Extract only <table>...</table> fragments if a region model returns a full HTML doc."""
    stripped = raw_html.strip()
    if not stripped:
        return ""
    matches = _TABLE_BLOCK_RE.findall(stripped)
    if matches:
        return "\n".join(matches)
    return stripped


def _strip_html_to_text(raw_html: str) -> str:
    """Convert raw HTML table/markup into readable plain text."""
    text = re.sub(r"</(?:tr|p|div|li|h[1-6])>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:td|th)>", "\t", text, flags=re.IGNORECASE)
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(ln.split()) if "\t" not in ln else "\t".join(c.strip() for c in ln.split("\t")) for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()).strip()


def _markdown_table_to_html(md_text: str) -> str | None:
    """Convert a Markdown pipe table string into a semantic RTL HTML <table>, or None."""
    lines = [ln.strip() for ln in md_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    table_lines = [ln for ln in lines if "|" in ln]
    if len(table_lines) < 2:
        return None

    def _split_row(line: str) -> list[str]:
        inner = line.strip()
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        return [cell.strip() for cell in inner.split("|")]

    sep_idx: int | None = None
    for idx in range(1, len(table_lines)):
        cells = _split_row(table_lines[idx])
        if cells and all(_MD_SEP_CELL_RE.match(c.replace(" ", "")) for c in cells):
            sep_idx = idx
            break
    if sep_idx is None:
        return None

    header_cells = _split_row(table_lines[sep_idx - 1])
    body_rows = [_split_row(ln) for ln in table_lines[sep_idx + 1 :]]
    col_count = max([len(header_cells), *(len(r) for r in body_rows)], default=0)
    if col_count == 0:
        return None

    def _pad(row: list[str]) -> list[str]:
        return row + [""] * (col_count - len(row))

    thead = "    <tr>" + "".join(f"<th>{wrap_bidi_html(c)}</th>" for c in _pad(header_cells)) + "</tr>"
    tbody_rows = [
        "    <tr>" + "".join(f"<td>{wrap_bidi_html(c)}</td>" for c in _pad(r)) + "</tr>"
        for r in body_rows
    ]
    body_block = "\n".join(tbody_rows)
    return (
        '<table dir="rtl">\n'
        f"  <thead>\n{thead}\n  </thead>\n"
        f"  <tbody>\n{body_block}\n  </tbody>\n"
        "</table>"
    )


def to_json(doc: DocumentResult) -> str:
    """Serialize DocumentResult to pretty UTF-8 JSON."""
    return doc.model_dump_json(indent=2)


def _escape_md_cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").splitlines()).strip()


def _cells_to_markdown_table(cells: list[TableCell]) -> str | None:
    """Render simple grid TableCells to GitHub-flavored Markdown; return None if spans exist."""
    if not cells:
        return None
    if any((c.row_end - c.row_start) > 1 or (c.col_end - c.col_start) > 1 for c in cells):
        return None

    max_row = max(c.row_end for c in cells)
    max_col = max(c.col_end for c in cells)
    if max_row <= 0 or max_col <= 0:
        return None

    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    for cell in cells:
        if 0 <= cell.row_start < max_row and 0 <= cell.col_start < max_col:
            grid[cell.row_start][cell.col_start] = _escape_md_cell(cell.text)

    lines = [
        "| " + " | ".join(grid[0]) + " |",
        "| " + " | ".join("---" for _ in range(max_col)) + " |",
    ]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _cells_to_tsv(cells: list[TableCell]) -> str:
    """Render structured TableCells into tab-separated rows for plain text output."""
    if not cells:
        return ""
    rows: dict[int, list[TableCell]] = {}
    for cell in cells:
        rows.setdefault(cell.row_start, []).append(cell)
    lines: list[str] = []
    for r_idx in sorted(rows.keys()):
        row_cells = sorted(rows[r_idx], key=lambda c: c.col_start)
        lines.append("\t".join(" ".join(c.text.splitlines()).strip() for c in row_cells))
    return "\n".join(lines)


def _block_to_markdown(block: LayoutBlock) -> str:
    if block.type == "title":
        text = " ".join(block.text.splitlines()).strip()
        return f"## {text}" if text else ""

    if block.type == "table":
        md_table = _cells_to_markdown_table(block.cells)
        if md_table:
            return md_table
        if block.html:
            return _extract_html_table(block.html)
        return block.text.strip()

    if block.type == "figure":
        caption = block.text.strip()
        return f"[تصویر: {caption}]" if caption else "[تصویر]"

    if block.type == "header":
        return f"<!-- header: {block.text.strip()} -->" if block.text.strip() else ""

    if block.type == "footer":
        return f"<!-- footer: {block.text.strip()} -->" if block.text.strip() else ""

    return block.text.strip()


def _markdown_frontmatter(doc: DocumentResult) -> str:
    if doc.metadata is None:
        return ""
    meta = doc.metadata
    peak_vram = doc.metrics.peak_vram_mb if doc.metrics else None
    peak_rss = doc.metrics.peak_rss_ram_mb if doc.metrics else None
    lines = [
        "---",
        f'doc_id: "{doc.doc_id}"',
        f'source: "{doc.source_filename}"',
        f'model: "{meta.rec_model_label}"',
        f'model_variant: "{meta.rec_model_variant}"',
        f'device: "{meta.device}"',
        f"pages: {doc.page_count}",
        f"processing_time_sec: {doc.processing_time_sec:.3f}",
        f"peak_vram_mb: {peak_vram:.2f}" if peak_vram is not None else "peak_vram_mb: null",
        f"peak_rss_ram_mb: {peak_rss:.2f}" if peak_rss is not None else "peak_rss_ram_mb: null",
        f'created_at: "{meta.created_at}"',
        "---",
    ]
    return "\n".join(lines) + "\n\n"


def to_markdown(doc: DocumentResult) -> str:
    """Serialize DocumentResult to clean Markdown preserving RTL text, tables, and frontmatter."""
    page_sections: list[str] = []
    for page in doc.pages:
        ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        parts = [_block_to_markdown(blk) for blk in ordered_blocks]
        parts = [p for p in parts if p]
        if len(doc.pages) > 1:
            header = f"<!-- Page {page.page_number} -->"
            page_sections.append(f"{header}\n\n" + "\n\n".join(parts) if parts else header)
        else:
            page_sections.append("\n\n".join(parts))

    frontmatter = _markdown_frontmatter(doc)
    body = "\n\n---\n\n".join(page_sections).strip()
    if not body:
        return frontmatter
    return frontmatter + body + "\n"


def _block_to_text(block: LayoutBlock) -> str:
    if block.type == "table":
        if block.cells:
            return _cells_to_tsv(block.cells)
        if block.html:
            return _strip_html_to_text(block.html)
        return block.text.strip()
    if block.type == "figure":
        caption = block.text.strip()
        return f"[تصویر: {caption}]" if caption else ""
    return block.text.strip()


def to_text(doc: DocumentResult) -> str:
    """Serialize DocumentResult to clean UTF-8 plain text in RTL reading order."""
    page_sections: list[str] = []
    for page in doc.pages:
        ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        parts = [_block_to_text(blk) for blk in ordered_blocks]
        parts = [p for p in parts if p]
        if len(doc.pages) > 1:
            header = f"--- Page {page.page_number} ---"
            page_sections.append(f"{header}\n" + "\n\n".join(parts) if parts else header)
        else:
            page_sections.append("\n\n".join(parts))

    body = "\n\n".join(sec for sec in page_sections if sec).strip()
    return body + "\n" if body else ""


def wrap_bidi_html(text: str) -> str:
    """Escape HTML and wrap isolated LTR runs inside <span dir="ltr">."""
    if not text:
        return ""

    pieces: list[str] = []
    last_end = 0
    for match in _LTR_RUN_RE.finditer(text):
        start, end = match.span()
        if start > last_end:
            pieces.append(html.escape(text[last_end:start]))
        token = match.group(0)
        pieces.append(f'<span dir="ltr">{html.escape(token)}</span>')
        last_end = end

    if last_end < len(text):
        pieces.append(html.escape(text[last_end:]))

    return "".join(pieces).replace("\n", "<br>\n")


def _cells_to_html_table(cells: list[TableCell]) -> str:
    """Render TableCells into a semantic RTL HTML <table>."""
    if not cells:
        return '<table dir="rtl"></table>'

    rows: dict[int, list[TableCell]] = {}
    for cell in cells:
        rows.setdefault(cell.row_start, []).append(cell)

    row_html: list[str] = []
    for r_idx in sorted(rows.keys()):
        row_cells = sorted(rows[r_idx], key=lambda c: c.col_start)
        cell_tags: list[str] = []
        for cell in row_cells:
            tag = "th" if cell.is_header or r_idx == 0 else "td"
            attrs: list[str] = []
            rowspan = max(1, cell.row_end - cell.row_start)
            colspan = max(1, cell.col_end - cell.col_start)
            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            cell_tags.append(f"<{tag}{attr_str}>{wrap_bidi_html(cell.text)}</{tag}>")
        row_html.append("    <tr>" + "".join(cell_tags) + "</tr>")

    return '<table dir="rtl">\n  <tbody>\n' + "\n".join(row_html) + "\n  </tbody>\n</table>"


def _block_to_html(block: LayoutBlock) -> str:
    if block.type == "title":
        return f'    <h2 dir="rtl">{wrap_bidi_html(block.text)}</h2>'
    if block.type == "table":
        if block.cells:
            return f'    <div class="table-block" dir="rtl">\n{_cells_to_html_table(block.cells)}\n    </div>'
        if block.html:
            clean_table = _extract_html_table(block.html)
            return f'    <div class="table-block" dir="rtl">\n{clean_table}\n    </div>'
        md_html = _markdown_table_to_html(block.text)
        if md_html is not None:
            return f'    <div class="table-block" dir="rtl">\n{md_html}\n    </div>'
        return f'    <p dir="rtl">{wrap_bidi_html(block.text)}</p>'
    if block.type == "figure":
        caption = wrap_bidi_html(block.text) if block.text else "تصویر"
        return f'    <figure dir="rtl"><figcaption>{caption}</figcaption></figure>'
    if block.type == "header":
        return f'    <header dir="rtl">{wrap_bidi_html(block.text)}</header>'
    if block.type == "footer":
        return f'    <footer dir="rtl">{wrap_bidi_html(block.text)}</footer>'
    return f'    <p dir="rtl">{wrap_bidi_html(block.text)}</p>'


def _render_html_run_banner(doc: DocumentResult) -> str:
    if doc.metadata is None and doc.metrics is None:
        return ""

    chips: list[str] = []
    chips.append(f'<span><strong>فایل:</strong> <span dir="ltr">{html.escape(doc.source_filename)}</span></span>')
    chips.append(f'<span><strong>شناسه:</strong> <span dir="ltr">{html.escape(doc.doc_id)}</span></span>')
    chips.append(f'<span><strong>صفحات:</strong> {doc.page_count}</span>')

    if doc.metadata is not None:
        m = doc.metadata
        chips.append(
            f'<span><strong>مدل تشخیص:</strong> <span dir="ltr">{html.escape(m.rec_model_label)}</span> '
            f'(<span dir="ltr">{html.escape(m.rec_model_variant)}</span>, {html.escape(m.granularity)})</span>'
        )
        stack_parts = [m.layout_model]
        if m.det_model:
            stack_parts.append(m.det_model)
        if m.table_model:
            stack_parts.append(m.table_model)
        chips.append(
            f'<span><strong>پشته:</strong> <span dir="ltr">{html.escape(" + ".join(stack_parts))}</span></span>'
        )
        dev_label = f"{m.device} ({m.gpu_name})" if m.gpu_name else m.device
        chips.append(f'<span><strong>سخت‌افزار:</strong> <span dir="ltr">{html.escape(dev_label)}</span></span>')
        chips.append(f'<span><strong>DPI:</strong> {m.dpi}</span>')
        chips.append(f'<span><strong>زمان اجرا:</strong> <span dir="ltr">{html.escape(m.created_at)}</span></span>')

    if doc.metrics is not None:
        dm = doc.metrics
        chips.append(f'<span><strong>زمان کل:</strong> <span dir="ltr">{dm.total_sec:.2f}s</span></span>')
        chips.append(f'<span><strong>میانگین/صفحه:</strong> <span dir="ltr">{dm.mean_page_latency_sec:.2f}s</span></span>')
        if dm.peak_vram_mb is not None:
            chips.append(f'<span><strong>اوج VRAM:</strong> <span dir="ltr">{dm.peak_vram_mb:.1f} MB</span></span>')
        if dm.peak_rss_ram_mb is not None:
            chips.append(f'<span><strong>اوج RAM:</strong> <span dir="ltr">{dm.peak_rss_ram_mb:.1f} MB</span></span>')
        chips.append(
            f'<span><strong>خروجی:</strong> {dm.total_blocks} بلوک / {dm.total_lines} خط / {dm.total_tables} جدول</span>'
        )
    else:
        chips.append(f'<span><strong>زمان کل:</strong> <span dir="ltr">{doc.processing_time_sec:.2f}s</span></span>')

    items_html = "\n      ".join(chips)
    return (
        '  <header class="run-meta" dir="rtl">\n'
        '    <div class="run-meta-grid">\n'
        f"      {items_html}\n"
        "    </div>\n"
        "  </header>\n"
    )


def _render_page_meta_bar(page: PageResult) -> str:
    if page.metrics is None:
        return ""
    pm = page.metrics
    st = pm.stages
    parts = [
        f"صفحه {page.page_number}",
        f'<span dir="ltr">{pm.latency_sec:.2f}s</span>',
        (
            '<span dir="ltr">'
            f"pre={st.preprocess_sec:.2f}s layout={st.layout_sec:.2f}s "
            f"det={st.detect_sec:.2f}s table={st.table_sec:.2f}s "
            f"rec={st.recognize_sec:.2f}s post={st.postprocess_sec:.2f}s"
            "</span>"
        ),
    ]
    if pm.peak_vram_mb is not None:
        parts.append(f'VRAM <span dir="ltr">{pm.peak_vram_mb:.1f} MB</span>')
    if pm.rss_ram_mb is not None:
        parts.append(f'RAM <span dir="ltr">{pm.rss_ram_mb:.1f} MB</span>')
    parts.append(f"{pm.num_blocks or len(page.blocks)} بلوک")
    return '    <div class="page-meta">' + " · ".join(parts) + "</div>\n"


def to_html(doc: DocumentResult) -> str:
    """Serialize DocumentResult to semantic RTL-aware HTML5 with provenance & BiDi isolation."""
    sections: list[str] = []
    for page in doc.pages:
        ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        page_meta = _render_page_meta_bar(page)
        body = "\n".join(_block_to_html(blk) for blk in ordered_blocks)
        sections.append(
            f'  <section class="page" data-page="{page.page_number}" dir="rtl">\n'
            f"{page_meta}{body}\n"
            "  </section>"
        )

    if doc.metadata is not None:
        title_str = f"{doc.source_filename} — {doc.metadata.rec_model_label}"
        meta_tags = (
            f'\n  <meta name="ocr-doc-id" content="{html.escape(doc.doc_id)}">'
            f'\n  <meta name="ocr-model" content="{html.escape(doc.metadata.rec_model_label)}">'
            f'\n  <meta name="ocr-model-variant" content="{html.escape(doc.metadata.rec_model_variant)}">'
            f'\n  <meta name="ocr-device" content="{html.escape(doc.metadata.device)}">'
            f'\n  <meta name="ocr-created-at" content="{html.escape(doc.metadata.created_at)}">'
        )
    else:
        title_str = doc.source_filename
        meta_tags = ""

    title = html.escape(title_str)
    run_banner = _render_html_run_banner(doc)
    joined_sections = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="fa" dir="rtl">\n'
        "<head>\n"
        '  <meta charset="utf-8">'
        f"{meta_tags}\n"
        f"  <title>{title}</title>\n"
        "  <style>\n"
        "    body { font-family: Vazirmatn, Tahoma, Arial, sans-serif; line-height: 1.8; "
        "max-width: 1020px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }\n"
        "    header.run-meta { background: #f7f8fa; border: 1px solid #dfe3ea; border-radius: 8px; "
        "padding: 0.85rem 1rem; margin-bottom: 1.75rem; font-size: 0.88em; color: #333; }\n"
        "    .run-meta-grid { display: flex; flex-wrap: wrap; gap: 0.45rem 1.25rem; }\n"
        "    .page-meta { font-size: 0.8em; color: #555; background: #fafafa; border-right: 3px solid #c5ccd6; "
        "padding: 0.25rem 0.65rem; margin-bottom: 1rem; }\n"
        "    section.page { margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #ddd; }\n"
        '    [dir="ltr"] { unicode-bidi: isolate; display: inline-block; }\n'
        "    .table-block { overflow-x: auto; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }\n"
        "    th, td { border: 1px solid #bbb; padding: 0.45rem 0.65rem; text-align: right; }\n"
        "    th { background: #f5f5f5; font-weight: 600; }\n"
        "    header, footer { color: #666; font-size: 0.9em; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{run_banner}"
        f"{joined_sections}\n"
        "</body>\n"
        "</html>\n"
    )

"""Single-GPU sequential per-page inference engine orchestrating layout, det, rec, and tables."""

from __future__ import annotations

import gc
import html
import logging
import re
import threading
import time
from typing import Any

import cv2
import numpy as np
import psutil

from config.settings import Settings, configure_cache_env
from src.inference.detector import DetectedLine, TextLineDetector
from src.inference.layout import LayoutAnalyzer, LayoutBox, TableStructureOutput
from src.inference.recognizer import (
    BinaNemotronRecognizer,
    BinaRecognizer,
    OpenAIVLMRecognizer,
    TextRecognizerProtocol,
    create_recognizer,
    create_vlm_recognizer,
)
from src.schemas.document import (
    BBox,
    LayoutBlock,
    PageMetrics,
    PageResult,
    StageTimings,
    TableCell,
    TextLine,
)

logger = logging.getLogger(__name__)

_PERSIAN_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_HYBRID_COMPLEX_RAW_LABELS = frozenset({"formula", "algorithm", "seal", "chart"})
_VLM_MARKUP_TAG_RE = re.compile(r"</?(?:chart|table|thead|tbody|t[rdh])[^>]*>", re.IGNORECASE)
_VLM_MBOX_RE = re.compile(r"\\mbox\s*\{[^}]*\}")
_VLM_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^}]*)\}")
_VLM_MD_SEP_RE = re.compile(r"(?:\|\s*)?:?-{3,}:?(?:\s*\|)?")
_VLM_UNDERSCORE_RE = re.compile(r"_{4,}")
_VLM_CSS_BOILERPLATE_RE = re.compile(
    r"<!doctype|<html\b|<style\b|box-sizing\s*:|\.page\s*\{",
    re.IGNORECASE,
)
_MD_PIPE_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")
_ATTR_INT_RE = {
    "colspan": re.compile(r'colspan\s*=\s*["\']?(\d+)', re.IGNORECASE),
    "rowspan": re.compile(r'rowspan\s*=\s*["\']?(\d+)', re.IGNORECASE),
}


def _box_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _overlap_over_line(line_box: BBox, container_box: BBox) -> float:
    line_area = _box_area(line_box)
    if line_area <= 1e-6:
        return 0.0
    return _intersection_area(line_box, container_box) / line_area


def _parse_table_cells(
    table_struct: TableStructureOutput,
    offset_x: float,
    offset_y: float,
) -> list[TableCell]:
    """Parse PaddleOCR table HTML tokens and cell bounding boxes into grid TableCells."""
    tokens = table_struct.structure_tokens
    bboxes = table_struct.cell_bboxes
    if not tokens or not bboxes:
        return []

    cells: list[TableCell] = []
    occupied: set[tuple[int, int]] = set()
    row_idx = -1
    col_idx = 0
    in_thead = False
    bbox_cursor = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        low = tok.strip().lower()

        if low == "<thead>":
            in_thead = True
        elif low == "</thead>":
            in_thead = False
        elif low == "<tr>":
            row_idx += 1
            col_idx = 0
        elif low in {"<td>", "<td></td>", "<td"} or low.startswith("<td"):
            if row_idx < 0:
                row_idx = 0

            tag_buf = tok
            if low == "<td":
                j = i + 1
                while j < len(tokens):
                    tag_buf += tokens[j]
                    if ">" in tokens[j]:
                        i = j
                        break
                    j += 1

            colspan = 1
            rowspan = 1
            m_col = _ATTR_INT_RE["colspan"].search(tag_buf)
            m_row = _ATTR_INT_RE["rowspan"].search(tag_buf)
            if m_col:
                colspan = max(1, int(m_col.group(1)))
            if m_row:
                rowspan = max(1, int(m_row.group(1)))

            while (row_idx, col_idx) in occupied:
                col_idx += 1

            if bbox_cursor < len(bboxes):
                bx1, by1, bx2, by2 = bboxes[bbox_cursor]
                page_bbox: BBox = (
                    bx1 + offset_x,
                    by1 + offset_y,
                    bx2 + offset_x,
                    by2 + offset_y,
                )
            else:
                page_bbox = (offset_x, offset_y, offset_x, offset_y)
            bbox_cursor += 1

            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied.add((r, c))

            cells.append(
                TableCell(
                    row_start=row_idx,
                    row_end=row_idx + rowspan,
                    col_start=col_idx,
                    col_end=col_idx + colspan,
                    bbox=page_bbox,
                    text="",
                    is_header=in_thead or (row_idx == 0),
                    confidence=0.0,
                )
            )
            col_idx += colspan
        i += 1

    return cells


def _build_rtl_table_html(cells: list[TableCell]) -> str:
    """Build clean RTL HTML <table> from mapped TableCell objects."""
    if not cells:
        return '<table dir="rtl"></table>'
    rows: dict[int, list[TableCell]] = {}
    for cell in cells:
        rows.setdefault(cell.row_start, []).append(cell)

    row_parts: list[str] = []
    for r_idx in sorted(rows.keys()):
        tds: list[str] = []
        for cell in sorted(rows[r_idx], key=lambda c: c.col_start):
            tag = "th" if cell.is_header else "td"
            attrs: list[str] = []
            rs = max(1, cell.row_end - cell.row_start)
            cs = max(1, cell.col_end - cell.col_start)
            if rs > 1:
                attrs.append(f'rowspan="{rs}"')
            if cs > 1:
                attrs.append(f'colspan="{cs}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            tds.append(f"<{tag}{attr_str}>{html.escape(cell.text)}</{tag}>")
        row_parts.append("<tr>" + "".join(tds) + "</tr>")

    return '<table dir="rtl"><tbody>' + "".join(row_parts) + "</tbody></table>"


def _cell_has_ink(cell_bgr: np.ndarray) -> bool:
    """Return True if a table cell interior contains text ink (ignoring outer gridlines)."""
    if cell_bgr is None or cell_bgr.size == 0:
        return False
    h, w = cell_bgr.shape[:2]
    if h < 6 or w < 6:
        return False
    pad_y = min(4, max(2, h // 8))
    pad_x = min(4, max(2, w // 8))
    if h - 2 * pad_y < 2 or w - 2 * pad_x < 2:
        interior = cell_bgr
    else:
        interior = cell_bgr[pad_y : h - pad_y, pad_x : w - pad_x]

    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray))
    diff = np.abs(gray.astype(np.float32) - med)
    return bool(int((diff > 35.0).sum()) >= 6 and float(diff.max()) >= 50.0)


def _flip_rtl_table_columns(table_cells: list[TableCell]) -> list[TableCell]:
    """Reverse table column indices when Persian/Arabic text is present (visual LTR -> logical RTL)."""
    if not any(_PERSIAN_ARABIC_RE.search(c.text) for c in table_cells):
        return table_cells
    total_cols = max((c.col_end for c in table_cells), default=0)
    if total_cols <= 1:
        return table_cells
    return [
        c.model_copy(
            update={
                "col_start": total_cols - c.col_end,
                "col_end": total_cols - c.col_start,
            }
        )
        for c in table_cells
    ]


def _prepare_vlm_cell_crop(bgr_img: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Trim gridline borders from a table cell, verify ink presence, and add clean padding."""
    if bgr_img is None or bgr_img.size == 0:
        return None
    img_h, img_w = bgr_img.shape[:2]
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(img_w, int(np.ceil(bbox[2])))
    y2 = min(img_h, int(np.ceil(bbox[3])))

    trim_x = 2 if (x2 - x1) >= 12 else 0
    trim_y = 2 if (y2 - y1) >= 12 else 0
    ix1, ix2 = x1 + trim_x, x2 - trim_x
    iy1, iy2 = y1 + trim_y, y2 - trim_y
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    interior = bgr_img[iy1:iy2, ix1:ix2]
    if not _cell_has_ink(interior):
        return None

    border_px = np.vstack([interior[:1, :], interior[-1:, :]]).reshape(-1, 3)
    bg_color = tuple(int(v) for v in np.median(border_px, axis=0))
    return cv2.copyMakeBorder(
        interior,
        top=6,
        bottom=6,
        left=6,
        right=6,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,
    )


def _clean_vlm_cell_text(raw_text: str) -> str:
    """Strip stray table markup/separator/latex artifacts from VLM table-cell predictions."""
    if not raw_text:
        return ""
    text = _VLM_MARKUP_TAG_RE.sub(" ", raw_text)
    text = _VLM_MBOX_RE.sub(" ", text)
    text = _VLM_LATEX_TEXT_RE.sub(r"\1", text)
    text = _VLM_MD_SEP_RE.sub(" ", text)
    text = _VLM_UNDERSCORE_RE.sub(" ", text)
    text = text.replace("|", " ")
    return " ".join(text.split()).strip()


def _is_vlm_boilerplate(text: str) -> bool:
    """Return True if VLM output collapsed into synthetic HTML/CSS template boilerplate."""
    return bool(text and _VLM_CSS_BOILERPLATE_RE.search(text))


def _split_md_pipe_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _parse_markdown_table_cells(md_text: str, table_bbox: BBox) -> list[TableCell]:
    """Parse a clean Markdown pipe table into logical-RTL TableCell rows (>=2 rows, >=2 cols)."""
    table_lines = [ln.strip() for ln in md_text.splitlines() if "|" in ln]
    if len(table_lines) < 2:
        return []

    sep_idx: int | None = None
    for idx in range(1, len(table_lines)):
        row_cells = _split_md_pipe_row(table_lines[idx])
        if row_cells and all(_MD_PIPE_SEP_CELL_RE.match(c.replace(" ", "")) for c in row_cells):
            sep_idx = idx
            break
    if sep_idx is None:
        return []

    header_row = _split_md_pipe_row(table_lines[sep_idx - 1])
    body_rows: list[list[str]] = []
    for ln in table_lines[sep_idx + 1 :]:
        parsed = _split_md_pipe_row(ln)
        if parsed and all(_MD_PIPE_SEP_CELL_RE.match(c.replace(" ", "")) for c in parsed):
            continue
        if any(c.strip() for c in parsed):
            body_rows.append(parsed)

    all_rows = [header_row, *body_rows]
    num_cols = max((len(r) for r in all_rows), default=0)
    if len(all_rows) < 2 or num_cols < 2:
        return []

    cells: list[TableCell] = []
    for r_idx, row in enumerate(all_rows):
        padded = row + [""] * (num_cols - len(row))
        for c_idx in range(num_cols):
            val = _clean_vlm_cell_text(padded[c_idx])
            cells.append(
                TableCell(
                    row_start=r_idx,
                    row_end=r_idx + 1,
                    col_start=c_idx,
                    col_end=c_idx + 1,
                    bbox=table_bbox,
                    text=val,
                    is_header=(r_idx == 0),
                    confidence=1.0 if val else 0.0,
                )
            )
    return cells


def _is_valid_vlm_table_markdown(
    raw_text: str,
    conf: float,
    table_bbox: BBox,
) -> list[TableCell]:
    """Validate VLM whole-table output (non-capped, non-CSS) and return TableCells if valid."""
    if not raw_text or not raw_text.strip() or conf < 0.65 or _is_vlm_boilerplate(raw_text):
        return []
    return _parse_markdown_table_cells(raw_text, table_bbox)


def _has_multicolumn_overlap(lines: list[TextLine]) -> bool:
    """Detect side-by-side multi-column lines sharing vertical Y bands while disjoint in X."""
    if len(lines) < 4:
        return False
    paired = 0
    for i, a in enumerate(lines):
        ax1, ay1, ax2, ay2 = a.bbox
        ah = max(1.0, ay2 - ay1)
        for j, b in enumerate(lines):
            if i == j:
                continue
            bx1, by1, bx2, by2 = b.bbox
            bh = max(1.0, by2 - by1)
            ov_y = max(0.0, min(ay2, by2) - max(ay1, by1))
            if ov_y / min(ah, bh) < 0.55:
                continue
            # Horizontally disjoint (no X intersection)
            if min(ax2, bx2) <= max(ax1, bx1):
                paired += 1
                break
    return (paired / len(lines)) >= 0.30


class InferenceEngine:
    """Sequential, VRAM-bounded per-page OCR engine (Paddle Layout + Line/Region/Hybrid Recognizer)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        self._gpu_lock = threading.Lock()
        self.layout_analyzer = LayoutAnalyzer(settings)
        self.is_hybrid = settings.rec_model_variant == "hybrid"
        self.recognizer: TextRecognizerProtocol = create_recognizer(settings)
        self._vlm_recognizer: TextRecognizerProtocol | None = None
        self.text_detector: TextLineDetector | None = (
            TextLineDetector(settings)
            if (self.is_hybrid or self.recognizer.granularity == "line")
            else None
        )

    def _get_vlm_recognizer(self) -> TextRecognizerProtocol:
        """Lazily load Bina-0.4-Koochik VLM only when a hybrid page routes a block to it."""
        if self._vlm_recognizer is None:
            self._vlm_recognizer = create_vlm_recognizer(self.settings)
        return self._vlm_recognizer

    def _cuda_is_active(self) -> bool:
        if self.settings.device != "gpu":
            return False
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                return True
        except Exception:
            pass
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _reset_vram_peaks() -> None:
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                paddle.device.cuda.reset_max_memory_allocated(0)
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass

    @staticmethod
    def _read_vram_mb() -> tuple[float, float]:
        """Return combined (allocated_mb, peak_allocated_mb) across Paddle and PyTorch."""
        alloc_bytes = 0
        peak_bytes = 0
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                alloc_bytes += int(paddle.device.cuda.memory_allocated(0))
                peak_bytes += int(paddle.device.cuda.max_memory_allocated(0))
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                alloc_bytes += int(torch.cuda.memory_allocated(0))
                peak_bytes += int(torch.cuda.max_memory_allocated(0))
        except Exception:
            pass
        return round(alloc_bytes / (1024 * 1024), 2), round(peak_bytes / (1024 * 1024), 2)

    @staticmethod
    def _empty_cuda_caches() -> None:
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                paddle.device.cuda.empty_cache()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def get_gpu_name(self) -> str | None:
        """Return active CUDA device name from Paddle or PyTorch, or None."""
        if not self._cuda_is_active():
            return None
        try:
            import paddle

            if paddle.device.cuda.device_count() > 0:
                return str(getattr(paddle.device.cuda.get_device_properties(0), "name", "GPU"))
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        return None

    def unload(self) -> None:
        """Release model references and clear CUDA caches for safe model switching."""
        with self._gpu_lock:
            self.layout_analyzer = None  # type: ignore[assignment]
            self.text_detector = None
            self.recognizer = None  # type: ignore[assignment]
            self._vlm_recognizer = None
            BinaRecognizer.clear_cache()
            BinaNemotronRecognizer.clear_cache()
            OpenAIVLMRecognizer.clear_cache()
            gc.collect()
            self._empty_cuda_caches()

    def health_info(self) -> dict[str, Any]:
        """Return GPU & loaded model telemetry for /health."""
        cuda_ok = self._cuda_is_active()
        gpu_name = self.get_gpu_name() if cuda_ok else None
        alloc_mb = 0.0
        peak_mb = 0.0
        if cuda_ok:
            alloc_mb, peak_mb = self._read_vram_mb()

        models: list[str] = []
        if self.layout_analyzer is not None:
            models.extend(self.layout_analyzer.loaded_model_names)
        if self.text_detector is not None:
            models.append(self.settings.det_model_name)
        if self.recognizer is not None:
            models.append(self.recognizer.model_label)
        if getattr(self, "_vlm_recognizer", None) is not None:
            models.append(self._vlm_recognizer.model_label)  # type: ignore[union-attr]
        models = list(dict.fromkeys(models))

        return {
            "device": self.settings.device,
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "loaded_models": models,
            "vram_allocated_mb": alloc_mb,
            "vram_peak_mb": peak_mb,
        }

    def _process_line_mode(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        layout_boxes: list[LayoutBox],
    ) -> tuple[list[LayoutBlock], dict[str, float]]:
        """Run TextDetection -> Bina-0.2 line recognition -> SLANet_plus table cell mapping."""
        assert self.text_detector is not None
        img_h, img_w = bgr_img.shape[:2]

        t_det0 = time.perf_counter()
        page_det_lines = self.text_detector.detect(bgr_img)
        detect_sec = time.perf_counter() - t_det0

        block_lines_map: dict[int, list[DetectedLine]] = {i: [] for i in range(len(layout_boxes))}
        unassigned_lines: list[DetectedLine] = []

        for det_line in page_det_lines:
            best_idx: int | None = None
            best_overlap = 0.35
            for b_idx, lbox in enumerate(layout_boxes):
                if lbox.type == "figure":
                    continue
                ov = _overlap_over_line(det_line.bbox, lbox.bbox)
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = b_idx
            if best_idx is not None:
                block_lines_map[best_idx].append(det_line)
            else:
                unassigned_lines.append(det_line)

        if unassigned_lines:
            ux1 = min(ln.bbox[0] for ln in unassigned_lines)
            uy1 = min(ln.bbox[1] for ln in unassigned_lines)
            ux2 = max(ln.bbox[2] for ln in unassigned_lines)
            uy2 = max(ln.bbox[3] for ln in unassigned_lines)
            fallback_idx = len(layout_boxes)
            layout_boxes.append(
                LayoutBox(
                    type="text",
                    raw_label="text",
                    bbox=(ux1, uy1, ux2, uy2),
                    confidence=0.5,
                )
            )
            block_lines_map[fallback_idx] = unassigned_lines

        table_sec = 0.0
        table_structs: dict[int, tuple[TableStructureOutput, float, float]] = {}
        if self.settings.enable_tables:
            for b_idx, lbox in enumerate(layout_boxes):
                if lbox.type != "table":
                    continue
                tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
                ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
                tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
                ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
                table_crop = bgr_img[ty1:ty2, tx1:tx2]
                if table_crop.size == 0:
                    continue

                t_tbl0 = time.perf_counter()
                t_struct = self.layout_analyzer.analyze_table(table_crop)
                table_sec += time.perf_counter() - t_tbl0
                table_structs[b_idx] = (t_struct, float(tx1), float(ty1))

                if len(block_lines_map[b_idx]) < 2:
                    t_sub_det0 = time.perf_counter()
                    crop_lines = self.text_detector.detect(table_crop)
                    detect_sec += time.perf_counter() - t_sub_det0
                    if len(crop_lines) > len(block_lines_map[b_idx]):
                        offset_lines: list[DetectedLine] = []
                        for cl in crop_lines:
                            cx1, cy1, cx2, cy2 = cl.bbox
                            offset_lines.append(
                                DetectedLine(
                                    bbox=(cx1 + tx1, cy1 + ty1, cx2 + tx1, cy2 + ty1),
                                    polygon=[[p[0] + tx1, p[1] + ty1] for p in cl.polygon],
                                    score=cl.score,
                                )
                            )
                        block_lines_map[b_idx] = offset_lines

    def _extract_table_structure(
        self,
        bgr_img: np.ndarray,
        lbox: LayoutBox,
    ) -> tuple[TableStructureOutput, float, float, int]:
        """Run SLANet_plus on a cropped table block and return (struct, off_x, off_y, crop_h)."""
        img_h, img_w = bgr_img.shape[:2]
        tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
        ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
        tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
        ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
        table_crop = bgr_img[ty1:ty2, tx1:tx2]
        if table_crop.size == 0:
            return TableStructureOutput(structure_tokens=[], cell_bboxes=[]), float(tx1), float(ty1), 0
        t_struct = self.layout_analyzer.analyze_table(table_crop)
        return t_struct, float(tx1), float(ty1), int(ty2 - ty1)

    def _run_ctc_blocks(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        work_boxes: list[LayoutBox],
        block_lines_map: dict[int, list[DetectedLine]],
        ctc_indices: list[int],
        table_structs: dict[int, tuple[TableStructureOutput, float, float]],
        routing_reasons: dict[int, str] | None = None,
    ) -> tuple[dict[int, LayoutBlock], float, float]:
        """Batch-recognize detected lines plus recovered empty table-cell/footer ink crops on CTC."""
        assert self.text_detector is not None
        detect_sec = 0.0
        raw_cells_map: dict[int, list[TableCell]] = {}
        line_to_cell_map: dict[tuple[int, int], int] = {}
        cell_has_line: dict[int, set[int]] = {}

        # Sub-detect sparse CTC tables and pre-map detected lines onto SLANet_plus cells
        for b_idx in ctc_indices:
            if b_idx not in table_structs:
                continue
            t_struct, off_x, off_y = table_structs[b_idx]
            if len(block_lines_map.get(b_idx, [])) < 2:
                img_h, img_w = bgr_img.shape[:2]
                lbox = work_boxes[b_idx]
                tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
                ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
                tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
                ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
                table_crop = bgr_img[ty1:ty2, tx1:tx2]
                if table_crop.size > 0:
                    t_sub_det0 = time.perf_counter()
                    crop_lines = self.text_detector.detect(table_crop)
                    detect_sec += time.perf_counter() - t_sub_det0
                    if len(crop_lines) > len(block_lines_map.get(b_idx, [])):
                        block_lines_map[b_idx] = [
                            DetectedLine(
                                bbox=(cl.bbox[0] + tx1, cl.bbox[1] + ty1, cl.bbox[2] + tx1, cl.bbox[3] + ty1),
                                polygon=[[p[0] + tx1, p[1] + ty1] for p in cl.polygon],
                                score=cl.score,
                            )
                            for cl in crop_lines
                        ]

            raw_cells = _parse_table_cells(t_struct, offset_x=off_x, offset_y=off_y)
            raw_cells_map[b_idx] = raw_cells
            covered: set[int] = set()
            for l_idx, det_line in enumerate(block_lines_map.get(b_idx, [])):
                best_cell_idx: int | None = None
                best_metric = (0.30, 0.0)
                for c_idx, cell in enumerate(raw_cells):
                    ov = _overlap_over_line(det_line.bbox, cell.bbox)
                    if ov >= 0.30:
                        metric = (ov, _iou(det_line.bbox, cell.bbox))
                        if metric > best_metric:
                            best_metric = metric
                            best_cell_idx = c_idx
                if best_cell_idx is not None:
                    line_to_cell_map[(b_idx, l_idx)] = best_cell_idx
                    covered.add(best_cell_idx)
            cell_has_line[b_idx] = covered

        flat_crops: list[np.ndarray] = []
        # Tuple: (kind, b_idx, idx_or_c_idx, obj)
        crop_refs: list[tuple[str, int, int, Any]] = []

        for b_idx in ctc_indices:
            for l_idx, det_line in enumerate(block_lines_map.get(b_idx, [])):
                crop = self.text_detector.extract_crop(bgr_img, det_line.polygon, det_line.bbox)
                flat_crops.append(crop)
                crop_refs.append(("det", b_idx, l_idx, det_line))

        if getattr(self.settings, "enable_cell_ink_recovery", True):
            for b_idx in ctc_indices:
                if b_idx in raw_cells_map:
                    covered = cell_has_line.get(b_idx, set())
                    for c_idx, cell in enumerate(raw_cells_map[b_idx]):
                        if c_idx in covered:
                            continue
                        cell_crop = _prepare_vlm_cell_crop(bgr_img, cell.bbox)
                        if cell_crop is not None:
                            flat_crops.append(cell_crop)
                            crop_refs.append(("cell_ink", b_idx, c_idx, cell.bbox))
                elif work_boxes[b_idx].type in {"header", "footer"} and not block_lines_map.get(b_idx):
                    blk_crop = _prepare_vlm_cell_crop(bgr_img, work_boxes[b_idx].bbox)
                    if blk_crop is not None:
                        flat_crops.append(blk_crop)
                        crop_refs.append(("block_ink", b_idx, -1, work_boxes[b_idx].bbox))

        recognize_sec = 0.0
        recognized: list[tuple[str, float]] = []
        if flat_crops:
            t_rec0 = time.perf_counter()
            recognized = self.recognizer.recognize_batch(
                flat_crops, batch_size=self.settings.rec_batch_size
            )
            recognize_sec = time.perf_counter() - t_rec0
        del flat_crops

        recognized_by_block: dict[int, list[TextLine]] = {b_idx: [] for b_idx in ctc_indices}
        cell_lines_map: dict[int, list[list[TextLine]]] = {
            b_idx: [[] for _ in cells] for b_idx, cells in raw_cells_map.items()
        }

        for (kind, b_idx, item_idx, payload), (rec_text, rec_score) in zip(crop_refs, recognized):
            if kind == "det":
                det_line = payload
                tline = TextLine(
                    text=rec_text,
                    bbox=det_line.bbox,
                    polygon=det_line.polygon,
                    confidence=round(rec_score, 4),
                )
                recognized_by_block[b_idx].append(tline)
                mapped_cell = line_to_cell_map.get((b_idx, item_idx))
                if mapped_cell is not None:
                    cell_lines_map[b_idx][mapped_cell].append(tline)
            elif kind == "cell_ink":
                cleaned = rec_text.strip()
                if not cleaned:
                    continue
                tline = TextLine(
                    text=cleaned,
                    bbox=payload,
                    polygon=[],
                    confidence=round(rec_score, 4),
                )
                recognized_by_block[b_idx].append(tline)
                cell_lines_map[b_idx][item_idx].append(tline)
            elif kind == "block_ink":
                cleaned = rec_text.strip()
                if not cleaned:
                    continue
                recognized_by_block[b_idx].append(
                    TextLine(
                        text=cleaned,
                        bbox=payload,
                        polygon=[],
                        confidence=round(rec_score, 4),
                    )
                )

        ctc_blocks_out: dict[int, LayoutBlock] = {}
        for b_idx in ctc_indices:
            lbox = work_boxes[b_idx]
            text_lines = recognized_by_block.get(b_idx, [])
            block_id = f"p{page_number}_b{b_idx}"
            table_cells: list[TableCell] = []
            table_html: str | None = None

            if lbox.type == "table" and b_idx in raw_cells_map:
                raw_cells = raw_cells_map[b_idx]
                if raw_cells:
                    cell_lines = cell_lines_map[b_idx]
                    for c_idx, cell in enumerate(raw_cells):
                        assigned = cell_lines[c_idx]
                        assigned.sort(
                            key=lambda ln: (ln.bbox[1], -((ln.bbox[0] + ln.bbox[2]) / 2.0))
                        )
                        cell_text = " ".join(ln.text.strip() for ln in assigned if ln.text.strip())
                        cell_conf = (
                            sum(ln.confidence for ln in assigned) / len(assigned)
                            if assigned
                            else 0.0
                        )
                        table_cells.append(
                            cell.model_copy(
                                update={
                                    "text": cell_text,
                                    "confidence": round(cell_conf, 4),
                                }
                            )
                        )
                    table_cells = _flip_rtl_table_columns(table_cells)
                    table_html = _build_rtl_table_html(table_cells)

            joined_text = "\n".join(ln.text for ln in text_lines if ln.text)
            reason = routing_reasons.get(b_idx) if routing_reasons else None
            ctc_blocks_out[b_idx] = LayoutBlock(
                block_id=block_id,
                type=lbox.type,
                bbox=lbox.bbox,
                reading_order=b_idx,
                text=joined_text,
                html=table_html,
                cells=table_cells,
                confidence=round(lbox.confidence, 4),
                lines=text_lines,
                recognizer=self.recognizer.model_label,
                routing_reason=reason,
            )

        return ctc_blocks_out, detect_sec, recognize_sec

    def _process_line_mode(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        layout_boxes: list[LayoutBox],
    ) -> tuple[list[LayoutBlock], dict[str, Any]]:
        """Run TextDetection -> Bina-0.2 batched CTC + SLANet_plus cell/footer ink recovery."""
        assert self.text_detector is not None
        img_h, img_w = bgr_img.shape[:2]

        t_det0 = time.perf_counter()
        page_det_lines = self.text_detector.detect(bgr_img)
        detect_sec = time.perf_counter() - t_det0

        block_lines_map: dict[int, list[DetectedLine]] = {i: [] for i in range(len(layout_boxes))}
        unassigned_lines: list[DetectedLine] = []

        for det_line in page_det_lines:
            best_idx: int | None = None
            best_overlap = 0.35
            for b_idx, lbox in enumerate(layout_boxes):
                if lbox.type == "figure":
                    continue
                ov = _overlap_over_line(det_line.bbox, lbox.bbox)
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = b_idx
            if best_idx is not None:
                block_lines_map[best_idx].append(det_line)
            else:
                unassigned_lines.append(det_line)

        if unassigned_lines:
            ux1 = min(ln.bbox[0] for ln in unassigned_lines)
            uy1 = min(ln.bbox[1] for ln in unassigned_lines)
            ux2 = max(ln.bbox[2] for ln in unassigned_lines)
            uy2 = max(ln.bbox[3] for ln in unassigned_lines)
            fallback_idx = len(layout_boxes)
            layout_boxes.append(
                LayoutBox(
                    type="text",
                    raw_label="text",
                    bbox=(ux1, uy1, ux2, uy2),
                    confidence=0.5,
                )
            )
            block_lines_map[fallback_idx] = unassigned_lines

        table_sec = 0.0
        table_structs: dict[int, tuple[TableStructureOutput, float, float]] = {}
        if self.settings.enable_tables:
            for b_idx, lbox in enumerate(layout_boxes):
                if lbox.type != "table":
                    continue
                t_tbl0 = time.perf_counter()
                t_struct, tx1, ty1, _ = self._extract_table_structure(bgr_img, lbox)
                table_sec += time.perf_counter() - t_tbl0
                table_structs[b_idx] = (t_struct, tx1, ty1)

        ctc_indices = [i for i, lb in enumerate(layout_boxes) if lb.type != "figure"]
        ctc_blocks_out, sub_det_sec, recognize_sec = self._run_ctc_blocks(
            page_number=page_number,
            bgr_img=bgr_img,
            work_boxes=layout_boxes,
            block_lines_map=block_lines_map,
            ctc_indices=ctc_indices,
            table_structs=table_structs,
        )
        detect_sec += sub_det_sec

        blocks: list[LayoutBlock] = []
        for b_idx, lbox in enumerate(layout_boxes):
            if lbox.type == "figure":
                blocks.append(
                    LayoutBlock(
                        block_id=f"p{page_number}_b{b_idx}",
                        type=lbox.type,
                        bbox=lbox.bbox,
                        reading_order=b_idx,
                        text="",
                        confidence=round(lbox.confidence, 4),
                        lines=[],
                        recognizer=None,
                    )
                )
            else:
                blocks.append(ctc_blocks_out[b_idx])

        ctc_blocks = sum(1 for b in blocks if b.type != "figure")
        return blocks, {
            "detect_sec": round(detect_sec, 3),
            "table_sec": round(table_sec, 3),
            "recognize_sec": round(recognize_sec, 3),
            "ctc_sec": round(recognize_sec, 3),
            "vlm_sec": 0.0,
            "ctc_blocks": ctc_blocks,
            "vlm_blocks": 0,
        }

    @staticmethod
    def _region_text_to_lines(
        region_text: str,
        bbox: BBox,
        confidence: float,
    ) -> list[TextLine]:
        raw_lines = [ln.strip() for ln in region_text.splitlines() if ln.strip()]
        text_lines: list[TextLine] = []
        if raw_lines:
            step_y = max(1.0, (bbox[3] - bbox[1]) / len(raw_lines))
            for l_idx, line_str in enumerate(raw_lines):
                ly1 = bbox[1] + l_idx * step_y
                ly2 = min(bbox[3], ly1 + step_y)
                text_lines.append(
                    TextLine(
                        text=line_str,
                        bbox=(bbox[0], ly1, bbox[2], ly2),
                        confidence=round(confidence, 4),
                    )
                )
        return text_lines

    def _recognize_table_vlm(
        self,
        bgr_img: np.ndarray,
        lbox: LayoutBox,
        block_id: str,
        reading_order: int,
        vlm_recognizer: TextRecognizerProtocol,
    ) -> tuple[LayoutBlock, float, float]:
        """Run SLANet_plus cell grid + clean borderless VLM cell recognition on one table block."""
        img_h, img_w = bgr_img.shape[:2]
        tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
        ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
        tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
        ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
        table_crop = bgr_img[ty1:ty2, tx1:tx2]

        table_sec = 0.0
        recognize_sec = 0.0
        raw_cells: list[TableCell] = []
        if table_crop.size > 0:
            t_tbl0 = time.perf_counter()
            t_struct = self.layout_analyzer.analyze_table(table_crop)
            table_sec = time.perf_counter() - t_tbl0
            raw_cells = _parse_table_cells(
                t_struct,
                offset_x=float(tx1),
                offset_y=float(ty1),
            )
        del table_crop

        if raw_cells:
            table_cells: list[TableCell] = []
            text_lines: list[TextLine] = []
            for cell in raw_cells:
                cell_crop = _prepare_vlm_cell_crop(bgr_img, cell.bbox)
                if cell_crop is not None:
                    t_rec0 = time.perf_counter()
                    raw_cell_text, cell_conf = vlm_recognizer.recognize_region(
                        cell_crop,
                        block_type="table_cell",
                        max_new_tokens=128,
                    )
                    recognize_sec += time.perf_counter() - t_rec0
                    cell_text = _clean_vlm_cell_text(raw_cell_text)
                else:
                    cell_text = ""
                    cell_conf = 0.0
                del cell_crop

                table_cells.append(
                    cell.model_copy(
                        update={
                            "text": cell_text,
                            "confidence": round(cell_conf, 4),
                        }
                    )
                )
                if cell_text:
                    text_lines.append(
                        TextLine(
                            text=cell_text,
                            bbox=cell.bbox,
                            confidence=round(cell_conf, 4),
                        )
                    )

            table_cells = _flip_rtl_table_columns(table_cells)
            table_html = _build_rtl_table_html(table_cells)
            block = LayoutBlock(
                block_id=block_id,
                type=lbox.type,
                bbox=lbox.bbox,
                reading_order=reading_order,
                text="\n".join(ln.text for ln in text_lines if ln.text),
                html=table_html,
                cells=table_cells,
                confidence=round(lbox.confidence, 4),
                lines=text_lines,
                recognizer=vlm_recognizer.model_label,
                routing_reason="table",
            )
            return block, table_sec, recognize_sec

        # Fallback to plain OCR if table structure failed so Nemotron does not emit CSS/HTML boilerplate
        bx1 = max(0, int(np.floor(lbox.bbox[0])) - 6)
        by1 = max(0, int(np.floor(lbox.bbox[1])) - 6)
        bx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 6)
        by2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 6)
        region_crop = bgr_img[by1:by2, bx1:bx2]
        t_rec0 = time.perf_counter()
        region_text, rec_conf = vlm_recognizer.recognize_region(
            region_crop,
            block_type="text",
        )
        recognize_sec += time.perf_counter() - t_rec0
        del region_crop

        text_lines = self._region_text_to_lines(region_text, lbox.bbox, rec_conf)
        block = LayoutBlock(
            block_id=block_id,
            type=lbox.type,
            bbox=lbox.bbox,
            reading_order=reading_order,
            text=region_text,
            html=None,
            cells=[],
            confidence=round(lbox.confidence, 4),
            lines=text_lines,
            recognizer=vlm_recognizer.model_label,
            routing_reason="table_fallback",
        )
        return block, table_sec, recognize_sec

    def _process_region_mode(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        layout_boxes: list[LayoutBox],
    ) -> tuple[list[LayoutBlock], dict[str, Any]]:
        """Run region-level recognizer (Bina-0.4-Koochik) with SLANet_plus table cell grids."""
        img_h, img_w = bgr_img.shape[:2]
        blocks: list[LayoutBlock] = []
        table_sec = 0.0
        recognize_sec = 0.0

        for b_idx, lbox in enumerate(layout_boxes):
            block_id = f"p{page_number}_b{b_idx}"
            if lbox.type == "figure":
                blocks.append(
                    LayoutBlock(
                        block_id=block_id,
                        type=lbox.type,
                        bbox=lbox.bbox,
                        reading_order=b_idx,
                        text="",
                        confidence=round(lbox.confidence, 4),
                        lines=[],
                    )
                )
                continue

            if lbox.type == "table" and self.settings.enable_tables:
                tbl_block, t_sec, r_sec = self._recognize_table_vlm(
                    bgr_img=bgr_img,
                    lbox=lbox,
                    block_id=block_id,
                    reading_order=b_idx,
                    vlm_recognizer=self.recognizer,
                )
                table_sec += t_sec
                recognize_sec += r_sec
                blocks.append(tbl_block)
                continue

            bx1 = max(0, int(np.floor(lbox.bbox[0])) - 6)
            by1 = max(0, int(np.floor(lbox.bbox[1])) - 6)
            bx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 6)
            by2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 6)
            region_crop = bgr_img[by1:by2, bx1:bx2]

            # Fall back to plain OCR if table structure failed so Nemotron does not emit CSS/HTML boilerplate
            prompt_block_type = "text" if lbox.type == "table" else lbox.type
            t_rec0 = time.perf_counter()
            region_text, rec_conf = self.recognizer.recognize_region(
                region_crop,
                block_type=prompt_block_type,
            )
            recognize_sec += time.perf_counter() - t_rec0
            del region_crop

            text_lines = self._region_text_to_lines(region_text, lbox.bbox, rec_conf)
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type=lbox.type,
                    bbox=lbox.bbox,
                    reading_order=b_idx,
                    text=region_text,
                    html=None,
                    cells=[],
                    confidence=round(lbox.confidence, 4),
                    lines=text_lines,
                    recognizer=self.recognizer.model_label,
                )
            )
        vlm_blocks = sum(1 for b in blocks if b.type != "figure")
        return blocks, {
            "detect_sec": 0.0,
            "table_sec": round(table_sec, 3),
            "recognize_sec": round(recognize_sec, 3),
            "ctc_sec": 0.0,
            "vlm_sec": round(recognize_sec, 3),
            "ctc_blocks": 0,
            "vlm_blocks": vlm_blocks,
        }

    def _process_hybrid_mode(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        layout_boxes: list[LayoutBox],
    ) -> tuple[list[LayoutBlock], dict[str, Any]]:
        """Smart hybrid router: Enhanced CTC + cell-ink recovery primary, guarded VLM escalation."""
        assert self.text_detector is not None
        img_h, img_w = bgr_img.shape[:2]
        work_boxes = list(layout_boxes)

        skip_indices: set[int] = set()
        prebuilt_blocks: dict[int, LayoutBlock] = {}
        table_structs: dict[int, tuple[TableStructureOutput, float, float]] = {}
        ctc_candidate_indices: list[int] = []
        ctc_reasons: dict[int, str] = {}

        table_sec = 0.0
        vlm_sec = 0.0
        max_vlm_rows = getattr(self.settings, "hybrid_vlm_max_table_rows", 8)
        max_vlm_h = getattr(self.settings, "hybrid_vlm_max_table_height_px", 550)

        for b_idx, lbox in enumerate(work_boxes):
            block_id = f"p{page_number}_b{b_idx}"
            raw_low = lbox.raw_label.lower().strip()

            if lbox.type == "table" and self.settings.enable_tables:
                tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
                ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
                tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
                ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
                table_crop = bgr_img[ty1:ty2, tx1:tx2]
                crop_h = int(ty2 - ty1)

                if self.settings.hybrid_route_tables and self.settings.hybrid_vlm_variant == "openai-vlm":
                    vlm = self._get_vlm_recognizer()
                    t_vlm0 = time.perf_counter()
                    raw_md, vlm_conf = vlm.recognize_region(
                        table_crop, block_type="table", max_new_tokens=2048
                    )
                    vlm_sec += time.perf_counter() - t_vlm0
                    md_cells = _is_valid_vlm_table_markdown(raw_md, vlm_conf, lbox.bbox)
                    if md_cells:
                        prebuilt_blocks[b_idx] = LayoutBlock(
                            block_id=block_id,
                            type=lbox.type,
                            bbox=lbox.bbox,
                            reading_order=b_idx,
                            text=raw_md,
                            html=_build_rtl_table_html(md_cells),
                            cells=md_cells,
                            confidence=round(lbox.confidence, 4),
                            lines=self._region_text_to_lines(raw_md, lbox.bbox, vlm_conf),
                            recognizer=vlm.model_label,
                            routing_reason="table_vlm_whole",
                        )
                        continue
                    t_tbl0 = time.perf_counter()
                    t_struct, sx1, sy1, _ = self._extract_table_structure(bgr_img, lbox)
                    table_sec += time.perf_counter() - t_tbl0
                    table_structs[b_idx] = (t_struct, sx1, sy1)
                    ctc_candidate_indices.append(b_idx)
                    ctc_reasons[b_idx] = "table_ctc_fallback"
                    continue

                t_tbl0 = time.perf_counter()
                t_struct, sx1, sy1, struct_h = self._extract_table_structure(bgr_img, lbox)
                table_sec += time.perf_counter() - t_tbl0
                raw_cells = _parse_table_cells(t_struct, offset_x=sx1, offset_y=sy1)
                num_rows = max((c.row_end for c in raw_cells), default=0)

                if (
                    self.settings.hybrid_route_tables
                    and 0 < num_rows <= max_vlm_rows
                    and max(crop_h, struct_h) <= max_vlm_h
                ):
                    vlm = self._get_vlm_recognizer()
                    t_vlm0 = time.perf_counter()
                    raw_md, vlm_conf = vlm.recognize_region(
                        table_crop, block_type="text", max_new_tokens=2048
                    )
                    vlm_sec += time.perf_counter() - t_vlm0
                    md_cells = _is_valid_vlm_table_markdown(raw_md, vlm_conf, lbox.bbox)
                    if md_cells:
                        prebuilt_blocks[b_idx] = LayoutBlock(
                            block_id=block_id,
                            type=lbox.type,
                            bbox=lbox.bbox,
                            reading_order=b_idx,
                            text=raw_md,
                            html=_build_rtl_table_html(md_cells),
                            cells=md_cells,
                            confidence=round(lbox.confidence, 4),
                            lines=self._region_text_to_lines(raw_md, lbox.bbox, vlm_conf),
                            recognizer=vlm.model_label,
                            routing_reason="table_vlm_compact",
                        )
                        continue
                    table_structs[b_idx] = (t_struct, sx1, sy1)
                    ctc_candidate_indices.append(b_idx)
                    ctc_reasons[b_idx] = "table_ctc_fallback"
                    continue

                table_structs[b_idx] = (t_struct, sx1, sy1)
                ctc_candidate_indices.append(b_idx)
                ctc_reasons[b_idx] = (
                    "table_ctc_tall" if self.settings.hybrid_route_tables else "table_ctc_direct"
                )
                continue

            if self.settings.hybrid_route_complex and raw_low in _HYBRID_COMPLEX_RAW_LABELS:
                bx1 = max(0, int(np.floor(lbox.bbox[0])) - 6)
                by1 = max(0, int(np.floor(lbox.bbox[1])) - 6)
                bx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 6)
                by2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 6)
                region_crop = bgr_img[by1:by2, bx1:bx2]
                vlm = self._get_vlm_recognizer()
                t_vlm0 = time.perf_counter()
                region_text, rec_conf = vlm.recognize_region(region_crop, block_type="text")
                vlm_sec += time.perf_counter() - t_vlm0
                del region_crop
                if region_text.strip() and rec_conf >= 0.65 and not _is_vlm_boilerplate(region_text):
                    prebuilt_blocks[b_idx] = LayoutBlock(
                        block_id=block_id,
                        type=lbox.type,
                        bbox=lbox.bbox,
                        reading_order=b_idx,
                        text=region_text,
                        html=None,
                        cells=[],
                        confidence=round(lbox.confidence, 4),
                        lines=self._region_text_to_lines(region_text, lbox.bbox, rec_conf),
                        recognizer=vlm.model_label,
                        routing_reason=f"complex_label:{lbox.raw_label}",
                    )
                    continue

            if lbox.type == "figure":
                skip_indices.add(b_idx)
                prebuilt_blocks[b_idx] = LayoutBlock(
                    block_id=block_id,
                    type=lbox.type,
                    bbox=lbox.bbox,
                    reading_order=b_idx,
                    text="",
                    confidence=round(lbox.confidence, 4),
                    lines=[],
                )
                continue

            ctc_candidate_indices.append(b_idx)

        detect_sec = 0.0
        ctc_sec = 0.0
        ctc_blocks_out: dict[int, LayoutBlock] = {}

        if ctc_candidate_indices or not prebuilt_blocks:
            block_lines_map: dict[int, list[DetectedLine]] = {i: [] for i in range(len(work_boxes))}
            unassigned_lines: list[DetectedLine] = []

            t_det0 = time.perf_counter()
            page_det_lines = self.text_detector.detect(bgr_img)
            detect_sec = time.perf_counter() - t_det0

            for det_line in page_det_lines:
                best_idx: int | None = None
                best_overlap = 0.35
                for b_idx, lbox in enumerate(work_boxes):
                    if b_idx in skip_indices:
                        continue
                    ov = _overlap_over_line(det_line.bbox, lbox.bbox)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_idx = b_idx
                if best_idx is not None:
                    # Do not steal or duplicate lines already handled by whole-table/complex VLM
                    if best_idx not in prebuilt_blocks:
                        block_lines_map[best_idx].append(det_line)
                else:
                    unassigned_lines.append(det_line)

            if unassigned_lines:
                ux1 = min(ln.bbox[0] for ln in unassigned_lines)
                uy1 = min(ln.bbox[1] for ln in unassigned_lines)
                ux2 = max(ln.bbox[2] for ln in unassigned_lines)
                uy2 = max(ln.bbox[3] for ln in unassigned_lines)
                fallback_idx = len(work_boxes)
                work_boxes.append(
                    LayoutBox(
                        type="text",
                        raw_label="text",
                        bbox=(ux1, uy1, ux2, uy2),
                        confidence=0.5,
                    )
                )
                block_lines_map[fallback_idx] = unassigned_lines
                ctc_candidate_indices.append(fallback_idx)

            ctc_blocks_out, sub_det_sec, ctc_sec = self._run_ctc_blocks(
                page_number=page_number,
                bgr_img=bgr_img,
                work_boxes=work_boxes,
                block_lines_map=block_lines_map,
                ctc_indices=ctc_candidate_indices,
                table_structs=table_structs,
                routing_reasons=ctc_reasons,
            )
            detect_sec += sub_det_sec

        blocks: list[LayoutBlock] = []
        ctc_blocks = 0
        vlm_blocks = 0

        for b_idx, lbox in enumerate(work_boxes):
            block_id = f"p{page_number}_b{b_idx}"
            if b_idx in prebuilt_blocks:
                blk = prebuilt_blocks[b_idx]
                if blk.type != "figure" or blk.routing_reason:
                    vlm_blocks += 1
                blocks.append(blk)
                continue

            blk = ctc_blocks_out[b_idx]
            if lbox.type == "table":
                ctc_blocks += 1
                blocks.append(blk)
                continue

            text_lines = blk.lines
            joined_text = blk.text
            trigger_reason: str | None = None
            region_crop: np.ndarray | None = None

            if self.settings.hybrid_route_complex:
                if _has_multicolumn_overlap(text_lines):
                    trigger_reason = "complex_multicolumn"
                elif text_lines:
                    mean_conf = sum(ln.confidence for ln in text_lines) / len(text_lines)
                    if mean_conf < self.settings.hybrid_ctc_conf_threshold:
                        trigger_reason = "ctc_low_conf"
                elif not joined_text.strip():
                    bx1 = max(0, int(np.floor(lbox.bbox[0])) - 6)
                    by1 = max(0, int(np.floor(lbox.bbox[1])) - 6)
                    bx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 6)
                    by2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 6)
                    region_crop = bgr_img[by1:by2, bx1:bx2]
                    if _cell_has_ink(region_crop):
                        trigger_reason = "ctc_empty_fallback"

            if trigger_reason is not None:
                if region_crop is None:
                    bx1 = max(0, int(np.floor(lbox.bbox[0])) - 6)
                    by1 = max(0, int(np.floor(lbox.bbox[1])) - 6)
                    bx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 6)
                    by2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 6)
                    region_crop = bgr_img[by1:by2, bx1:bx2]

                vlm = self._get_vlm_recognizer()
                prompt_type = "text" if lbox.type == "table" else lbox.type
                t_vlm0 = time.perf_counter()
                vlm_text, vlm_conf = vlm.recognize_region(region_crop, block_type=prompt_type)
                vlm_sec += time.perf_counter() - t_vlm0
                del region_crop

                if vlm_text.strip() and vlm_conf >= 0.65 and not _is_vlm_boilerplate(vlm_text):
                    vlm_lines = self._region_text_to_lines(vlm_text, lbox.bbox, vlm_conf)
                    vlm_blocks += 1
                    blocks.append(
                        LayoutBlock(
                            block_id=block_id,
                            type=lbox.type,
                            bbox=lbox.bbox,
                            reading_order=b_idx,
                            text=vlm_text,
                            html=None,
                            cells=[],
                            confidence=round(lbox.confidence, 4),
                            lines=vlm_lines,
                            recognizer=vlm.model_label,
                            routing_reason=trigger_reason,
                        )
                    )
                    continue
            elif region_crop is not None:
                del region_crop

            ctc_blocks += 1
            blocks.append(blk.model_copy(update={"routing_reason": blk.routing_reason or "ctc_simple"}))

        return blocks, {
            "detect_sec": round(detect_sec, 3),
            "table_sec": round(table_sec, 3),
            "recognize_sec": round(ctc_sec + vlm_sec, 3),
            "ctc_sec": round(ctc_sec, 3),
            "vlm_sec": round(vlm_sec, 3),
            "ctc_blocks": ctc_blocks,
            "vlm_blocks": vlm_blocks,
        }

    def process_page(self, page_number: int, bgr_img: np.ndarray) -> PageResult:
        """Process exactly 1 page sequentially under GPU lock and release VRAM afterward."""
        with self._gpu_lock:
            t0 = time.perf_counter()
            cuda_active = self._cuda_is_active()
            if cuda_active:
                self._reset_vram_peaks()

            img_h, img_w = bgr_img.shape[:2]
            t_layout0 = time.perf_counter()
            layout_boxes = self.layout_analyzer.detect_layout(bgr_img)
            layout_sec = round(time.perf_counter() - t_layout0, 3)

            if not layout_boxes:
                layout_boxes = [
                    LayoutBox(
                        type="text",
                        raw_label="text",
                        bbox=(0.0, 0.0, float(img_w), float(img_h)),
                        confidence=1.0,
                    )
                ]

            if getattr(self, "is_hybrid", False):
                blocks, sub_timings = self._process_hybrid_mode(page_number, bgr_img, layout_boxes)
                model_label = f"Hybrid ({self.recognizer.model_label} + {self.settings.hybrid_vlm_variant})"
            elif self.recognizer.granularity == "region":
                blocks, sub_timings = self._process_region_mode(page_number, bgr_img, layout_boxes)
                model_label = self.recognizer.model_label
            else:
                blocks, sub_timings = self._process_line_mode(page_number, bgr_img, layout_boxes)
                model_label = self.recognizer.model_label

            latency_sec = round(time.perf_counter() - t0, 3)
            peak_vram_mb: float | None = None
            if cuda_active:
                _, peak_vram_mb = self._read_vram_mb()

            rss_ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
            stages = StageTimings(
                layout_sec=layout_sec,
                detect_sec=float(sub_timings["detect_sec"]),
                table_sec=float(sub_timings["table_sec"]),
                recognize_sec=float(sub_timings["recognize_sec"]),
                ctc_sec=float(sub_timings.get("ctc_sec", 0.0)),
                vlm_sec=float(sub_timings.get("vlm_sec", 0.0)),
            )
            num_blocks = len(blocks)
            num_lines = sum(len(b.lines) for b in blocks)
            num_tables = sum(1 for b in blocks if b.type == "table")
            char_count = sum(len(b.text) for b in blocks)
            ctc_blocks = int(sub_timings.get("ctc_blocks", 0))
            vlm_blocks = int(sub_timings.get("vlm_blocks", 0))

            routing_suffix = (
                f" [ctc={ctc_blocks}({stages.ctc_sec:.2f}s) vlm={vlm_blocks}({stages.vlm_sec:.2f}s)]"
                if getattr(self, "is_hybrid", False)
                else ""
            )
            logger.info(
                "Page %d (%s) in %.2fs [layout=%.2fs det=%.2fs table=%.2fs rec=%.2fs]%s | "
                "blocks=%d lines=%d tables=%d | peak_vram=%s MB | rss_ram=%.1f MB",
                page_number,
                model_label,
                latency_sec,
                stages.layout_sec,
                stages.detect_sec,
                stages.table_sec,
                stages.recognize_sec,
                routing_suffix,
                num_blocks,
                num_lines,
                num_tables,
                f"{peak_vram_mb:.1f}" if peak_vram_mb is not None else "n/a",
                rss_ram_mb,
            )

            gc.collect()
            if cuda_active and self.settings.clear_cuda_cache_per_page:
                self._empty_cuda_caches()

            return PageResult(
                page_number=page_number,
                width=img_w,
                height=img_h,
                blocks=blocks,
                metrics=PageMetrics(
                    latency_sec=latency_sec,
                    peak_vram_mb=peak_vram_mb,
                    rss_ram_mb=rss_ram_mb,
                    stages=stages,
                    num_blocks=num_blocks,
                    num_lines=num_lines,
                    num_tables=num_tables,
                    char_count=char_count,
                    ctc_blocks=ctc_blocks,
                    vlm_blocks=vlm_blocks,
                ),
            )

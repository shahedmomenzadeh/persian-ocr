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
    TextRecognizerProtocol,
    create_recognizer,
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


class InferenceEngine:
    """Sequential, VRAM-bounded per-page OCR engine (Paddle Layout + Line/Region Recognizer)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        self._gpu_lock = threading.Lock()
        self.layout_analyzer = LayoutAnalyzer(settings)
        self.recognizer: TextRecognizerProtocol = create_recognizer(settings)
        self.text_detector: TextLineDetector | None = (
            TextLineDetector(settings) if self.recognizer.granularity == "line" else None
        )

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
            BinaRecognizer.clear_cache()
            BinaNemotronRecognizer.clear_cache()
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

        flat_crops: list[np.ndarray] = []
        crop_refs: list[tuple[int, DetectedLine]] = []
        for b_idx, det_lines in block_lines_map.items():
            if layout_boxes[b_idx].type == "figure":
                continue
            for det_line in det_lines:
                crop = self.text_detector.extract_crop(bgr_img, det_line.polygon, det_line.bbox)
                flat_crops.append(crop)
                crop_refs.append((b_idx, det_line))

        t_rec0 = time.perf_counter()
        recognized = self.recognizer.recognize_batch(
            flat_crops, batch_size=self.settings.rec_batch_size
        )
        recognize_sec = time.perf_counter() - t_rec0
        del flat_crops

        recognized_by_block: dict[int, list[TextLine]] = {i: [] for i in range(len(layout_boxes))}
        for (b_idx, det_line), (rec_text, rec_score) in zip(crop_refs, recognized):
            recognized_by_block[b_idx].append(
                TextLine(
                    text=rec_text,
                    bbox=det_line.bbox,
                    polygon=det_line.polygon,
                    confidence=round(rec_score, 4),
                )
            )

        blocks: list[LayoutBlock] = []
        for b_idx, lbox in enumerate(layout_boxes):
            text_lines = recognized_by_block.get(b_idx, [])
            block_id = f"p{page_number}_b{b_idx}"
            table_cells: list[TableCell] = []
            table_html: str | None = None

            if lbox.type == "table" and b_idx in table_structs:
                t_struct, off_x, off_y = table_structs[b_idx]
                raw_cells = _parse_table_cells(t_struct, offset_x=off_x, offset_y=off_y)
                if raw_cells:
                    cell_lines: list[list[TextLine]] = [[] for _ in raw_cells]
                    for tline in text_lines:
                        best_cell_idx: int | None = None
                        best_metric = (0.30, 0.0)
                        for c_idx, cell in enumerate(raw_cells):
                            ov = _overlap_over_line(tline.bbox, cell.bbox)
                            if ov >= 0.30:
                                metric = (ov, _iou(tline.bbox, cell.bbox))
                                if metric > best_metric:
                                    best_metric = metric
                                    best_cell_idx = c_idx
                        if best_cell_idx is not None:
                            cell_lines[best_cell_idx].append(tline)

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
            blocks.append(
                LayoutBlock(
                    block_id=block_id,
                    type=lbox.type,
                    bbox=lbox.bbox,
                    reading_order=b_idx,
                    text=joined_text,
                    html=table_html,
                    cells=table_cells,
                    confidence=round(lbox.confidence, 4),
                    lines=text_lines,
                )
            )
        return blocks, {
            "detect_sec": round(detect_sec, 3),
            "table_sec": round(table_sec, 3),
            "recognize_sec": round(recognize_sec, 3),
        }

    def _process_region_mode(
        self,
        page_number: int,
        bgr_img: np.ndarray,
        layout_boxes: list[LayoutBox],
    ) -> tuple[list[LayoutBlock], dict[str, float]]:
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
                tx1 = max(0, int(np.floor(lbox.bbox[0])) - 4)
                ty1 = max(0, int(np.floor(lbox.bbox[1])) - 4)
                tx2 = min(img_w, int(np.ceil(lbox.bbox[2])) + 4)
                ty2 = min(img_h, int(np.ceil(lbox.bbox[3])) + 4)
                table_crop = bgr_img[ty1:ty2, tx1:tx2]

                raw_cells: list[TableCell] = []
                if table_crop.size > 0:
                    t_tbl0 = time.perf_counter()
                    t_struct = self.layout_analyzer.analyze_table(table_crop)
                    table_sec += time.perf_counter() - t_tbl0
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
                        cx1 = max(0, int(np.floor(cell.bbox[0])) - 2)
                        cy1 = max(0, int(np.floor(cell.bbox[1])) - 2)
                        cx2 = min(img_w, int(np.ceil(cell.bbox[2])) + 2)
                        cy2 = min(img_h, int(np.ceil(cell.bbox[3])) + 2)
                        cell_crop = bgr_img[cy1:cy2, cx1:cx2]

                        if _cell_has_ink(cell_crop):
                            t_rec0 = time.perf_counter()
                            raw_cell_text, cell_conf = self.recognizer.recognize_region(
                                cell_crop,
                                block_type="table_cell",
                                max_new_tokens=128,
                            )
                            recognize_sec += time.perf_counter() - t_rec0
                            cell_text = " ".join(
                                ln.strip() for ln in raw_cell_text.splitlines() if ln.strip()
                            )
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
                    blocks.append(
                        LayoutBlock(
                            block_id=block_id,
                            type=lbox.type,
                            bbox=lbox.bbox,
                            reading_order=b_idx,
                            text="\n".join(ln.text for ln in text_lines if ln.text),
                            html=table_html,
                            cells=table_cells,
                            confidence=round(lbox.confidence, 4),
                            lines=text_lines,
                        )
                    )
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

            raw_lines = [ln.strip() for ln in region_text.splitlines() if ln.strip()]
            text_lines = []
            if raw_lines:
                step_y = max(1.0, (lbox.bbox[3] - lbox.bbox[1]) / len(raw_lines))
                for l_idx, line_str in enumerate(raw_lines):
                    ly1 = lbox.bbox[1] + l_idx * step_y
                    ly2 = min(lbox.bbox[3], ly1 + step_y)
                    text_lines.append(
                        TextLine(
                            text=line_str,
                            bbox=(lbox.bbox[0], ly1, lbox.bbox[2], ly2),
                            confidence=round(rec_conf, 4),
                        )
                    )

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
                )
            )
        return blocks, {
            "detect_sec": 0.0,
            "table_sec": round(table_sec, 3),
            "recognize_sec": round(recognize_sec, 3),
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

            if self.recognizer.granularity == "region":
                blocks, sub_timings = self._process_region_mode(page_number, bgr_img, layout_boxes)
            else:
                blocks, sub_timings = self._process_line_mode(page_number, bgr_img, layout_boxes)

            latency_sec = round(time.perf_counter() - t0, 3)
            peak_vram_mb: float | None = None
            if cuda_active:
                _, peak_vram_mb = self._read_vram_mb()

            rss_ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
            stages = StageTimings(
                layout_sec=layout_sec,
                detect_sec=sub_timings["detect_sec"],
                table_sec=sub_timings["table_sec"],
                recognize_sec=sub_timings["recognize_sec"],
            )
            num_blocks = len(blocks)
            num_lines = sum(len(b.lines) for b in blocks)
            num_tables = sum(1 for b in blocks if b.type == "table")
            char_count = sum(len(b.text) for b in blocks)

            logger.info(
                "Page %d (%s) in %.2fs [layout=%.2fs det=%.2fs table=%.2fs rec=%.2fs] | "
                "blocks=%d lines=%d tables=%d | peak_vram=%s MB | rss_ram=%.1f MB",
                page_number,
                self.recognizer.model_label,
                latency_sec,
                stages.layout_sec,
                stages.detect_sec,
                stages.table_sec,
                stages.recognize_sec,
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
                ),
            )

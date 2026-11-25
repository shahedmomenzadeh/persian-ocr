"""PaddleOCR 3.x LayoutDetection & TableStructureRecognition wrappers (no formula/chart/seal)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config.settings import Settings, configure_cache_env
from src.schemas.document import BBox, BlockType

logger = logging.getLogger(__name__)

# Map PaddleOCR/PP-DocLayout labels to canonical BlockType values
_LAYOUT_LABEL_MAP: dict[str, BlockType] = {
    "title": "title",
    "doc_title": "title",
    "paragraph_title": "title",
    "section_header": "title",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "formula": "figure",
    "header": "header",
    "header_image": "header",
    "page_header": "header",
    "footer": "footer",
    "footer_image": "footer",
    "page_footer": "footer",
    "footnote": "footer",
    "number": "footer",
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "aside_text": "text",
    "figure_title": "text",
    "table_title": "text",
    "algorithm": "text",
    "list": "text",
}


def to_paddle_device(device: str) -> str:
    """Convert Settings.device ('gpu' | 'cpu') to PaddleX device string."""
    if device == "gpu":
        return "gpu:0"
    return "cpu"


def _normalize_bbox(coord: list[float] | tuple[float, ...], img_w: int, img_h: int) -> BBox:
    """Normalize 4-point [x1,y1,x2,y2] or 8-point polygon coordinates into clamped BBox."""
    if len(coord) == 4:
        x1, y1, x2, y2 = coord
    elif len(coord) >= 8:
        xs = coord[0::2]
        ys = coord[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return (0.0, 0.0, float(img_w), float(img_h))

    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(x1, min(float(x2), float(img_w)))
    y2 = max(y1, min(float(y2), float(img_h)))
    return (x1, y1, x2, y2)


@dataclass(slots=True)
class LayoutBox:
    """Single detected layout region."""

    type: BlockType
    raw_label: str
    bbox: BBox
    confidence: float


@dataclass(slots=True)
class TableStructureOutput:
    """Raw table structure HTML tokens and corresponding cell bounding boxes."""

    structure_tokens: list[str]
    cell_bboxes: list[BBox]


class LayoutAnalyzer:
    """Wraps PaddleOCR LayoutDetection and lazy TableStructureRecognition."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        from paddleocr import LayoutDetection

        paddle_dev = to_paddle_device(settings.device)
        logger.info(
            "Loading LayoutDetection model=%s on device=%s",
            settings.layout_model_name,
            paddle_dev,
        )
        self._layout_model = LayoutDetection(
            model_name=settings.layout_model_name,
            device=paddle_dev,
            layout_nms=True,
        )
        self._table_model = None

    def _get_table_model(self):
        if self._table_model is None:
            configure_cache_env(self.settings.models_dir)
            from paddleocr import TableStructureRecognition

            paddle_dev = to_paddle_device(self.settings.device)
            logger.info(
                "Loading TableStructureRecognition model=%s on device=%s",
                self.settings.table_model_name,
                paddle_dev,
            )
            self._table_model = TableStructureRecognition(
                model_name=self.settings.table_model_name,
                device=paddle_dev,
            )
        return self._table_model

    @property
    def loaded_model_names(self) -> list[str]:
        names = [self.settings.layout_model_name]
        if self._table_model is not None:
            names.append(self.settings.table_model_name)
        return names

    def detect_layout(self, bgr_img: np.ndarray) -> list[LayoutBox]:
        """Run layout detection on a single BGR page image."""
        img_h, img_w = bgr_img.shape[:2]
        results = self._layout_model.predict(input=bgr_img, batch_size=1, layout_nms=True)
        boxes: list[LayoutBox] = []
        for res in results:
            raw_boxes = res.get("boxes", []) if hasattr(res, "get") else res["boxes"]
            for item in raw_boxes:
                raw_label = str(item.get("label", "text"))
                mapped_type = _LAYOUT_LABEL_MAP.get(raw_label.lower().strip(), "text")
                coord = item.get("coordinate", [0, 0, img_w, img_h])
                score = float(item.get("score", 0.0))
                bbox = _normalize_bbox(coord, img_w=img_w, img_h=img_h)
                if (bbox[2] - bbox[0]) < 2 or (bbox[3] - bbox[1]) < 2:
                    continue
                boxes.append(
                    LayoutBox(
                        type=mapped_type,
                        raw_label=raw_label,
                        bbox=bbox,
                        confidence=score,
                    )
                )
        return boxes

    def analyze_table(self, table_bgr: np.ndarray) -> TableStructureOutput:
        """Predict HTML table structure tokens and cell bounding boxes on a cropped table."""
        if table_bgr.size == 0 or table_bgr.shape[0] < 4 or table_bgr.shape[1] < 4:
            return TableStructureOutput(structure_tokens=[], cell_bboxes=[])

        img_h, img_w = table_bgr.shape[:2]
        model = self._get_table_model()
        results = model.predict(input=table_bgr, batch_size=1)
        if not results:
            return TableStructureOutput(structure_tokens=[], cell_bboxes=[])

        res = results[0]
        tokens = [str(tok) for tok in (res.get("structure", []) if hasattr(res, "get") else res["structure"])]
        raw_bboxes = res.get("bbox", []) if hasattr(res, "get") else res["bbox"]
        bboxes = [_normalize_bbox(list(b), img_w=img_w, img_h=img_h) for b in raw_bboxes]
        return TableStructureOutput(structure_tokens=tokens, cell_bboxes=bboxes)

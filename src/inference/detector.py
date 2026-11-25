"""PaddleOCR 3.x TextDetection wrapper and dot-safe line crop/warp extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import Settings, configure_cache_env
from src.inference.layout import to_paddle_device
from src.schemas.document import BBox

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedLine:
    """Single detected text line polygon and bounding box."""

    bbox: BBox
    polygon: list[list[float]]
    score: float


def _poly_to_bbox(poly: np.ndarray, img_w: int, img_h: int) -> BBox:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = max(0.0, min(float(xs.min()), float(img_w)))
    y1 = max(0.0, min(float(ys.min()), float(img_h)))
    x2 = max(x1, min(float(xs.max()), float(img_w)))
    y2 = max(y1, min(float(ys.max()), float(img_h)))
    return (x1, y1, x2, y2)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class TextLineDetector:
    """Wraps PaddleOCR TextDetection with max_side_limit VRAM safety."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        from paddleocr import TextDetection

        paddle_dev = to_paddle_device(settings.device)
        logger.info(
            "Loading TextDetection model=%s (limit_side_len=%d) on device=%s",
            settings.det_model_name,
            settings.det_max_side_len,
            paddle_dev,
        )
        self._model = TextDetection(
            model_name=settings.det_model_name,
            limit_side_len=settings.det_max_side_len,
            limit_type="max",
            device=paddle_dev,
        )

    def detect(self, bgr_img: np.ndarray) -> list[DetectedLine]:
        """Detect text line polygons in a BGR image."""
        if bgr_img.size == 0 or bgr_img.shape[0] < 4 or bgr_img.shape[1] < 4:
            return []

        img_h, img_w = bgr_img.shape[:2]
        results = self._model.predict(input=bgr_img, batch_size=1)
        lines: list[DetectedLine] = []

        for res in results:
            polys = res.get("dt_polys", []) if hasattr(res, "get") else res["dt_polys"]
            scores = res.get("dt_scores", []) if hasattr(res, "get") else res.get("dt_scores", [])
            for idx, poly_raw in enumerate(polys):
                poly = np.asarray(poly_raw, dtype=np.float32).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                bbox = _poly_to_bbox(poly, img_w=img_w, img_h=img_h)
                if (bbox[2] - bbox[0]) < 2 or (bbox[3] - bbox[1]) < 2:
                    continue
                score = float(scores[idx]) if idx < len(scores) else 1.0
                lines.append(
                    DetectedLine(
                        bbox=bbox,
                        polygon=[[float(p[0]), float(p[1])] for p in poly],
                        score=score,
                    )
                )
        return lines

    @staticmethod
    def extract_crop(
        bgr_img: np.ndarray,
        polygon: list[list[float]],
        bbox: BBox,
    ) -> np.ndarray:
        """Extract a rectified or gently padded line crop preserving Persian dots."""
        img_h, img_w = bgr_img.shape[:2]
        x1, y1, x2, y2 = bbox
        box_h = max(1.0, y2 - y1)
        box_w = max(1.0, x2 - x1)

        # Add ~5% vertical padding (min 2px) so upper/lower Persian dots are never clipped
        pad_y = max(2, int(round(box_h * 0.06)))
        pad_x = max(2, int(round(min(box_h * 0.1, 6.0))))

        if len(polygon) == 4:
            pts = _order_quad_points(np.asarray(polygon, dtype=np.float32))
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) if abs(dx) > 1.0 else 0.0

            # Use perspective warp only when line is noticeably tilted (> 2 deg)
            if 2.0 < angle < 45.0:
                w_top = np.linalg.norm(pts[1] - pts[0])
                w_bot = np.linalg.norm(pts[2] - pts[3])
                h_left = np.linalg.norm(pts[3] - pts[0])
                h_right = np.linalg.norm(pts[2] - pts[1])
                dst_w = max(4, int(round(max(w_top, w_bot))))
                dst_h = max(4, int(round(max(h_left, h_right))))

                # Expand slightly vertically in destination coords
                dst_pts = np.array(
                    [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                    dtype=np.float32,
                )
                matrix = cv2.getPerspectiveTransform(pts, dst_pts)
                warped = cv2.warpPerspective(
                    bgr_img,
                    matrix,
                    (dst_w, dst_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                if warped.shape[0] >= 2 and warped.shape[1] >= 2:
                    return warped

        cx1 = max(0, int(np.floor(x1)) - pad_x)
        cy1 = max(0, int(np.floor(y1)) - pad_y)
        cx2 = min(img_w, int(np.ceil(x2)) + pad_x)
        cy2 = min(img_h, int(np.ceil(y2)) + pad_y)
        if cx2 <= cx1 or cy2 <= cy1 or box_w < 2:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return bgr_img[cy1:cy2, cx1:cx2].copy()

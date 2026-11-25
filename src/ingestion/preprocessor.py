"""Persian-safe image preprocessing: gentle deskew & dot-preserving contrast norm."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config.settings import Settings


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew angle in degrees within [-15, +15]."""
    h, w = gray.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=max(40, small.shape[1] // 10),
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Keep near-horizontal text/rule lines only
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0

    return float(np.median(angles))


def _rotate_bgr(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate BGR image around center with white background padding."""
    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _apply_mild_clahe(bgr: np.ndarray) -> np.ndarray:
    """Enhance local contrast mildly on LAB L-channel without eroding Persian dots."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_norm = clahe.apply(l_chan)
    merged = cv2.merge((l_norm, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_page(image: Image.Image, settings: Settings) -> np.ndarray:
    """Convert PIL RGB image to preprocessed BGR uint8 ndarray."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if settings.enable_deskew:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        angle = _estimate_skew_angle(gray)
        if 0.3 < abs(angle) <= 15.0:
            bgr = _rotate_bgr(bgr, angle)

    if settings.enable_denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    if settings.enable_contrast_norm:
        bgr = _apply_mild_clahe(bgr)

    return bgr

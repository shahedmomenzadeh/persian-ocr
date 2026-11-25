"""Pluggable text recognizers: Paddle Bina-0.2 (line) & HF Bina-0.4-Koochik (region)."""

from __future__ import annotations

import copy
import importlib.util
import logging
import math
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image

from config.settings import RecognizerVariant, Settings, configure_cache_env
from src.inference.layout import to_paddle_device

logger = logging.getLogger(__name__)

_BINA_BASE_HEIGHT = 48
_BINA_BASE_WIDTH = 768

_NEMOTRON_TEXT_PROMPT = "</s><s><predict_text_in_pic>"
_NEMOTRON_MARKUP_PROMPT = "</s><s><output_markdown><predict_text_in_pic>"


@runtime_checkable
class TextRecognizerProtocol(Protocol):
    """Common interface for line-level Paddle and region-level Transformers recognizers."""

    model_label: str
    granularity: Literal["line", "region"]

    def recognize_batch(
        self,
        crops: list[np.ndarray],
        batch_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Recognize a batch of BGR crops and return (text, confidence) pairs."""
        ...

    def recognize_region(
        self,
        crop: np.ndarray,
        block_type: str = "text",
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Recognize a single BGR layout region crop and return (text, confidence)."""
        ...


def _load_module_from_path(module_name: str, script_path: Path) -> Any:
    """Dynamically load a Python module from an explicit file path."""
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required model script not found at {script_path}. "
            "Run `python scripts/download_models.py` first."
        )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_crop_orientation(crop: np.ndarray) -> np.ndarray:
    """Rotate tall vertical text crops 90 degrees clockwise per Bina-0.2's input contract."""
    h, w = crop.shape[:2]
    if h >= 32 and h > 1.5 * w:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _find_whitespace_split(
    ink_density: np.ndarray,
    col_bright: np.ndarray,
    win_start: int,
    win_end: int,
    ideal_x: int,
) -> int:
    """Find the midpoint of the cleanest low-ink whitespace valley inside [win_start, win_end)."""
    if win_end <= win_start + 1:
        return max(win_start, min(win_end - 1, ideal_x))

    sub_ink = ink_density[win_start:win_end]
    min_ink = float(sub_ink.min())
    valley_mask = sub_ink <= (min_ink + 0.02)

    best_score: tuple[int, float, float] | None = None
    best_mid = ideal_x
    idx = 0
    n = len(valley_mask)
    while idx < n:
        if not valley_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and valley_mask[idx]:
            idx += 1
        end = idx
        run_len = end - start
        mid = win_start + (start + end - 1) // 2
        avg_bright = float(col_bright[win_start + start : win_start + end].mean())
        score = (run_len, avg_bright, -abs(mid - ideal_x))
        if best_score is None or score > best_score:
            best_score = score
            best_mid = mid

    return best_mid


def _split_wide_line(
    crop: np.ndarray,
    max_wh_ratio: float = 14.0,
    target_chunk_ratio: float = 12.0,
) -> list[np.ndarray]:
    """Split wide line crops at internal whitespace valleys in RTL (right-to-left) order."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0 or (w / float(h)) <= max_wh_ratio:
        return [crop]

    n_parts = max(2, math.ceil((w / float(h)) / max(2.0, target_chunk_ratio)))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink_thresh = min(230.0, float(np.percentile(gray, 85)) - 10.0)
    ink_density = (gray < ink_thresh).mean(axis=0)
    col_bright = gray.mean(axis=0)

    margin = max(8, int(round(w * 0.08)))
    split_points: list[int] = [0]
    for k in range(1, n_parts):
        ideal_x = int(round(k * w / n_parts))
        win_start = max(split_points[-1] + 4, ideal_x - margin)
        win_end = min(w - 4, ideal_x + margin)
        if win_end <= win_start:
            split_x = max(split_points[-1] + 2, min(w - 2, ideal_x))
        else:
            split_x = _find_whitespace_split(ink_density, col_bright, win_start, win_end, ideal_x)
        split_points.append(split_x)
    split_points.append(w)

    border_rows = np.vstack([crop[:1, :], crop[-1:, :]]).reshape(-1, 3)
    bg_color = tuple(int(v) for v in np.median(border_rows, axis=0))

    segments: list[np.ndarray] = []
    for left_x, right_x in reversed(list(zip(split_points[:-1], split_points[1:]))):
        seg = crop[:, left_x:right_x]
        if seg.shape[1] < 2:
            continue
        pad_l = 4 if left_x > 0 else 0
        pad_r = 4 if right_x < w else 0
        if pad_l or pad_r:
            seg = cv2.copyMakeBorder(
                seg,
                top=0,
                bottom=0,
                left=pad_l,
                right=pad_r,
                borderType=cv2.BORDER_CONSTANT,
                value=bg_color,
            )
        segments.append(seg)

    return segments or [crop]


def _estimate_resized_width(crop: np.ndarray) -> int:
    """Match PaddleX OCRReisizeNormImg width calculation for 3x48x768."""
    h, w = crop.shape[:2]
    if h <= 0:
        return _BINA_BASE_WIDTH
    return max(_BINA_BASE_WIDTH, min(3200, int(round(_BINA_BASE_HEIGHT * (w / float(h))))))


def _bucket_by_resized_width(
    segments: list[np.ndarray],
    batch_size: int,
    max_width_ratio_diff: float = 1.05,
) -> list[list[int]]:
    """Group segment indices into uniform-width buckets so ToBatch adds minimal/zero padding."""
    if not segments:
        return []
    if batch_size <= 1:
        return [[i] for i in range(len(segments))]

    widths = [_estimate_resized_width(seg) for seg in segments]
    sorted_indices = sorted(range(len(segments)), key=lambda i: widths[i])

    buckets: list[list[int]] = []
    current_bucket: list[int] = []
    bucket_min_w = 0

    for idx in sorted_indices:
        w = widths[idx]
        if not current_bucket:
            current_bucket = [idx]
            bucket_min_w = w
            continue
        if len(current_bucket) >= batch_size or w > int(bucket_min_w * max_width_ratio_diff):
            buckets.append(current_bucket)
            current_bucket = [idx]
            bucket_min_w = w
        else:
            current_bucket.append(idx)

    if current_bucket:
        buckets.append(current_bucket)
    return buckets


def _resolve_bina_paddle_dir(settings: Settings) -> tuple[Path, str]:
    """Return (model_dir, human_label) for Paddle-based Bina-0.2 variants."""
    if settings.rec_model_variant == "bina-0.2-rizeh":
        return Path(settings.bina_rizeh_model_dir).resolve(), "Bina-0.2-Rizeh"
    return Path(settings.bina_model_dir).resolve(), "Bina-0.2-RizehPizeh"


class BinaRecognizer:
    """Batched in-memory line recognizer for Bina-0.2-RizehPizeh and Bina-0.2-Rizeh."""

    granularity: Literal["line", "region"] = "line"
    _instances: dict[tuple[str, str], "BinaRecognizer"] = {}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        bina_dir, label = _resolve_bina_paddle_dir(settings)
        self.model_label = label
        weights_dir = bina_dir / "inference"
        if not weights_dir.is_dir():
            raise FileNotFoundError(
                f"{label} inference weights not found at {weights_dir}. "
                f"Run `python scripts/download_models.py --model {settings.rec_model_variant}` first."
            )

        bina_mod = _load_module_from_path(
            f"bina_text_recognition_{settings.rec_model_variant.replace('-', '_')}",
            bina_dir / "bina_text_recognition.py",
        )
        paddle_dev = to_paddle_device(settings.device)
        logger.info("Loading %s recognizer from %s on device=%s", label, weights_dir, paddle_dev)
        self._bina = bina_mod.BinaTextRecognition(model_dir=weights_dir, device=paddle_dev)

    @classmethod
    def get_instance(cls, settings: Settings) -> "BinaRecognizer":
        """Return a cached singleton BinaRecognizer per (model_dir, device)."""
        bina_dir, _ = _resolve_bina_paddle_dir(settings)
        key = (str(bina_dir), settings.device)
        if key not in cls._instances:
            cls._instances[key] = cls(settings)
        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Release cached recognizer instances."""
        cls._instances.clear()

    def recognize_region(
        self,
        crop: np.ndarray,
        block_type: str = "text",
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Recognize a single crop (delegates to recognize_batch)."""
        del max_new_tokens
        return self.recognize_batch([crop], batch_size=1)[0]

    def recognize_batch(
        self,
        crops: list[np.ndarray],
        batch_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Recognize cropped BGR line images safely using wide-line splits and width bucketing."""
        if not crops:
            return []

        eff_batch = max(1, batch_size if batch_size is not None else self.settings.rec_batch_size)
        results: list[tuple[str, float]] = [("", 0.0) for _ in range(len(crops))]

        flat_segments: list[np.ndarray] = []
        seg_owner_map: list[int] = []

        for orig_idx, crop in enumerate(crops):
            if (
                crop is None
                or crop.size == 0
                or crop.ndim != 3
                or crop.shape[0] < 2
                or crop.shape[1] < 2
            ):
                continue

            oriented = _prepare_crop_orientation(crop)
            segments = _split_wide_line(
                oriented,
                max_wh_ratio=self.settings.rec_max_wh_ratio,
                target_chunk_ratio=self.settings.rec_target_chunk_ratio,
            )
            for seg in segments:
                flat_segments.append(seg)
                seg_owner_map.append(orig_idx)

        if not flat_segments:
            return results

        seg_outputs: list[tuple[str, float]] = [("", 0.0) for _ in range(len(flat_segments))]
        buckets = _bucket_by_resized_width(flat_segments, batch_size=eff_batch)

        for bucket in buckets:
            chunk = [flat_segments[i] for i in bucket]
            preds = list(self._bina.predict(chunk, batch_size=len(chunk)))
            for seg_idx, pred in zip(bucket, preds):
                seg_outputs[seg_idx] = (
                    str(pred.get("text", "")),
                    float(pred.get("score", 0.0)),
                )

        grouped: dict[int, list[tuple[str, float]]] = {}
        for owner_idx, out in zip(seg_owner_map, seg_outputs):
            grouped.setdefault(owner_idx, []).append(out)

        for owner_idx, parts in grouped.items():
            non_empty = [(t.strip(), s) for t, s in parts if t.strip()]
            if not non_empty:
                continue
            joined_text = " ".join(t for t, _ in non_empty)
            total_chars = sum(len(t) for t, _ in non_empty)
            weighted_score = (
                sum(len(t) * s for t, s in non_empty) / total_chars
                if total_chars > 0
                else 0.0
            )
            results[owner_idx] = (joined_text, weighted_score)

        return results


def _patch_nemotron_cache_compat(model: Any) -> None:
    """Forward transformers >=5.20 `cache` kwarg into Nemotron's `past_key_values`."""
    orig_prepare = getattr(model, "prepare_inputs_for_generation", None)
    if orig_prepare is None:
        return

    def _compat_prepare_inputs_for_generation(
        input_ids: Any,
        past_key_values: Any = None,
        cache: Any = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is None and cache is not None:
            past_key_values = cache
        return orig_prepare(input_ids, past_key_values=past_key_values, **kwargs)

    model.prepare_inputs_for_generation = _compat_prepare_inputs_for_generation


class BinaNemotronRecognizer:
    """Hugging Face Transformers recognizer for Reza2kn/Bina-0.4-Koochik (Nemotron-3.5-Parse)."""

    model_label = "Bina-0.4-Koochik"
    granularity: Literal["line", "region"] = "region"
    _instances: dict[tuple[str, str], "BinaNemotronRecognizer"] = {}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_cache_env(settings.models_dir)
        model_dir = Path(settings.bina_koochik_model_dir).resolve()
        if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Bina-0.4-Koochik weights not found at {model_dir}. "
                "Run `python scripts/download_models.py --model bina-0.4-koochik` first."
            )

        import torch
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        torch_dev = (
            "cuda:0" if settings.device == "gpu" and torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if torch_dev.startswith("cuda") else torch.float32
        logger.info(
            "Loading Bina-0.4-Koochik from %s on device=%s (dtype=%s)",
            model_dir,
            torch_dev,
            dtype,
        )
        self._torch = torch
        self._device = torch_dev
        self._model = (
            AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(torch_dev)
        )
        _patch_nemotron_cache_compat(self._model)
        self._processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self._gen_config = GenerationConfig.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self._gen_config.max_new_tokens = settings.nemotron_max_new_tokens
        self._gen_config.do_sample = False
        self._gen_config.num_beams = 1
        self._gen_config.repetition_penalty = 1.1

        decoding_mod = _load_module_from_path(
            "bina_koochik_decoding",
            model_dir / "decoding.py",
        )
        self._generate_repaired = decoding_mod.generate_repaired

    @classmethod
    def get_instance(cls, settings: Settings) -> "BinaNemotronRecognizer":
        """Return a cached singleton BinaNemotronRecognizer per (model_dir, device)."""
        key = (str(Path(settings.bina_koochik_model_dir).resolve()), settings.device)
        if key not in cls._instances:
            cls._instances[key] = cls(settings)
        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Release cached recognizer instances."""
        cls._instances.clear()

    @staticmethod
    def select_prompt(block_type: str) -> str:
        """Use markup prompt for tables and plain OCR prompt for text/table_cell regions."""
        if block_type == "table":
            return _NEMOTRON_MARKUP_PROMPT
        if block_type == "table_cell":
            return _NEMOTRON_TEXT_PROMPT
        return _NEMOTRON_TEXT_PROMPT

    def recognize_region(
        self,
        crop: np.ndarray,
        block_type: str = "text",
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Recognize a BGR layout region crop using Bina-0.4-Koochik."""
        if (
            crop is None
            or crop.size == 0
            or crop.ndim != 3
            or crop.shape[0] < 4
            or crop.shape[1] < 4
        ):
            return ("", 0.0)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        prompt = self.select_prompt(block_type)
        gen_cfg = copy.deepcopy(self._gen_config)
        if max_new_tokens is not None:
            if isinstance(gen_cfg, dict):
                gen_cfg["max_new_tokens"] = max_new_tokens
            elif hasattr(gen_cfg, "max_new_tokens"):
                gen_cfg.max_new_tokens = max_new_tokens

        with self._torch.inference_mode():
            outcome = self._generate_repaired(
                self._model,
                self._processor,
                pil_img,
                prompt,
                gen_cfg,
            )

        selected = outcome.get("selected", {})
        text = str(selected.get("ocr_prediction", "")).strip()
        if not text:
            return ("", 0.0)

        flagged = bool(selected.get("signal", {}).get("flagged", False))
        ended_eos = bool(selected.get("ended_with_eos", False))
        capped = bool(selected.get("capped", False))
        if ended_eos and not capped and not flagged:
            conf = 1.0
        elif not flagged:
            conf = 0.65
        else:
            conf = 0.30
        return (text, conf)

    def recognize_batch(
        self,
        crops: list[np.ndarray],
        batch_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Recognize crops sequentially via recognize_region."""
        return [self.recognize_region(crop, block_type="text") for crop in crops]


def check_model_available(variant: RecognizerVariant, settings: Settings) -> bool:
    """Return True if the recognizer weights required for variant exist on disk."""
    if variant == "bina-0.2-rizehpizeh":
        base = Path(settings.bina_model_dir)
        return (base / "bina_text_recognition.py").is_file() and (
            base / "inference" / "inference.pdiparams"
        ).is_file()
    if variant == "bina-0.2-rizeh":
        base = Path(settings.bina_rizeh_model_dir)
        return (base / "bina_text_recognition.py").is_file() and (
            base / "inference" / "inference.pdiparams"
        ).is_file()
    if variant == "bina-0.4-koochik":
        base = Path(settings.bina_koochik_model_dir)
        return (base / "config.json").is_file() and (base / "model.safetensors").is_file()
    return False


def create_recognizer(settings: Settings) -> TextRecognizerProtocol:
    """Factory returning the configured recognizer backend for settings.rec_model_variant."""
    variant: RecognizerVariant = settings.rec_model_variant
    if variant in {"bina-0.2-rizehpizeh", "bina-0.2-rizeh"}:
        return BinaRecognizer.get_instance(settings)
    if variant == "bina-0.4-koochik":
        return BinaNemotronRecognizer.get_instance(settings)
    raise ValueError(f"Unsupported rec_model_variant: {variant!r}")

"""FastAPI service exposing /health, /models, /config, /ocr, /runs, and /outputs."""

from __future__ import annotations

import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import psutil
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import RecognizerVariant, Settings, SpellcheckMode, configure_cache_env
from src.inference.recognizer import check_model_available
from src.pipeline import DocumentPipeline
from src.postprocess.serializers import VALID_FORMATS, parse_formats, to_html, to_markdown, to_text
from src.storage.local import LocalStorageBackend
from src.storage.outputs import OutputManager

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[RecognizerVariant, ...] = (
    "bina-0.2-rizehpizeh",
    "bina-0.2-rizeh",
    "bina-0.4-koochik",
    "hybrid",
)
_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

_pipeline_lock = threading.Lock()
_pipeline_instance: DocumentPipeline | None = None
_active_model: RecognizerVariant = Settings().rec_model_variant
_default_formats: list[str] = ["json"]
_spellcheck_mode: SpellcheckMode = Settings().spellcheck_mode
_total_requests: int = 0
_last_run: dict[str, Any] | None = None


class ConfigUpdateRequest(BaseModel):
    """Runtime model, spellcheck, and default output-format configuration."""

    model: RecognizerVariant | None = None
    out_formats: list[str] | str | None = None
    spellcheck_mode: SpellcheckMode | None = None
    preload: bool = False


def get_pipeline(
    model: RecognizerVariant | None = None,
    spellcheck_mode: SpellcheckMode | None = None,
) -> DocumentPipeline:
    """Return the active DocumentPipeline, switching/unloading safely if model changes."""
    global _pipeline_instance, _active_model
    target_spellcheck = spellcheck_mode or _spellcheck_mode
    with _pipeline_lock:
        if _pipeline_instance is not None and (
            model is None or _pipeline_instance.settings.rec_model_variant == model
        ):
            if _pipeline_instance.settings.spellcheck_mode != target_spellcheck:
                _pipeline_instance.settings = _pipeline_instance.settings.model_copy(
                    update={"spellcheck_mode": target_spellcheck}
                )
            return _pipeline_instance

        target_model = model or _active_model
        if _pipeline_instance is not None:
            _pipeline_instance.unload()
            _pipeline_instance = None

        settings = Settings(rec_model_variant=target_model, spellcheck_mode=target_spellcheck)
        configure_cache_env(settings.models_dir)
        _pipeline_instance = DocumentPipeline(
            settings=settings,
            storage=LocalStorageBackend(settings.data_dir / "api_outputs"),
        )
        _active_model = target_model
        return _pipeline_instance


@asynccontextmanager
async def lifespan(_: FastAPI):
    pipeline = get_pipeline()
    logger.info(
        "FastAPI Persian OCR service initialized (device=%s, active_model=%s).",
        pipeline.settings.device,
        pipeline.settings.rec_model_variant,
    )
    yield


app = FastAPI(
    title="Persian OCR Pipeline API",
    version="0.1.0",
    lifespan=lifespan,
)


def _check_cuda_available() -> bool:
    try:
        import paddle

        return bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


def _get_gpu_name() -> str | None:
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return str(getattr(paddle.device.cuda.get_device_properties(0), "name", "GPU"))
    except Exception:
        pass
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    """Report active config, GPU status, loaded models, and observability metrics."""
    pipeline = get_pipeline()
    info = (
        pipeline.engine.health_info()
        if pipeline._engine is not None
        else {
            "device": pipeline.settings.device,
            "cuda_available": _check_cuda_available(),
            "gpu_name": _get_gpu_name(),
            "loaded_models": [],
            "vram_allocated_mb": 0.0,
            "vram_peak_mb": 0.0,
        }
    )
    rss_ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    base_dir = getattr(pipeline.storage, "base_dir", None)
    return {
        "status": "ok",
        "active_model": pipeline.settings.rec_model_variant,
        "default_formats": list(_default_formats),
        "spellcheck_mode": pipeline.settings.spellcheck_mode,
        "available_models": [
            v for v in ALL_MODELS if check_model_available(v, pipeline.settings)
        ],
        "output_dir": str(base_dir) if base_dir is not None else None,
        "total_requests": _total_requests,
        "rss_ram_mb": rss_ram_mb,
        "last_run": _last_run,
        **info,
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    """List supported OCR recognizer models, disk availability, and active/loaded state."""
    pipeline = get_pipeline()
    loaded_labels = (
        set(pipeline.engine.health_info().get("loaded_models", []))
        if pipeline._engine is not None
        else set()
    )
    ctc_label = (
        "Bina-0.2-Rizeh"
        if pipeline.settings.hybrid_ctc_variant == "bina-0.2-rizeh"
        else "Bina-0.2-RizehPizeh"
    )
    vlm_label = (
        f"OpenAI-VLM ({pipeline.settings.vlm_model_name or 'unconfigured'})"
        if pipeline.settings.hybrid_vlm_variant == "openai-vlm"
        else "Bina-0.4-Koochik"
    )
    label_map = {
        "bina-0.2-rizehpizeh": "Bina-0.2-RizehPizeh",
        "bina-0.2-rizeh": "Bina-0.2-Rizeh",
        "bina-0.4-koochik": "Bina-0.4-Koochik",
        "hybrid": f"Hybrid ({ctc_label} + {vlm_label})",
    }
    models = []
    for variant in ALL_MODELS:
        label = label_map[variant]
        granularity = (
            "hybrid"
            if variant == "hybrid"
            else ("region" if variant == "bina-0.4-koochik" else "line")
        )
        is_loaded = (
            bool(loaded_labels) and pipeline.settings.rec_model_variant == "hybrid"
            if variant == "hybrid"
            else label in loaded_labels
        )
        models.append(
            {
                "variant": variant,
                "label": label,
                "granularity": granularity,
                "available_on_disk": check_model_available(variant, pipeline.settings),
                "is_active": pipeline.settings.rec_model_variant == variant,
                "is_loaded": is_loaded,
            }
        )
    return {
        "active_model": pipeline.settings.rec_model_variant,
        "default_formats": list(_default_formats),
        "valid_formats": list(VALID_FORMATS),
        "models": models,
    }


@app.post("/config")
@app.post("/models/select")
def update_config(payload: ConfigUpdateRequest) -> dict[str, Any]:
    """Set the default recognizer model, spellcheck mode, and/or default output formats."""
    global _active_model, _default_formats, _spellcheck_mode

    if payload.out_formats is not None:
        try:
            _default_formats = parse_formats(payload.out_formats, default=_default_formats)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.spellcheck_mode is not None:
        _spellcheck_mode = payload.spellcheck_mode

    if payload.model is not None:
        _active_model = payload.model
        pipeline = get_pipeline(model=payload.model, spellcheck_mode=_spellcheck_mode)
    else:
        pipeline = get_pipeline(spellcheck_mode=_spellcheck_mode)

    if payload.preload:
        _ = pipeline.engine

    return {
        "status": "ok",
        "active_model": pipeline.settings.rec_model_variant,
        "default_formats": list(_default_formats),
        "spellcheck_mode": pipeline.settings.spellcheck_mode,
        "engine_loaded": pipeline._engine is not None,
    }


@app.post("/ocr")
@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    model: RecognizerVariant | None = Query(None, description="Override recognizer model name."),
    out_format: list[str] | None = Query(
        None,
        description="Output formats (repeatable: ?out_format=html&out_format=md&out_format=txt).",
    ),
    formats: str | None = Query(
        None,
        description="Comma- or space-separated formats: json,md,html,txt.",
    ),
    spellcheck: SpellcheckMode | None = Query(
        None,
        description="Override spellcheck mode: off, lexicon, kenlm.",
    ),
    save_outputs: bool = Query(True, description="Save artifacts and manifest.json to output dir."),
) -> dict[str, Any]:
    """Accept an uploaded PDF or image, run single-GPU OCR sequentially, and return results."""
    global _total_requests, _last_run

    raw_fmt_inputs: list[str] = []
    if out_format:
        raw_fmt_inputs.extend(out_format)
    if formats:
        raw_fmt_inputs.append(formats)

    try:
        requested = parse_formats(raw_fmt_inputs or None, default=_default_formats)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = file.filename or "upload.bin"
    suffix = Path(filename).suffix or ".bin"
    tmp_path: Path | None = None

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        pipeline = get_pipeline(model=model, spellcheck_mode=spellcheck)
        doc_result, written = await anyio.to_thread.run_sync(
            lambda: pipeline.process_file(
                tmp_path,
                formats=requested,
                source_filename=filename,
                save_outputs=save_outputs,
                save_manifest=save_outputs,
            )
        )

        _total_requests += 1
        _last_run = {
            "doc_id": doc_result.doc_id,
            "source_filename": doc_result.source_filename,
            "model": pipeline.settings.rec_model_variant,
            "formats": requested,
            "page_count": doc_result.page_count,
            "processing_time_sec": doc_result.processing_time_sec,
            "peak_vram_mb": doc_result.metrics.peak_vram_mb if doc_result.metrics else None,
            "peak_rss_ram_mb": doc_result.metrics.peak_rss_ram_mb if doc_result.metrics else None,
        }

        response: dict[str, Any] = {
            "doc_id": doc_result.doc_id,
            "model": pipeline.settings.rec_model_variant,
            "model_label": doc_result.metadata.rec_model_label if doc_result.metadata else None,
            "formats": requested,
            "result": doc_result.model_dump(),
            "saved_files": {k: str(v) for k, v in written.items()},
            "observability": {
                "metadata": doc_result.metadata.model_dump() if doc_result.metadata else None,
                "metrics": doc_result.metrics.model_dump() if doc_result.metrics else None,
            },
        }
        if "md" in requested:
            response["markdown"] = to_markdown(doc_result)
        if "html" in requested:
            response["html"] = to_html(doc_result)
        if "txt" in requested:
            response["text"] = to_text(doc_result)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    """List saved OCR run manifests newest-first."""
    pipeline = get_pipeline()
    runs = OutputManager(pipeline.storage).list_manifests(limit=limit)
    return {"count": len(runs), "runs": runs}


@app.get("/outputs/{file_path:path}")
def download_output(file_path: str) -> FileResponse:
    """Serve a saved output artifact safely from the API output directory."""
    pipeline = get_pipeline()
    try:
        target = pipeline.storage.resolve_path(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Output file not found: {file_path}")
    media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(path=target, media_type=media_type, filename=target.name)

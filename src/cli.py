"""Typer CLI for running the Persian OCR pipeline locally."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, cast

import typer

from config.settings import RecognizerVariant, Settings, SpellcheckMode, configure_cache_env
from src.ingestion.pdf_loader import SUPPORTED_IMAGE_EXTENSIONS
from src.pipeline import DocumentPipeline
from src.postprocess.serializers import parse_formats
from src.storage.local import LocalStorageBackend

ALL_MODELS: tuple[RecognizerVariant, ...] = (
    "bina-0.2-rizehpizeh",
    "bina-0.2-rizeh",
    "bina-0.4-koochik",
    "hybrid",
)

app = typer.Typer(help="Local-first Persian (RTL) OCR pipeline CLI.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Persian OCR pipeline commands."""


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _collect_inputs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        allowed = {".pdf", *SUPPORTED_IMAGE_EXTENSIONS}
        files = sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in allowed
        )
        return files
    raise typer.BadParameter(f"Input path does not exist: {target}")


def _resolve_models(model_arg: str | None, default_variant: RecognizerVariant) -> list[RecognizerVariant]:
    if model_arg is None:
        return [default_variant]
    cleaned = model_arg.strip().lower()
    if cleaned == "all":
        return list(ALL_MODELS)
    if cleaned not in ALL_MODELS:
        raise typer.BadParameter(
            f"Invalid model {model_arg!r}. Choose from {list(ALL_MODELS)} or 'all'."
        )
    return [cast(RecognizerVariant, cleaned)]


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": False})
def run(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., help="Input PDF, image file, or directory of files."),
    output_dir: Path = typer.Option(Path("./out"), "--output-dir", "-o", help="Root directory for outputs."),
    out_format: list[str] | None = typer.Option(
        None,
        "--out-format",
        "--formats",
        "-f",
        help="Output formats (html, md, txt, json). Repeatable or comma/space-separated.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "--rec-model",
        "-m",
        help="Recognizer model: bina-0.2-rizehpizeh, bina-0.2-rizeh, bina-0.4-koochik, hybrid, or all.",
    ),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Recognition batch size."),
    device: Literal["gpu", "cpu"] | None = typer.Option(None, "--device", help="Override device (gpu/cpu)."),
    dpi: int | None = typer.Option(None, "--dpi", help="PDF rasterization DPI."),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table structure recognition."),
    no_manifest: bool = typer.Option(False, "--no-manifest", help="Skip saving manifest.json alongside outputs."),
    spellcheck: SpellcheckMode | None = typer.Option(
        None,
        "--spellcheck",
        help="Persian OCR spellcheck mode: off, lexicon, or kenlm.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run Persian OCR on a file or directory and save structured labeled outputs."""
    _configure_logging(verbose)

    raw_formats: list[str] = list(out_format or [])
    if ctx.args:
        for extra in ctx.args:
            try:
                parse_formats(extra)
            except ValueError as exc:
                raise typer.BadParameter(f"Unexpected argument or unsupported format: {extra!r}") from exc
            raw_formats.append(extra)

    try:
        fmt_list = parse_formats(raw_formats or None, default=("json", "md", "html"))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    base_settings = Settings()
    models_to_run = _resolve_models(model, base_settings.rec_model_variant)

    files = _collect_inputs(input_path)
    if not files:
        typer.echo(f"No supported PDF/image files found in {input_path}")
        raise typer.Exit(code=1)

    storage = LocalStorageBackend(output_dir)

    for m_idx, variant in enumerate(models_to_run, start=1):
        settings = base_settings.model_copy(update={"rec_model_variant": variant})
        if batch_size is not None:
            settings = settings.model_copy(update={"rec_batch_size": batch_size})
        if device is not None:
            settings = settings.model_copy(update={"device": device})
        if dpi is not None:
            settings = settings.model_copy(update={"dpi": dpi})
        if no_tables:
            settings = settings.model_copy(update={"enable_tables": False})
        if spellcheck is not None:
            settings = settings.model_copy(update={"spellcheck_mode": spellcheck})
        configure_cache_env(settings.models_dir)

        typer.echo(
            f"=== Model [{m_idx}/{len(models_to_run)}]: {variant} "
            f"(formats={','.join(fmt_list)}, spellcheck={settings.spellcheck_mode}) ==="
        )
        pipeline = DocumentPipeline(settings=settings, storage=storage)
        try:
            for idx, file_path in enumerate(files, start=1):
                typer.echo(f"[{idx}/{len(files)}] Processing {file_path} with {variant}...")
                doc_result, written = pipeline.process_file(
                    file_path,
                    formats=fmt_list,
                    save_manifest=not no_manifest,
                )
                for page in doc_result.pages:
                    metrics = page.metrics
                    vram_str = (
                        f"{metrics.peak_vram_mb:.1f} MB"
                        if metrics and metrics.peak_vram_mb is not None
                        else "n/a"
                    )
                    ram_str = (
                        f"{metrics.rss_ram_mb:.1f} MB"
                        if metrics and metrics.rss_ram_mb is not None
                        else "n/a"
                    )
                    lat_str = f"{metrics.latency_sec:.2f}s" if metrics else "n/a"
                    stage_str = ""
                    corr_str = ""
                    if metrics is not None:
                        st = metrics.stages
                        spell_part = (
                            f" spell={st.spellcheck_sec:.2f}s" if st.spellcheck_sec > 0 else ""
                        )
                        stage_str = (
                            f" [pre={st.preprocess_sec:.2f}s layout={st.layout_sec:.2f}s "
                            f"det={st.detect_sec:.2f}s table={st.table_sec:.2f}s "
                            f"rec={st.recognize_sec:.2f}s post={st.postprocess_sec:.2f}s{spell_part}]"
                        )
                        if metrics.corrections_count > 0:
                            corr_str = f", corr={metrics.corrections_count}"
                    typer.echo(
                        f"  Page {page.page_number}: blocks={len(page.blocks)}, "
                        f"latency={lat_str}{stage_str}{corr_str}, peak_vram={vram_str}, rss_ram={ram_str}"
                    )
                typer.echo(
                    f"  Done in {doc_result.processing_time_sec:.2f}s -> "
                    + ", ".join(f"{k}:{v}" for k, v in written.items())
                )
        finally:
            if len(models_to_run) > 1:
                pipeline.unload()


if __name__ == "__main__":
    app()

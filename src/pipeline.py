"""End-to-end local-first Persian OCR document pipeline orchestrator."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import psutil

from config.settings import Settings, configure_cache_env, get_settings
from src.ingestion.pdf_loader import load_document_pages
from src.ingestion.preprocessor import preprocess_page
from src.postprocess.normalizer import normalize_persian
from src.postprocess.rtl_order import sort_blocks_rtl
from src.postprocess.serializers import VALID_FORMATS, _cells_to_html_table, parse_formats
from src.postprocess.spellcheck import get_spellchecker
from src.schemas.document import (
    DocumentMetrics,
    DocumentResult,
    LayoutBlock,
    PageMetrics,
    PageResult,
    RunMetadata,
    SpellcheckCorrection,
    StageTimings,
    TableCell,
    TextLine,
)
from src.storage.base import StorageBackend
from src.storage.local import LocalStorageBackend
from src.storage.outputs import OutputManager

if TYPE_CHECKING:
    from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

VALID_FORMATS_SET = set(VALID_FORMATS)


class DocumentPipeline:
    """Single-process sequential document OCR pipeline with pluggable storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        storage: StorageBackend | None = None,
        engine: InferenceEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_cache_env(self.settings.models_dir)
        self.storage: StorageBackend = storage or LocalStorageBackend(self.settings.data_dir)
        self._engine = engine
        self.last_page_images: dict[str, str] = {}

    @property
    def engine(self) -> InferenceEngine:
        """Lazily initialize InferenceEngine so non-inference tasks stay lightweight."""
        if self._engine is None:
            from src.inference.engine import InferenceEngine

            self._engine = InferenceEngine(self.settings)
        return self._engine

    def unload(self) -> None:
        """Release the underlying inference engine and clear GPU caches."""
        if self._engine is not None and hasattr(self._engine, "unload"):
            self._engine.unload()
        self._engine = None

    def _postprocess_page(
        self,
        page: PageResult,
    ) -> tuple[PageResult, float, list[SpellcheckCorrection]]:
        """Apply RTL sorting, Persian Unicode normalization, and optional OCR spellcheck."""
        ordered_blocks = sort_blocks_rtl(page.blocks, page_width=page.width)
        normalized_blocks: list[LayoutBlock] = []
        policy = self.settings.digit_policy
        spellchecker = get_spellchecker(
            self.settings.spellcheck_mode,
            custom_dict=self.settings.spellcheck_custom_dict,
            kenlm_path=self.settings.spellcheck_kenlm_path,
            real_word_conf_threshold=self.settings.spellcheck_real_word_conf_threshold,
        )
        page_corrections: list[SpellcheckCorrection] = []
        spellcheck_sec = 0.0

        for block in ordered_blocks:
            norm_lines = [
                line.model_copy(update={"text": normalize_persian(line.text, digit_policy=policy)})
                for line in block.lines
            ]
            norm_cells = [
                TableCell(
                    **{
                        **cell.model_dump(exclude={"text"}),
                        "text": normalize_persian(cell.text, digit_policy=policy),
                    }
                )
                for cell in block.cells
            ]

            if spellchecker is not None:
                t_sp0 = time.perf_counter()
                if norm_lines and not norm_cells:
                    corrected_lines: list[TextLine] = []
                    for ln in norm_lines:
                        fixed_txt, corrs = spellchecker.correct_text(
                            ln.text,
                            page_number=page.page_number,
                            line_confidence=ln.confidence,
                        )
                        page_corrections.extend(corrs)
                        corrected_lines.append(ln.model_copy(update={"text": fixed_txt}))
                    norm_lines = corrected_lines

                if norm_cells:
                    corrected_cells: list[TableCell] = []
                    for cell in norm_cells:
                        fixed_txt, corrs = spellchecker.correct_text(
                            cell.text,
                            page_number=page.page_number,
                            line_confidence=cell.confidence,
                        )
                        page_corrections.extend(corrs)
                        corrected_cells.append(cell.model_copy(update={"text": fixed_txt}))
                    norm_cells = corrected_cells
                    # Also keep lines consistent for table blocks when lines were derived from cells
                    if norm_lines:
                        corrected_lines = []
                        for ln in norm_lines:
                            fixed_txt, _ = spellchecker.correct_text(
                                ln.text,
                                page_number=page.page_number,
                                line_confidence=ln.confidence,
                            )
                            corrected_lines.append(ln.model_copy(update={"text": fixed_txt}))
                        norm_lines = corrected_lines
                spellcheck_sec += time.perf_counter() - t_sp0

            if norm_lines:
                block_text = "\n".join(ln.text for ln in norm_lines if ln.text)
            else:
                block_text = normalize_persian(block.text, digit_policy=policy)
                if spellchecker is not None and block_text:
                    t_sp0 = time.perf_counter()
                    block_text, corrs = spellchecker.correct_text(
                        block_text, page_number=page.page_number
                    )
                    page_corrections.extend(corrs)
                    spellcheck_sec += time.perf_counter() - t_sp0

            updated_html = _cells_to_html_table(norm_cells) if norm_cells else block.html

            normalized_blocks.append(
                block.model_copy(
                    update={
                        "lines": norm_lines,
                        "cells": norm_cells,
                        "text": block_text,
                        "html": updated_html,
                    }
                )
            )

        return (
            page.model_copy(update={"blocks": normalized_blocks}),
            round(spellcheck_sec, 3),
            page_corrections,
        )

    def _build_run_metadata(self, pages: list[PageResult]) -> RunMetadata:
        """Build provenance metadata from active engine/recognizer or settings."""
        eng = self._engine
        recognizer = getattr(eng, "recognizer", None) if eng is not None else None

        variant = self.settings.rec_model_variant
        ctc_label = (
            "Bina-0.2-Rizeh"
            if self.settings.hybrid_ctc_variant == "bina-0.2-rizeh"
            else "Bina-0.2-RizehPizeh"
        )
        vlm_label = (
            f"OpenAI-VLM ({self.settings.vlm_model_name or 'unconfigured'})"
            if self.settings.hybrid_vlm_variant == "openai-vlm"
            else "Bina-0.4-Koochik"
        )
        default_labels = {
            "bina-0.2-rizehpizeh": "Bina-0.2-RizehPizeh",
            "bina-0.2-rizeh": "Bina-0.2-Rizeh",
            "bina-0.4-koochik": "Bina-0.4-Koochik",
            "hybrid": f"Hybrid ({ctc_label} + {vlm_label})",
        }
        if variant == "hybrid":
            rec_label = default_labels["hybrid"]
            granularity = "hybrid"
        else:
            rec_label = getattr(recognizer, "model_label", None) or default_labels.get(variant, variant)
            granularity = getattr(
                recognizer,
                "granularity",
                "region" if variant == "bina-0.4-koochik" else "line",
            )
        if pages and all(
            len(p.blocks) == 1 and p.blocks[0].block_id.endswith("_native") for p in pages
        ):
            granularity = "native"

        gpu_name = None
        if eng is not None and hasattr(eng, "get_gpu_name"):
            gpu_name = eng.get_gpu_name()
        elif eng is not None and hasattr(eng, "health_info"):
            gpu_name = eng.health_info().get("gpu_name")

        det_model = self.settings.det_model_name if granularity in {"line", "hybrid"} else None
        table_model = (
            self.settings.table_model_name
            if granularity in {"line", "region", "hybrid"} and self.settings.enable_tables
            else None
        )

        return RunMetadata(
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            rec_model_variant=variant,
            rec_model_label=str(rec_label),
            granularity=granularity,  # type: ignore[arg-type]
            layout_model=self.settings.layout_model_name,
            det_model=det_model,
            table_model=table_model,
            device=self.settings.device,
            gpu_name=gpu_name,
            dpi=self.settings.dpi,
            digit_policy=self.settings.digit_policy,
            spellcheck_mode=self.settings.spellcheck_mode,
        )

    @staticmethod
    def _build_document_metrics(
        pages: list[PageResult],
        total_sec: float,
        corrections: list[SpellcheckCorrection] | None = None,
    ) -> DocumentMetrics:
        """Aggregate page-level latency, memory peaks, stage timings, and content counts."""
        latencies = [p.metrics.latency_sec for p in pages if p.metrics is not None]
        vrams = [
            p.metrics.peak_vram_mb
            for p in pages
            if p.metrics is not None and p.metrics.peak_vram_mb is not None
        ]
        rams = [
            p.metrics.rss_ram_mb
            for p in pages
            if p.metrics is not None and p.metrics.rss_ram_mb is not None
        ]

        stage_totals = StageTimings(
            preprocess_sec=round(
                sum(p.metrics.stages.preprocess_sec for p in pages if p.metrics), 3
            ),
            layout_sec=round(sum(p.metrics.stages.layout_sec for p in pages if p.metrics), 3),
            detect_sec=round(sum(p.metrics.stages.detect_sec for p in pages if p.metrics), 3),
            table_sec=round(sum(p.metrics.stages.table_sec for p in pages if p.metrics), 3),
            recognize_sec=round(sum(p.metrics.stages.recognize_sec for p in pages if p.metrics), 3),
            ctc_sec=round(sum(p.metrics.stages.ctc_sec for p in pages if p.metrics), 3),
            vlm_sec=round(sum(p.metrics.stages.vlm_sec for p in pages if p.metrics), 3),
            postprocess_sec=round(
                sum(p.metrics.stages.postprocess_sec for p in pages if p.metrics), 3
            ),
            spellcheck_sec=round(
                sum(p.metrics.stages.spellcheck_sec for p in pages if p.metrics), 3
            ),
        )

        total_blocks = sum(len(p.blocks) for p in pages)
        total_lines = sum(sum(len(b.lines) for b in p.blocks) for p in pages)
        total_tables = sum(sum(1 for b in p.blocks if b.type == "table") for p in pages)
        total_chars = sum(sum(len(b.text) for b in p.blocks) for p in pages)
        total_ctc_blocks = sum(p.metrics.ctc_blocks for p in pages if p.metrics)
        total_vlm_blocks = sum(p.metrics.vlm_blocks for p in pages if p.metrics)
        all_corrs = corrections or []

        return DocumentMetrics(
            total_sec=total_sec,
            mean_page_latency_sec=round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            peak_vram_mb=max(vrams) if vrams else None,
            peak_rss_ram_mb=max(rams) if rams else None,
            total_blocks=total_blocks,
            total_lines=total_lines,
            total_tables=total_tables,
            total_chars=total_chars,
            total_corrections=len(all_corrs),
            total_ctc_blocks=total_ctc_blocks,
            total_vlm_blocks=total_vlm_blocks,
            corrections=all_corrs[:200],
            stage_totals=stage_totals,
        )

    def process_file(
        self,
        input_path: Path | str,
        formats: str | Sequence[str] = ("json",),
        output_prefix: str | None = None,
        source_filename: str | None = None,
        save_outputs: bool = True,
        save_manifest: bool = True,
        save_page_images: bool = False,
    ) -> tuple[DocumentResult, dict[str, Path]]:
        """Run end-to-end OCR on a PDF or image file and persist requested outputs."""
        src_path = Path(input_path)
        requested_formats = parse_formats(formats, default=("json",))
        effective_name = source_filename or src_path.name

        start_time = time.perf_counter()
        pages: list[PageResult] = []
        all_corrections: list[SpellcheckCorrection] = []
        doc_id = uuid.uuid4().hex[:12]
        saved_page_images: dict[str, str] = {}
        self.last_page_images = {}
        out_mgr = OutputManager(self.storage) if save_outputs else None

        for page_input in load_document_pages(src_path, dpi=self.settings.dpi):
            if (
                self.settings.use_native_pdf_text
                and page_input.has_native_text
                and page_input.native_text
            ):
                t_nat0 = time.perf_counter()
                w, h = page_input.image.size
                norm_text = normalize_persian(
                    page_input.native_text,
                    digit_policy=self.settings.digit_policy,
                )
                post_sec = round(time.perf_counter() - t_nat0, 3)
                rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
                if out_mgr is not None and save_page_images:
                    import cv2
                    import numpy as np

                    nat_bgr = cv2.cvtColor(
                        np.asarray(page_input.image.convert("RGB"), dtype=np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    saved_img = out_mgr.save_page_image(
                        source_filename=effective_name,
                        model_variant=self.settings.rec_model_variant,
                        page_number=page_input.page_number,
                        bgr_img=nat_bgr,
                        output_prefix=output_prefix,
                    )
                    del nat_bgr
                    if saved_img is not None:
                        saved_page_images[str(page_input.page_number)] = saved_img[0]
                page_res = PageResult(
                    page_number=page_input.page_number,
                    width=w,
                    height=h,
                    blocks=[
                        LayoutBlock(
                            block_id=f"p{page_input.page_number}_native",
                            type="text",
                            bbox=(0.0, 0.0, float(w), float(h)),
                            reading_order=0,
                            text=norm_text,
                            confidence=1.0,
                            lines=[
                                TextLine(
                                    text=norm_text,
                                    bbox=(0.0, 0.0, float(w), float(h)),
                                    confidence=1.0,
                                )
                            ],
                        )
                    ],
                    metrics=PageMetrics(
                        latency_sec=post_sec,
                        peak_vram_mb=None,
                        rss_ram_mb=rss_mb,
                        stages=StageTimings(postprocess_sec=post_sec),
                        num_blocks=1,
                        num_lines=1,
                        num_tables=0,
                        char_count=len(norm_text),
                    ),
                )
            else:
                t_pre0 = time.perf_counter()
                bgr_img = preprocess_page(page_input.image, self.settings)
                pre_sec = round(time.perf_counter() - t_pre0, 3)

                try:
                    raw_page = self.engine.process_page(page_input.page_number, bgr_img)
                    if out_mgr is not None and save_page_images:
                        saved_img = out_mgr.save_page_image(
                            source_filename=effective_name,
                            model_variant=self.settings.rec_model_variant,
                            page_number=page_input.page_number,
                            bgr_img=bgr_img,
                            output_prefix=output_prefix,
                        )
                        if saved_img is not None:
                            saved_page_images[str(page_input.page_number)] = saved_img[0]
                finally:
                    del bgr_img

                t_post0 = time.perf_counter()
                page_res, spell_sec, page_corrs = self._postprocess_page(raw_page)
                post_sec = round(time.perf_counter() - t_post0, 3)
                all_corrections.extend(page_corrs)

                if page_res.metrics is not None:
                    updated_stages = page_res.metrics.stages.model_copy(
                        update={
                            "preprocess_sec": pre_sec,
                            "postprocess_sec": post_sec,
                            "spellcheck_sec": spell_sec,
                        }
                    )
                    updated_metrics = page_res.metrics.model_copy(
                        update={
                            "stages": updated_stages,
                            "num_blocks": len(page_res.blocks),
                            "num_lines": sum(len(b.lines) for b in page_res.blocks),
                            "num_tables": sum(1 for b in page_res.blocks if b.type == "table"),
                            "char_count": sum(len(b.text) for b in page_res.blocks),
                            "corrections_count": len(page_corrs),
                        }
                    )
                    page_res = page_res.model_copy(update={"metrics": updated_metrics})

            pages.append(page_res)

        total_sec = round(time.perf_counter() - start_time, 3)
        metadata = self._build_run_metadata(pages)
        doc_metrics = self._build_document_metrics(
            pages, total_sec=total_sec, corrections=all_corrections
        )
        doc_result = DocumentResult(
            doc_id=doc_id,
            source_filename=effective_name,
            page_count=len(pages),
            pages=pages,
            processing_time_sec=total_sec,
            metadata=metadata,
            metrics=doc_metrics,
        )

        written: dict[str, Path] = {}
        if out_mgr is not None and requested_formats:
            written = out_mgr.save_outputs(
                doc=doc_result,
                formats=requested_formats,
                model_variant=self.settings.rec_model_variant,
                output_prefix=output_prefix,
                save_manifest=save_manifest,
                page_images=saved_page_images,
            )
        self.last_page_images = dict(saved_page_images)

        logger.info(
            "Completed doc_id=%s file=%s model=%s pages=%d in %.2fs | "
            "peak_vram=%s MB | peak_rss=%s MB | outputs=%s",
            doc_id,
            effective_name,
            metadata.rec_model_variant,
            len(pages),
            total_sec,
            f"{doc_metrics.peak_vram_mb:.1f}" if doc_metrics.peak_vram_mb is not None else "n/a",
            f"{doc_metrics.peak_rss_ram_mb:.1f}" if doc_metrics.peak_rss_ram_mb is not None else "n/a",
            {k: str(v) for k, v in written.items()},
        )

        return doc_result, written

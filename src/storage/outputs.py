"""Structured, collision-free output saving and run manifest management."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from src.postprocess.serializers import to_html, to_json, to_markdown, to_text
from src.schemas.document import DocumentResult, RunManifest
from src.storage.base import StorageBackend

_UNSAFE_SLUG_RE = re.compile(r"[^\w.\-]+", re.UNICODE)


def sanitize_slug(name: str) -> str:
    """Convert an arbitrary document stem or model name into a safe path slug."""
    cleaned = _UNSAFE_SLUG_RE.sub("_", name.strip()).strip("._-")
    return cleaned or "document"


class OutputManager:
    """Saves per-document/per-model OCR artifacts and manifests via StorageBackend."""

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    @staticmethod
    def run_subdir(source_filename: str, model_variant: str) -> str:
        """Return structured relative subdirectory: <doc_stem>/<model_variant>."""
        doc_slug = sanitize_slug(Path(source_filename).stem)
        model_slug = sanitize_slug(model_variant)
        return f"{doc_slug}/{model_slug}"

    @staticmethod
    def artifact_key(
        source_filename: str,
        model_variant: str,
        ext: str,
        output_prefix: str | None = None,
    ) -> str:
        """Build collision-free relative storage key for a serialized artifact."""
        if output_prefix:
            return f"{output_prefix}.{ext}"
        doc_slug = sanitize_slug(Path(source_filename).stem)
        model_slug = sanitize_slug(model_variant)
        return f"{doc_slug}/{model_slug}/{doc_slug}__{model_slug}.{ext}"

    @staticmethod
    def page_image_key(
        source_filename: str,
        model_variant: str,
        page_number: int,
        output_prefix: str | None = None,
    ) -> str:
        """Return collision-free relative storage key for a preprocessed page JPEG preview."""
        if output_prefix:
            return f"{output_prefix}_pages/page_{page_number:03d}.jpg"
        subdir = OutputManager.run_subdir(source_filename, model_variant)
        return f"{subdir}/pages/page_{page_number:03d}.jpg"

    def save_page_image(
        self,
        source_filename: str,
        model_variant: str,
        page_number: int,
        bgr_img: Any,
        output_prefix: str | None = None,
    ) -> tuple[str, Path] | None:
        """Encode a BGR page ndarray to JPEG and persist it for Debug UI layout visualization."""
        if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
            return None
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None
        key = self.page_image_key(
            source_filename=source_filename,
            model_variant=model_variant,
            page_number=page_number,
            output_prefix=output_prefix,
        )
        saved_path = self.storage.save_bytes(key, buf.tobytes())
        return key, saved_path

    def save_outputs(
        self,
        doc: DocumentResult,
        formats: Sequence[str],
        model_variant: str,
        output_prefix: str | None = None,
        save_manifest: bool = True,
        page_images: dict[str, str] | None = None,
    ) -> dict[str, Path]:
        """Serialize and save requested formats plus an optional run manifest."""
        serializers = {
            "json": to_json,
            "md": to_markdown,
            "html": to_html,
            "txt": to_text,
        }
        written: dict[str, Path] = {}

        for fmt in formats:
            serializer = serializers.get(fmt)
            if serializer is None:
                raise ValueError(f"Unsupported output format: {fmt!r}")
            key = self.artifact_key(
                source_filename=doc.source_filename,
                model_variant=model_variant,
                ext=fmt,
                output_prefix=output_prefix,
            )
            written[fmt] = self.storage.save_text(key, serializer(doc))

        if save_manifest:
            base_dir = getattr(self.storage, "base_dir", None)
            rel_outputs: dict[str, str] = {}
            for fmt, path in written.items():
                if base_dir is not None:
                    try:
                        rel_outputs[fmt] = str(path.resolve().relative_to(Path(base_dir).resolve()))
                        continue
                    except ValueError:
                        pass
                rel_outputs[fmt] = str(path)

            manifest = RunManifest(
                doc_id=doc.doc_id,
                source_filename=doc.source_filename,
                page_count=doc.page_count,
                metadata=doc.metadata,
                metrics=doc.metrics,
                outputs=rel_outputs,
                page_images=page_images or {},
            )
            if output_prefix:
                manifest_key = f"{output_prefix}.manifest.json"
            else:
                manifest_key = f"{self.run_subdir(doc.source_filename, model_variant)}/manifest.json"
            written["manifest"] = self.storage.save_text(
                manifest_key, manifest.model_dump_json(indent=2)
            )

        return written

    def list_manifests(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent run manifests sorted newest-first."""
        base_dir = getattr(self.storage, "base_dir", None)
        if base_dir is None:
            return []
        root = Path(base_dir)
        if not root.is_dir():
            return []

        manifest_paths = sorted(
            root.rglob("*manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[: max(0, limit)]

        results: list[dict[str, Any]] = []
        for path in manifest_paths:
            try:
                results.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return results

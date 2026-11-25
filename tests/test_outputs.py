"""Unit tests for OutputManager structured per-model output saving and manifests."""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas.document import DocumentMetrics, DocumentResult, LayoutBlock, PageResult, RunMetadata
from src.storage.local import LocalStorageBackend
from src.storage.outputs import OutputManager


def _make_doc(doc_id: str, model_variant: str, model_label: str) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        source_filename="report 2026.pdf",
        page_count=1,
        pages=[
            PageResult(
                page_number=1,
                width=800,
                height=1000,
                blocks=[
                    LayoutBlock(
                        block_id="b0",
                        type="text",
                        bbox=(0, 0, 800, 100),
                        reading_order=0,
                        text="متن نمونه فارسی",
                    )
                ],
            )
        ],
        processing_time_sec=0.42,
        metadata=RunMetadata(
            created_at="2026-11-25T12:00:00+00:00",
            rec_model_variant=model_variant,
            rec_model_label=model_label,
            granularity="line",
            layout_model="PP-DocLayout_plus-L",
            device="cpu",
        ),
        metrics=DocumentMetrics(total_sec=0.42, total_blocks=1),
    )


def test_output_manager_separates_models_and_writes_manifest(tmp_path: Path) -> None:
    storage = LocalStorageBackend(tmp_path)
    mgr = OutputManager(storage)

    doc_a = _make_doc("d1", "bina-0.2-rizehpizeh", "Bina-0.2-RizehPizeh")
    doc_b = _make_doc("d2", "bina-0.4-koochik", "Bina-0.4-Koochik")

    written_a = mgr.save_outputs(doc_a, formats=("html", "txt"), model_variant="bina-0.2-rizehpizeh")
    written_b = mgr.save_outputs(doc_b, formats=("html", "md"), model_variant="bina-0.4-koochik")

    assert written_a["html"] == (tmp_path / "report_2026" / "bina-0.2-rizehpizeh" / "report_2026__bina-0.2-rizehpizeh.html").resolve()
    assert written_a["txt"].is_file()
    assert written_a["manifest"].name == "manifest.json"

    assert written_b["html"] == (tmp_path / "report_2026" / "bina-0.4-koochik" / "report_2026__bina-0.4-koochik.html").resolve()
    assert written_b["md"].is_file()
    assert written_a["html"].exists() and written_b["html"].exists()

    manifest_data = json.loads(written_a["manifest"].read_text(encoding="utf-8"))
    assert manifest_data["doc_id"] == "d1"
    assert manifest_data["metadata"]["rec_model_variant"] == "bina-0.2-rizehpizeh"
    assert "html" in manifest_data["outputs"]

    manifests = mgr.list_manifests(limit=10)
    assert len(manifests) == 2

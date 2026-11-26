"""FastAPI tests for /health, /models, /config, /ocr, /runs, and /outputs."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import src.api as api_mod
from config.settings import Settings
from src.pipeline import DocumentPipeline
from src.schemas.document import LayoutBlock, PageMetrics, PageResult, StageTimings, TextLine
from src.storage.local import LocalStorageBackend


class _StubEngine:
    def __init__(self, variant: str = "bina-0.2-rizehpizeh") -> None:
        self.variant = variant
        labels = {
            "bina-0.2-rizehpizeh": "Bina-0.2-RizehPizeh",
            "bina-0.2-rizeh": "Bina-0.2-Rizeh",
            "bina-0.4-koochik": "Bina-0.4-Koochik",
            "hybrid": "Hybrid (Bina-0.2-Rizeh + Bina-0.4-Koochik)",
        }
        self.model_label = labels.get(variant, variant)

    def get_gpu_name(self) -> str | None:
        return None

    def unload(self) -> None:
        return None

    def health_info(self) -> dict:
        return {
            "device": "cpu",
            "cuda_available": False,
            "gpu_name": None,
            "loaded_models": ["stub-layout", "stub-det", self.model_label],
            "vram_allocated_mb": 0.0,
            "vram_peak_mb": 0.0,
        }

    def process_page(self, page_number: int, bgr_img) -> PageResult:
        h, w = bgr_img.shape[:2]
        return PageResult(
            page_number=page_number,
            width=w,
            height=h,
            blocks=[
                LayoutBlock(
                    block_id=f"p{page_number}_b0",
                    type="text",
                    bbox=(10.0, 10.0, float(w - 10), 60.0),
                    reading_order=0,
                    text="سلام دنیا 1404",
                    confidence=0.98,
                    lines=[
                        TextLine(
                            text="سلام دنیا 1404",
                            bbox=(10.0, 10.0, float(w - 10), 60.0),
                            confidence=0.98,
                        )
                    ],
                )
            ],
            metrics=PageMetrics(
                latency_sec=0.01,
                peak_vram_mb=0.0,
                rss_ram_mb=128.0,
                stages=StageTimings(layout_sec=0.005, recognize_sec=0.005),
            ),
        )


def _make_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 120), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_ocr_endpoints(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        device="cpu",
        rec_model_variant="bina-0.2-rizehpizeh",
    )
    stub_pipeline = DocumentPipeline(
        settings=settings,
        storage=LocalStorageBackend(tmp_path / "out"),
        engine=_StubEngine(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(api_mod, "_pipeline_instance", stub_pipeline)
    monkeypatch.setattr(api_mod, "_active_model", "bina-0.2-rizehpizeh")
    monkeypatch.setattr(api_mod, "_default_formats", ["json"])

    with TestClient(api_mod.app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["active_model"] == "bina-0.2-rizehpizeh"
        assert "Bina-0.2-RizehPizeh" in health_data["loaded_models"]

        ocr_resp = client.post(
            "/ocr?formats=json,md,html,txt",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert ocr_resp.status_code == 200
        payload = ocr_resp.json()
        assert payload["result"]["source_filename"] == "test.png"
        assert payload["result"]["pages"][0]["blocks"][0]["text"] == "سلام دنیا ۱۴۰۴"
        assert "سلام دنیا ۱۴۰۴" in payload["markdown"]
        assert "سلام دنیا ۱۴۰۴" in payload["text"]
        assert 'dir="rtl"' in payload["html"]
        assert "Bina-0.2-RizehPizeh" in payload["html"]


def test_api_model_selection_config_and_output_serving(tmp_path: Path, monkeypatch) -> None:
    def _fake_pipeline_factory(settings: Settings, storage=None, engine=None):
        s = settings.model_copy(update={"data_dir": tmp_path, "device": "cpu"})
        return DocumentPipeline(
            settings=s,
            storage=LocalStorageBackend(tmp_path / "api_outputs"),
            engine=_StubEngine(variant=s.rec_model_variant),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(api_mod, "DocumentPipeline", _fake_pipeline_factory)
    monkeypatch.setattr(api_mod, "_pipeline_instance", None)
    monkeypatch.setattr(api_mod, "_active_model", "bina-0.2-rizehpizeh")
    monkeypatch.setattr(api_mod, "_default_formats", ["json"])

    with TestClient(api_mod.app) as client:
        models_resp = client.get("/models")
        assert models_resp.status_code == 200
        models_list = models_resp.json()["models"]
        assert len(models_list) == 4
        assert any(m["variant"] == "hybrid" and m["granularity"] == "hybrid" for m in models_list)

        cfg_resp = client.post(
            "/config",
            json={"model": "bina-0.2-rizeh", "out_formats": ["html", "md", "txt"]},
        )
        assert cfg_resp.status_code == 200
        assert cfg_resp.json()["active_model"] == "bina-0.2-rizeh"
        assert cfg_resp.json()["default_formats"] == ["html", "md", "txt"]

        ocr_resp = client.post(
            "/ocr?model=bina-0.4-koochik&out_format=html&out_format=md&out_format=txt",
            files={"file": ("sample_doc.png", _make_png_bytes(), "image/png")},
        )
        assert ocr_resp.status_code == 200
        data = ocr_resp.json()
        assert data["model"] == "bina-0.4-koochik"
        assert data["formats"] == ["html", "md", "txt"]
        assert "html" in data and "markdown" in data and "text" in data
        assert data["observability"]["metadata"]["rec_model_variant"] == "bina-0.4-koochik"

        saved_html = Path(data["saved_files"]["html"])
        assert saved_html.is_file()
        assert "sample_doc/bina-0.4-koochik/sample_doc__bina-0.4-koochik.html" in str(saved_html)

        runs_resp = client.get("/runs")
        assert runs_resp.status_code == 200
        assert runs_resp.json()["count"] >= 1

        rel_html = "sample_doc/bina-0.4-koochik/sample_doc__bina-0.4-koochik.html"
        dl_resp = client.get(f"/outputs/{rel_html}")
        assert dl_resp.status_code == 200
        assert "text/html" in dl_resp.headers["content-type"]
        assert "bina-0.4-koochik" in dl_resp.text

        hyb_resp = client.post(
            "/ocr?model=hybrid&out_format=html&out_format=json",
            files={"file": ("hyb_doc.png", _make_png_bytes(), "image/png")},
        )
        assert hyb_resp.status_code == 200
        hyb_data = hyb_resp.json()
        assert hyb_data["model"] == "hybrid"
        assert hyb_data["observability"]["metadata"]["granularity"] == "hybrid"

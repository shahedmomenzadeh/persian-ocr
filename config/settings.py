"""Application settings and local cache environment configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RecognizerVariant = Literal["bina-0.2-rizehpizeh", "bina-0.2-rizeh", "bina-0.4-koochik"]


class Settings(BaseSettings):
    """Runtime, path, VRAM, and post-processing settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./models/cache")
    bina_model_dir: Path = Path("./models/bina-0.2")
    bina_rizeh_model_dir: Path = Path("./models/bina-0.2-rizeh")
    bina_koochik_model_dir: Path = Path("./models/bina-0.4-koochik")

    # Ingestion & preprocessing
    dpi: int = 200
    enable_deskew: bool = True
    enable_denoise: bool = False
    enable_contrast_norm: bool = True
    use_native_pdf_text: bool = False

    # VRAM & inference limits
    device: Literal["gpu", "cpu"] = "gpu"
    det_max_side_len: int = 1280
    rec_model_variant: RecognizerVariant = "bina-0.2-rizehpizeh"
    rec_batch_size: int = 4
    rec_max_wh_ratio: float = 14.0
    rec_target_chunk_ratio: float = 12.0
    nemotron_max_new_tokens: int = 1024
    enable_tables: bool = True
    use_formula_recognition: bool = False
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    clear_cuda_cache_per_page: bool = True

    # Model identifiers
    layout_model_name: str = "PP-DocLayout_plus-L"
    det_model_name: str = "PP-OCRv6_medium_det"
    table_model_name: str = "SLANet_plus"

    # Persian post-processing
    digit_policy: Literal["persian", "latin", "preserve"] = "persian"


def configure_cache_env(models_dir: Path | str | None = None) -> dict[str, str]:
    """Point Hugging Face & PaddleOCR/PaddleX caches inside the local models directory."""
    base_models = Path(models_dir) if models_dir is not None else Path("./models")
    cache_root = (base_models / "cache").resolve()
    hf_home = cache_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    paddlex_home = cache_root / "paddlex"
    paddlehub_home = cache_root / "paddlehub"

    for directory in (hf_home, hf_hub_cache, paddlex_home, paddlehub_home):
        directory.mkdir(parents=True, exist_ok=True)

    env_defaults = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
        "PADDLE_PDX_CACHE_HOME": str(paddlex_home),
        "PADDLEX_HOME": str(paddlex_home),
        "HUB_HOME": str(paddlehub_home),
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    return env_defaults


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached default Settings instance."""
    return Settings()

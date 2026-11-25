#!/usr/bin/env python3
"""Verify CUDA availability and download Bina OCR recognizer weights."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MIN_WEIGHT_BYTES = 1_000_000  # 1 MB threshold to catch broken Git LFS pointer files


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    default_dir: Path
    kind: str  # "paddle" | "transformers"
    allow_patterns: list[str] | None
    required_files: list[str]


MODEL_SPECS: dict[str, ModelSpec] = {
    "bina-0.2-rizehpizeh": ModelSpec(
        name="bina-0.2-rizehpizeh",
        repo_id="Reza2kn/Bina-0.2-RizehPizeh",
        default_dir=Path("models/bina-0.2"),
        kind="paddle",
        allow_patterns=[
            "bina_text_recognition.py",
            "inference/*",
            "persian_arabic_bina02_dict.txt",
            "config.yml",
            "README.md",
            "LICENSE",
            "MODEL_PROVENANCE.json",
        ],
        required_files=["bina_text_recognition.py", "inference/inference.pdiparams"],
    ),
    "bina-0.2-rizeh": ModelSpec(
        name="bina-0.2-rizeh",
        repo_id="Reza2kn/Bina-0.2-Rizeh",
        default_dir=Path("models/bina-0.2-rizeh"),
        kind="paddle",
        allow_patterns=[
            "bina_text_recognition.py",
            "inference/*",
            "README.md",
            "LICENSE",
            "MODEL_PROVENANCE.json",
            "RELEASE.json",
        ],
        required_files=["bina_text_recognition.py", "inference/inference.pdiparams"],
    ),
    "bina-0.4-koochik": ModelSpec(
        name="bina-0.4-koochik",
        repo_id="Reza2kn/Bina-0.4-Koochik",
        default_dir=Path("models/bina-0.4-koochik"),
        kind="transformers",
        allow_patterns=None,
        required_files=[
            "config.json",
            "model.safetensors",
            "decoding.py",
            "hf_nemotron_parse_modeling.py",
        ],
    ),
    "persian-kenlm": ModelSpec(
        name="persian-kenlm",
        repo_id="PouyaZX4/persian-5gram-kenlm",
        default_dir=Path("models/kenlm"),
        kind="kenlm",
        allow_patterns=["persian_5gram_4bit.binary", "README.md"],
        required_files=["persian_5gram_4bit.binary"],
    ),
}


def _configure_hf_cache(models_root: Path) -> Path:
    """Set HF_HOME and HUGGINGFACE_HUB_CACHE inside the workspace models cache."""
    hf_home = (models_root.resolve() / "cache" / "huggingface").resolve()
    hf_hub = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    return hf_home


def check_cuda_runtimes() -> bool:
    """Print PaddlePaddle & PyTorch CUDA availability details."""
    paddle_ok = False
    try:
        import paddle

        cuda_compiled = paddle.device.is_compiled_with_cuda()
        print(f"PaddlePaddle version : {paddle.__version__}")
        print(f"Paddle CUDA compiled : {cuda_compiled}")
        if cuda_compiled:
            gpu_count = paddle.device.cuda.device_count()
            print(f"Paddle CUDA devices  : {gpu_count}")
            if gpu_count > 0:
                props = paddle.device.cuda.get_device_properties(0)
                total_mb = getattr(props, "total_memory", 0) / (1024 * 1024)
                name = getattr(props, "name", "unknown")
                print(f"GPU[0]               : {name} ({total_mb:.0f} MiB VRAM)")
                paddle_ok = True
    except ImportError:
        print("[WARN] paddlepaddle is not installed in the active Python environment.")
    except Exception as exc:
        print(f"[WARN] Could not query Paddle CUDA properties: {exc}")

    try:
        import torch

        print(f"PyTorch version      : {torch.__version__}")
        print(f"PyTorch CUDA avail   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU[0]       : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[INFO] torch is not installed (only required for bina-0.4-koochik).")

    return paddle_ok


def verify_model_dir(spec: ModelSpec, target_dir: Path) -> bool:
    """Verify required scripts/configs and real weight files (>1 MB) exist."""
    ok = True
    for rel in spec.required_files:
        p = target_dir / rel
        if not p.is_file():
            print(f"[FAIL] [{spec.name}] Missing {p}")
            ok = False

    weight_files = [
        p
        for p in target_dir.rglob("*")
        if p.is_file() and p.suffix in {".pdiparams", ".safetensors", ".bin", ".onnx", ".binary", ".klm"}
    ]
    if not weight_files:
        print(f"[FAIL] [{spec.name}] No weight files found under {target_dir}")
        return False

    for wf in weight_files:
        size = wf.stat().st_size
        print(f"  [{spec.name}] Weight: {wf.relative_to(target_dir)} ({size / (1024 * 1024):.2f} MiB)")
        if size < MIN_WEIGHT_BYTES:
            print(
                f"[FAIL] [{spec.name}] Weight file {wf} is too small ({size} bytes) — "
                "likely an unresolved Git LFS pointer."
            )
            ok = False

    if ok:
        print(f"[OK] {spec.name} verified at {target_dir.resolve()}")
    return ok


def download_via_hf_hub(
    spec: ModelSpec,
    target_dir: Path,
    force: bool = False,
    full: bool = False,
) -> None:
    """Download model weights using huggingface_hub.snapshot_download."""
    hf_home = _configure_hf_cache(target_dir.resolve().parent)
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={hf_home}")
    print(f"Downloading {spec.repo_id} via huggingface_hub into {target_dir}...")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(target_dir),
        allow_patterns=None if full else spec.allow_patterns,
        force_download=force,
    )


def download_via_git_lfs(spec: ModelSpec, target_dir: Path, force: bool = False) -> None:
    """Clone model repo using git + git-lfs."""
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed on PATH.")

    lfs_check = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
    if lfs_check.returncode != 0:
        raise RuntimeError(
            "git-lfs is not installed. Omit --use-git-lfs to download via huggingface_hub."
        )

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    repo_url = f"https://huggingface.co/{spec.repo_id}"
    if (target_dir / ".git").exists():
        subprocess.run(["git", "-C", str(target_dir), "lfs", "pull"], check=True)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA and download Bina OCR weights.")
    parser.add_argument(
        "--model",
        choices=[*MODEL_SPECS.keys(), "all"],
        default="bina-0.2-rizehpizeh",
        help="Which model variant to verify or download.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Override destination directory (only when selecting a single --model).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify CUDA and existing model files without downloading.",
    )
    parser.add_argument(
        "--use-git-lfs",
        action="store_true",
        help="Use git-lfs clone instead of huggingface_hub.snapshot_download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full Hugging Face repo including training checkpoints/datasets.",
    )
    args = parser.parse_args()

    cuda_ok = check_cuda_runtimes()
    if not cuda_ok:
        print("[WARN] Paddle CUDA is not ready (CPU fallback still possible).")

    selected = list(MODEL_SPECS.values()) if args.model == "all" else [MODEL_SPECS[args.model]]
    if args.target_dir is not None and len(selected) > 1:
        print("[ERROR] --target-dir cannot be used with --model all.")
        return 2

    all_ok = True
    for spec in selected:
        dest = args.target_dir or spec.default_dir
        if not args.verify_only:
            if args.use_git_lfs:
                download_via_git_lfs(spec, dest, force=args.force)
            else:
                download_via_hf_hub(spec, dest, force=args.force, full=args.full)
        if not verify_model_dir(spec, dest):
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

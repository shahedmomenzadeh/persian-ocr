#!/usr/bin/env python3
"""Benchmark Bina-0.2-RizehPizeh, Bina-0.2-Rizeh, and Bina-0.4-Koochik on 50 random samples."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable when run as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import RecognizerVariant, Settings, configure_cache_env  # noqa: E402
from src.pipeline import DocumentPipeline  # noqa: E402
from src.postprocess.normalizer import normalize_persian  # noqa: E402
from src.storage.local import LocalStorageBackend  # noqa: E402

ALL_MODELS: list[RecognizerVariant] = [
    "bina-0.2-rizehpizeh",
    "bina-0.2-rizeh",
    "bina-0.4-koochik",
]


def _query_nvidia_smi_used_mb() -> float | None:
    """Return total GPU memory used in MiB from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        first_line = out.strip().splitlines()[0].strip()
        return float(first_line)
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _load_gt_text(row: dict[str, Any]) -> str:
    """Extract normalized ground-truth text from an argilla metadata.jsonl row."""
    parts: list[str] = []
    for item in row.get("label", []):
        lbl = str(item.get("label", "")).lower()
        if "footer" in lbl:
            continue
        txt = normalize_persian(str(item.get("text", "")), digit_policy="persian")
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def _run_worker(
    variant: RecognizerVariant,
    sample_indices: list[int],
    dataset_dir: Path,
    out_file: Path,
) -> None:
    """Run all selected images for one model variant in an isolated process."""
    meta_path = dataset_dir / "metadata.jsonl"
    all_rows = [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_idx = {int(r["index"]): r for r in all_rows}

    settings = Settings(rec_model_variant=variant)
    configure_cache_env(settings.models_dir)

    t_load0 = time.perf_counter()
    pipeline = DocumentPipeline(
        settings=settings,
        storage=LocalStorageBackend(ROOT_DIR / "out" / f"bench_{variant}"),
    )
    # Force engine/model loading upfront so load_time is separated from per-image inference
    _ = pipeline.engine
    load_time_sec = round(time.perf_counter() - t_load0, 3)

    latencies: list[float] = []
    framework_vrams: list[float] = []
    rss_rams: list[float] = []
    similarities: list[float] = []
    samples_preview: list[dict[str, Any]] = []

    t_infer0 = time.perf_counter()
    for pos, idx in enumerate(sample_indices):
        row = rows_by_idx[idx]
        img_path = dataset_dir / row["image_file"]
        gt_text = _load_gt_text(row)

        doc_res, _ = pipeline.process_file(img_path, formats=("json",), output_prefix=f"sample_{idx:04d}")
        page = doc_res.pages[0]
        metrics = page.metrics
        lat = metrics.latency_sec if metrics else doc_res.processing_time_sec
        vram = metrics.peak_vram_mb if metrics and metrics.peak_vram_mb is not None else 0.0
        ram = metrics.rss_ram_mb if metrics and metrics.rss_ram_mb is not None else 0.0

        pred_text = "\n".join(
            b.text for b in page.blocks if b.text and b.type != "footer"
        )
        sim = difflib.SequenceMatcher(None, gt_text, pred_text).ratio() if gt_text else 0.0

        latencies.append(lat)
        framework_vrams.append(vram)
        rss_rams.append(ram)
        similarities.append(sim)

        if pos < 3:
            samples_preview.append(
                {
                    "index": idx,
                    "image_file": row["image_file"],
                    "latency_sec": lat,
                    "similarity": round(sim, 4),
                    "pred_preview": pred_text[:400],
                    "gt_preview": gt_text[:400],
                }
            )

        if (pos + 1) % 10 == 0 or pos == 0:
            print(
                f"[{variant}] {pos + 1}/{len(sample_indices)} | "
                f"lat={lat:.2f}s | fw_vram={vram:.1f}MB | rss={ram:.1f}MB | sim={sim*100:.1f}%",
                flush=True,
            )

    total_infer_sec = round(time.perf_counter() - t_infer0, 3)
    n = len(latencies)
    mean_lat = statistics.mean(latencies) if n else 0.0

    summary = {
        "variant": variant,
        "model_label": pipeline.engine.recognizer.model_label,
        "num_images": n,
        "load_time_sec": load_time_sec,
        "total_infer_sec": total_infer_sec,
        "mean_latency_sec": round(mean_lat, 3),
        "median_latency_sec": round(statistics.median(latencies), 3) if n else 0.0,
        "p95_latency_sec": round(_percentile(latencies, 95), 3),
        "throughput_img_per_sec": round(n / total_infer_sec, 3) if total_infer_sec > 0 else 0.0,
        "throughput_img_per_min": round((n / total_infer_sec) * 60.0, 2) if total_infer_sec > 0 else 0.0,
        "peak_framework_vram_mb": round(max(framework_vrams), 2) if framework_vrams else 0.0,
        "peak_rss_ram_mb": round(max(rss_rams), 2) if rss_rams else 0.0,
        "mean_gt_similarity_pct": round(statistics.mean(similarities) * 100.0, 2) if similarities else 0.0,
        "samples_preview": samples_preview,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_model_subprocess(
    variant: RecognizerVariant,
    sample_indices: list[int],
    dataset_dir: Path,
    worker_out: Path,
) -> dict[str, Any]:
    """Launch an isolated worker subprocess while sampling system nvidia-smi peak VRAM."""
    peak_smi = [0.0]
    stop_event = threading.Event()

    def _poll_smi() -> None:
        while not stop_event.is_set():
            val = _query_nvidia_smi_used_mb()
            if val is not None and val > peak_smi[0]:
                peak_smi[0] = val
            stop_event.wait(0.25)

    poller = threading.Thread(target=_poll_smi, daemon=True)
    poller.start()
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-model",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--indices-json",
            json.dumps(sample_indices),
            "--worker-out",
            str(worker_out),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
    finally:
        stop_event.set()
        poller.join(timeout=2.0)

    data = json.loads(worker_out.read_text(encoding="utf-8"))
    data["peak_nvidia_smi_vram_mb"] = round(peak_smi[0], 1)
    return data


def _format_markdown_report(results: list[dict[str, Any]], sample_indices: list[int], seed: int) -> str:
    lines = [
        f"# 50-Image Benchmark Report (seed={seed}, n={len(sample_indices)})",
        "",
        "| Model | Load (s) | Infer Total (s) | Mean Lat (s) | Median (s) | P95 (s) | Speed (img/s) | Speed (img/min) | FW Peak VRAM (MB) | nvidia-smi Peak (MB) | Peak RSS RAM (MB) | Mean GT Sim (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['model_label']} | {r['load_time_sec']:.2f} | {r['total_infer_sec']:.2f} | "
            f"{r['mean_latency_sec']:.3f} | {r['median_latency_sec']:.3f} | {r['p95_latency_sec']:.3f} | "
            f"{r['throughput_img_per_sec']:.2f} | {r['throughput_img_per_min']:.1f} | "
            f"{r['peak_framework_vram_mb']:.1f} | {r['peak_nvidia_smi_vram_mb']:.1f} | "
            f"{r['peak_rss_ram_mb']:.1f} | {r['mean_gt_similarity_pct']:.2f}% |"
        )
    lines.append("")
    lines.append(f"Sample indices: `{sample_indices}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all 3 Bina OCR models on random samples.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT_DIR / "data" / "samples" / "argilla_500")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "out")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    # Internal worker flags
    parser.add_argument("--worker-model", choices=ALL_MODELS, default=None)
    parser.add_argument("--indices-json", default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker_model is not None:
        assert args.indices_json is not None and args.worker_out is not None
        indices = [int(x) for x in json.loads(args.indices_json)]
        _run_worker(args.worker_model, indices, args.dataset_dir, args.worker_out)
        return 0

    meta_path = args.dataset_dir / "metadata.jsonl"
    all_rows = [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_indices = [int(r["index"]) for r in all_rows]
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(all_indices, min(args.num_samples, len(all_indices))))
    print(f"Selected {len(sample_indices)} random sample indices (seed={args.seed}).")

    results: list[dict[str, Any]] = []
    for variant in args.models:
        print(f"\n=== Running benchmark for {variant} ===")
        worker_out = args.out_dir / f"bench_worker_{variant}.json"
        res = _run_model_subprocess(variant, sample_indices, args.dataset_dir, worker_out)
        results.append(res)

    report_md = _format_markdown_report(results, sample_indices, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark_50_results.json").write_text(
        json.dumps({"seed": args.seed, "sample_indices": sample_indices, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "benchmark_50_report.md").write_text(report_md, encoding="utf-8")
    print("\n" + report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

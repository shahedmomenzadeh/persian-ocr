# Persian OCR Pipeline

Local-first, single-GPU (6 GB VRAM–safe) modular **Persian (RTL) document OCR pipeline** for scanned PDFs and images. Combines PaddleOCR layout & table structure analysis with pluggable Persian recognizers (`Bina-0.2-RizehPizeh`, `Bina-0.2-Rizeh`, `Bina-0.4-Koochik`), RTL reading-order reconstruction, Persian Unicode/ZWNJ normalization, 2-phase OCR spellchecking (`lexicon` + optional `kenlm` 5-gram rescoring), and structured HTML/Markdown/TXT/JSON outputs with full per-stage observability.

---

## Features

- **Lazy 1-page-at-a-time ingestion** (`pypdfium2` + Pillow) to bound host RAM on large PDFs, plus multi-frame TIFF and EXIF-transposed image support.
- **Persian-safe preprocessing**: gentle Hough-line deskew (`[-15°, +15°]`), optional denoising, and dot-preserving LAB-channel CLAHE contrast normalization.
- **Layout & RTL table reconstruction**: `PP-DocLayout_plus-L` region detection + `SLANet_plus` cell-grid parsing with automatic RTL column reversal and `rowspan`/`colspan` HTML/Markdown tables.
- **Pluggable recognizers**:
  - **Line-level CTC** (`Bina-0.2-RizehPizeh`, `Bina-0.2-Rizeh`) paired with `PP-OCRv6_medium_det`, dot-safe padding, RTL wide-line splitting, and width bucketing.
  - **Region-level VLM** (`Bina-0.4-Koochik` / Nemotron-3.5-Parse) paired with `SLANet_plus` cell-grid segmentation and ink-filtered cell OCR.
- **RTL reading-order sorting & Persian normalization**: band-aware right-to-left column sorting, NFC normalization, Arabic-to-Persian letter unification (`ي/ى→ی`, `ك→ک`, `ة→ه`, `ۀ→هٔ`), ZWNJ hygiene, and digit policy (`persian`, `latin`, `preserve`) that protects URLs, emails, and version numbers.
- **2-Phase Persian OCR spellchecker**:
  - **Phase 1 (`lexicon`)**: Persian visual-confusion weighted Damerau-Levenshtein + SymSpell 1-delete index (`wordfreq` + Iranian geography/admin lexicon), ZWNJ compound insertion, and adjacent split-word merging.
  - **Phase 2 (`kenlm`)**: Lazy memory-mapped 5-gram KenLM beam-search rescoring over OOV candidates and low-confidence real-word visual confusions (e.g. `ثبت استاد → ثبت اسناد`).
- **Structured, collision-free outputs & observability**: Saves per-document, per-model artifacts (`html`, `md`, `txt`, `json`) alongside `manifest.json` containing stage timings, VRAM/RAM peaks, and an audit log of spellcheck rewrites.
- **CLI & Single-Worker FastAPI service**: Switch models and output formats dynamically via CLI flags or HTTP endpoints.

---

## Supported Models

| Model Key (`--model`) | Hugging Face Repo | Role / Granularity | Local Directory |
| --- | --- | --- | --- |
| `bina-0.2-rizehpizeh` *(default)* | `Reza2kn/Bina-0.2-RizehPizeh` | Paddle CTC line recognizer | `models/bina-0.2/` |
| `bina-0.2-rizeh` | `Reza2kn/Bina-0.2-Rizeh` | Paddle CTC line recognizer | `models/bina-0.2-rizeh/` |
| `bina-0.4-koochik` | `Reza2kn/Bina-0.4-Koochik` | Transformers (`Nemotron-3.5-Parse`) region VLM | `models/bina-0.4-koochik/` |
| `persian-kenlm` *(optional)* | `PouyaZX4/persian-5gram-kenlm` | 4-bit quantized 5-gram Persian KenLM (`persian_5gram_4bit.binary`) | `models/kenlm/` |

PaddleOCR layout, text detection, and table structure models (`PP-DocLayout_plus-L`, `PP-OCRv6_medium_det`, `SLANet_plus`) are cached automatically under `models/cache/paddlex/` on first use.

---

## Pipeline Architecture

```text
PDF / Image
   │
   ▼  src/ingestion/pdf_loader.py & preprocessor.py
[1-Page Lazy Rasterization] ─▶ [Deskew + Mild LAB CLAHE]
   │
   ▼  src/inference/layout.py & engine.py
[PP-DocLayout_plus-L]
   ├── Line Mode (Bina-0.2-*): PP-OCRv6_medium_det + SLANet_plus ─▶ Batched Line CTC
   └── Region Mode (Bina-0.4-Koochik): SLANet_plus cell crops (ink-filtered) + Region VLM
   │
   ▼  src/postprocess/ (rtl_order.py, normalizer.py, spellcheck.py)
[RTL Column/Line Sort] ─▶ [Persian NFC + ZWNJ + Digit Policy] ─▶ [Spellcheck: off | lexicon | kenlm]
   │
   ▼  src/postprocess/serializers.py & src/storage/outputs.py
<out_dir>/<doc_stem>/<model_variant>/<doc_stem>__<model_variant>.{html,md,txt,json} + manifest.json
```

### Repository Structure

```text
persian-OCR/
├── config/
│   └── settings.py            # Pydantic settings & local HF/Paddle cache env setup
├── scripts/
│   ├── download_models.py     # CUDA check + Hugging Face model downloader & verifier
│   └── benchmark_models.py    # Subprocess-isolated 3-model latency/VRAM/RAM benchmark
├── src/
│   ├── api.py                 # FastAPI service (/health, /models, /config, /ocr, /runs, /outputs)
│   ├── cli.py                 # Typer CLI (persian-ocr run)
│   ├── pipeline.py            # DocumentPipeline orchestrator + observability aggregation
│   ├── ingestion/             # Lazy PDF rasterizer & Persian-safe image preprocessor
│   ├── inference/             # Layout, text detection, pluggable recognizers, GPU engine
│   ├── postprocess/           # RTL reading-order sorter, Persian normalizer, spellchecker, serializers
│   ├── schemas/               # Canonical DocumentResult, PageMetrics, StageTimings, RunManifest
│   └── storage/               # StorageBackend protocol, LocalStorageBackend, OutputManager
├── tests/                     # Pytest unit & integration suites
├── .env.example               # Default environment configuration
└── pyproject.toml             # Core & optional ([nemotron], [kenlm], [dev]) dependencies
```

---

## Setup & Installation

### 1. Prerequisites
- **Python**: `3.10` or `3.11` (`>=3.10,<3.12` required by `pyproject.toml`; `3.11` recommended)
- **GPU (optional, recommended)**: NVIDIA GPU with CUDA 12.x drivers (`>=6 GB` VRAM recommended; CPU fallback supported via `--device cpu` / `DEVICE=cpu`)
- **System build tools** *(only if installing `[kenlm]`, as `kenlm` compiles from source)*: `g++`, `cmake`, `make`, `zlib1g-dev`
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`

### 2. Create Virtual Environment & Install Dependencies

```bash
git clone https://github.com/shahedmomenzadeh/persian-ocr.git
cd persian-ocr

# Create a Python 3.11 virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate
```

Install **PaddlePaddle** first (GPU CUDA 12.6 build or CPU build):

```bash
# For NVIDIA GPU (CUDA 12.6):
uv pip install "paddlepaddle-gpu>=3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Or for CPU-only environments:
# uv pip install "paddlepaddle>=3.3.0"
```

Then install the project and desired optional extras (keep `numpy<2.3` for Paddle compatibility):

```bash
# Core pipeline (Bina-0.2 CTC models + CLI + FastAPI + Phase 1 lexicon spellcheck)
uv pip install -e ".[dev]" "numpy>=1.26.0,<2.3"

# Optional: add Bina-0.4-Koochik (Transformers/PyTorch) and/or Phase 2 KenLM rescoring
uv pip install -e ".[nemotron,kenlm,dev]" "numpy>=1.26.0,<2.3"
```

### 3. Configure Environment & Download Models

```bash
cp .env.example .env
```

Download and verify the model weights (all weights are stored locally under `models/`):

```bash
# Recommended default CTC model (Bina-0.2-RizehPizeh, ~60 MB)
python scripts/download_models.py --model bina-0.2-rizehpizeh

# Bina-0.2-Rizeh (~60 MB)
python scripts/download_models.py --model bina-0.2-rizeh

# Optional: Bina-0.4-Koochik (~2.41 GB)
python scripts/download_models.py --model bina-0.4-koochik

# Optional: Persian 5-gram KenLM (~744 MB, for --spellcheck kenlm)
python scripts/download_models.py --model persian-kenlm

# Or download all 4 at once:
python scripts/download_models.py --model all

# Verify existing weights and CUDA runtimes without downloading:
python scripts/download_models.py --model all --verify-only
```

---

## CLI Usage

You can run the CLI via `.venv/bin/python -m src.cli run ...` or the installed `persian-ocr run ...` entrypoint.

```bash
# Process a PDF with Bina-0.2-Rizeh and generate HTML, Markdown, and TXT
python -m src.cli run data/test_5page.pdf \
  --model bina-0.2-rizeh \
  --out-format html md txt \
  -o out

# Enable Phase 2 KenLM contextual spellcheck
python -m src.cli run data/test_5page.pdf \
  --model bina-0.2-rizeh \
  --spellcheck kenlm \
  --out-format html md txt json \
  -o out

# Run ALL 3 OCR models sequentially on the same document (auto-unloads GPU between models)
python -m src.cli run data/test_5page.pdf \
  --model all \
  --out-format html md txt json \
  -o out

# Run on a folder of images using CPU and custom DPI
python -m src.cli run ./my_scans/ \
  --model bina-0.2-rizehpizeh \
  --device cpu \
  --dpi 250 \
  --spellcheck lexicon \
  -o out
```

### CLI Options (`persian-ocr run`)

| Option | Description | Default |
| --- | --- | --- |
| `--model`, `--rec-model`, `-m` | `bina-0.2-rizehpizeh`, `bina-0.2-rizeh`, `bina-0.4-koochik`, or `all` | `Settings.rec_model_variant` |
| `--out-format`, `--formats`, `-f` | Repeatable, comma-separated, or space-separated: `html`, `md`, `txt`, `json` | `json,md,html` |
| `--output-dir`, `-o` | Root directory for structured outputs | `./out` |
| `--spellcheck` | `off`, `lexicon`, or `kenlm` | `Settings.spellcheck_mode` (`lexicon`) |
| `--device` | `gpu` or `cpu` | `Settings.device` (`gpu`) |
| `--dpi` | PDF rasterization DPI | `200` |
| `--batch-size` | CTC line recognition batch size | `4` |
| `--no-tables` | Disable `SLANet_plus` table structure recognition | `false` |
| `--no-manifest` | Skip saving `manifest.json` | `false` |
| `--verbose`, `-v` | Enable debug logging | `false` |

---

## FastAPI Service

Start the service in **single-worker mode** (safe for a single local GPU):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Endpoints

| Method & Path | Purpose |
| --- | --- |
| `GET /health` | Active model, default formats, spellcheck mode, GPU/VRAM/RSS telemetry, request count, `last_run` |
| `GET /models` | List all 3 OCR models, granularity, disk availability, and active/loaded state |
| `POST /config` *(alias `/models/select`)* | Switch default `model`, `out_formats`, `spellcheck_mode`, and optionally `preload` weights |
| `POST /ocr` *(alias `/process`)* | Upload a PDF/image, optionally override `?model=`, `?out_format=`, `?spellcheck=`, and save/return outputs |
| `GET /runs` | List recent saved `manifest.json` run records newest-first |
| `GET /outputs/{file_path}` | Download a saved artifact (`.html`, `.md`, `.txt`, `.json`) from `api_outputs` |

### `curl` Examples

```bash
# 1. Check health and available models
curl http://localhost:8000/health
curl http://localhost:8000/models

# 2. Set default model, output formats, and spellcheck mode
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"model": "bina-0.2-rizeh", "out_formats": ["html", "md", "txt"], "spellcheck_mode": "kenlm"}'

# 3. Process a document (with per-request overrides)
curl -X POST "http://localhost:8000/ocr?model=bina-0.2-rizeh&out_format=html&out_format=md&out_format=txt&spellcheck=kenlm" \
  -F "file=@data/test_5page.pdf"

# 4. List recent runs and download a generated HTML file
curl http://localhost:8000/runs
curl -O http://localhost:8000/outputs/test_5page/bina-0.2-rizeh/test_5page__bina-0.2-rizeh.html
```

---

## Output Layout & Observability

Artifacts from different models never overwrite each other. Every run writes to:

```text
<output_dir>/
└── <doc_stem>/
    └── <model_variant>/
        ├── <doc_stem>__<model_variant>.html
        ├── <doc_stem>__<model_variant>.md
        ├── <doc_stem>__<model_variant>.txt
        ├── <doc_stem>__<model_variant>.json
        └── manifest.json
```

Each output embeds provenance and performance telemetry:
- **HTML**: `<meta name="ocr-*">` tags, model name in `<title>`, an RTL `<header class="run-meta">` provenance banner, and per-page `<div class="page-meta">` bars showing latency, stage timings (`pre`, `layout`, `det`, `table`, `rec`, `post`, `spell`), VRAM, RSS RAM, block count, and correction count.
- **Markdown**: YAML frontmatter with `doc_id`, `model`, `model_variant`, `spellcheck_mode`, `total_corrections`, `processing_time_sec`, `peak_vram_mb`, and `peak_rss_ram_mb`.
- **JSON & `manifest.json`**: `RunMetadata`, `DocumentMetrics`, per-page `StageTimings`, and `metrics.corrections` listing every spellcheck rewrite (`page_number`, `original`, `corrected`, `cost`, `rule`).

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `gpu` | `gpu` or `cpu` |
| `REC_MODEL_VARIANT` | `bina-0.2-rizehpizeh` | `bina-0.2-rizehpizeh`, `bina-0.2-rizeh`, or `bina-0.4-koochik` |
| `DPI` | `200` | PDF rasterization DPI |
| `DET_MAX_SIDE_LEN` | `1280` | Max image side length for `PP-OCRv6_medium_det` |
| `REC_BATCH_SIZE` | `4` | Batch size for Bina-0.2 line recognition |
| `ENABLE_TABLES` | `true` | Enable `SLANet_plus` table structure recognition |
| `CLEAR_CUDA_CACHE_PER_PAGE` | `true` | Empty Paddle/PyTorch CUDA caches after each page |
| `DIGIT_POLICY` | `persian` | `persian`, `latin`, or `preserve` |
| `SPELLCHECK_MODE` | `lexicon` | `off`, `lexicon` (Phase 1), or `kenlm` (Phase 2) |
| `SPELLCHECK_KENLM_PATH` | `./models/kenlm/persian_5gram_4bit.binary` | Path to KenLM binary for `kenlm` mode |
| `SPELLCHECK_REAL_WORD_CONF_THRESHOLD` | `0.94` | Line confidence below which KenLM considers real-word visual swaps |
| `SPELLCHECK_CUSTOM_DICT` | `None` | Optional custom vocabulary file (`word` or `word<tab>freq` per line) |

---

## Benchmarking

Run the subprocess-isolated benchmark across all 3 models (sampling `data/samples/argilla_500`):

```bash
python scripts/benchmark_models.py --num-samples 50 --seed 42
```

Measured results on **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)**, 50 random samples (`seed=42`):

| Model | Load (s) | Infer Total (s) | Mean Lat (s) | Median (s) | P95 (s) | Speed (img/min) | FW Peak VRAM | `nvidia-smi` Peak | Peak RSS RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Bina-0.2-RizehPizeh** | 71.45 | **66.49** | **1.089** | **0.850** | **1.999** | **45.1** | **505.4 MB** | **1556.0 MB** | 2713.9 MB |
| **Bina-0.2-Rizeh** | **65.41** | 98.85 | 1.738 | 1.463 | 3.348 | 30.4 | 507.3 MB | **1556.0 MB** | **2454.8 MB** |
| **Bina-0.4-Koochik** | 140.80 | 626.36 | 12.222 | 11.968 | 21.464 | 4.8 | 1785.3 MB | 2980.0 MB | 3499.6 MB |

---

## Running Tests

```bash
.venv/bin/pytest -q
```

---

## Model Licenses & Attribution

- `Reza2kn/Bina-0.2-RizehPizeh` & `Reza2kn/Bina-0.2-Rizeh` — PaddleOCR-based Persian CTC models.
- `Reza2kn/Bina-0.4-Koochik` — Fine-tune of `nvidia/Nemotron-3.5-Parse` (OpenMDW 1.1).
- `PouyaZX4/persian-5gram-kenlm` — Apache 2.0 Persian 5-gram KenLM model.
- `PaddleOCR` / `PaddleX` (`PP-DocLayout_plus-L`, `PP-OCRv6_medium_det`, `SLANet_plus`) — Apache 2.0.

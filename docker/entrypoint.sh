#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/models /app/data

# Add bundled NVIDIA CUDA/cuDNN wheel library directories to LD_LIBRARY_PATH when present.
nvidia_libs=""
for lib_dir in /usr/local/lib/python3.11/site-packages/nvidia/*/lib; do
  if [ -d "$lib_dir" ]; then
    nvidia_libs="${nvidia_libs:+${nvidia_libs}:}${lib_dir}"
  fi
done
if [ -n "$nvidia_libs" ]; then
  export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

ensure_model() {
  local model_key="$1"
  local required_rel="$2"
  if [ ! -f "/app/models/${required_rel}" ]; then
    echo "[entrypoint] Model '${model_key}' missing at /app/models/${required_rel}; downloading..."
    python scripts/download_models.py --model "${model_key}"
  fi
}

ensure_by_key() {
  local key="$1"
  case "$key" in
    bina-0.2-rizeh)
      ensure_model "bina-0.2-rizeh" "bina-0.2-rizeh/inference/inference.pdiparams"
      ;;
    bina-0.2-rizehpizeh)
      ensure_model "bina-0.2-rizehpizeh" "bina-0.2/inference/inference.pdiparams"
      ;;
    bina-0.4-koochik)
      ensure_model "bina-0.4-koochik" "bina-0.4-koochik/model.safetensors"
      ;;
    persian-kenlm)
      ensure_model "persian-kenlm" "kenlm/persian_5gram_4bit.binary"
      ;;
    "")
      ;;
    *)
      echo "[entrypoint] WARN: Unknown DOWNLOAD_MODELS entry '${key}', skipping."
      ;;
  esac
}

auto_dl="${AUTO_DOWNLOAD_MODELS:-true}"
case "${auto_dl,,}" in
  true|1|yes)
    if [ -n "${DOWNLOAD_MODELS:-}" ]; then
      if [ "${DOWNLOAD_MODELS,,}" = "all" ]; then
        ensure_by_key "bina-0.2-rizeh"
        ensure_by_key "bina-0.2-rizehpizeh"
        ensure_by_key "bina-0.4-koochik"
        ensure_by_key "persian-kenlm"
      else
        IFS=', ' read -ra requested_models <<< "${DOWNLOAD_MODELS}"
        for m in "${requested_models[@]}"; do
          ensure_by_key "$m"
        done
      fi
    else
      rec_variant="${REC_MODEL_VARIANT:-bina-0.2-rizeh}"
      if [ "$rec_variant" = "hybrid" ]; then
        ensure_by_key "${HYBRID_CTC_VARIANT:-bina-0.2-rizeh}"
        if [ "${HYBRID_VLM_VARIANT:-bina-0.4-koochik}" = "bina-0.4-koochik" ]; then
          ensure_by_key "bina-0.4-koochik"
        fi
      else
        ensure_by_key "$rec_variant"
      fi

      if [ "${SPELLCHECK_MODE:-kenlm}" = "kenlm" ]; then
        ensure_by_key "persian-kenlm"
      fi
    fi
    ;;
  *)
    echo "[entrypoint] AUTO_DOWNLOAD_MODELS=${auto_dl}; skipping automatic model download."
    ;;
esac

exec "$@"

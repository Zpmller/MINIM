#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
export MINIM_FEATURE_CACHE_DIR="$PWD/runs/minim/feature_cache"
PY=.venv/bin/python
BATCH_SIZE="${BATCH_SIZE:-16}"
mkdir -p logs runs/minim
if [[ ! -s "$MINIM_EMBEDDER_PATH/model.safetensors" ]]; then
  scripts/download_embedder.sh 2>&1 | tee logs/embedder_download.log
fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 500 > logs/gpu_mem_full.csv &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" train_minim_scorer.py \
  --data_root data --output_dir runs/minim --model_type gatv2 --device cuda \
  --seed 0 --train_frac 0.8 --val_frac 0.1 --epochs 10 --batch_size "$BATCH_SIZE" \
  --lr 3e-4 --weight_decay 1e-5 --gat_hidden 496 --gat_heads 4 \
  --alpha 1.0 --ci_lambda 1.0 --sens_high_boost 0.0 --nec_low_boost 0.0 --nec_high_boost 0.0 \
  2>&1 | tee logs/train_full.log
.venv/bin/python scripts/plot_training.py logs/train_full.log

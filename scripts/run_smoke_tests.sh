#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
PY=.venv/bin/python
mkdir -p logs runs
if [[ ! -s "$MINIM_EMBEDDER_PATH/model.safetensors" ]]; then
  scripts/download_embedder.sh 2>&1 | tee logs/embedder_download.log
fi
"$PY" scripts/prepare_smoke_data.py | tee logs/smoke_prepare.log
OMP_NUM_THREADS=2 "$PY" train_minim_scorer.py --data_root data_smoke --output_dir runs/smoke_mlp \
  --model_type mlp --device cuda --epochs 1 --batch_size 4 --seed 0 2>&1 | tee logs/smoke_mlp.log
for bs in 4 8 16; do
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 250 > "logs/gpu_mem_bs${bs}.csv" &
  sampler=$!
  if OMP_NUM_THREADS=2 "$PY" train_minim_scorer.py --data_root data_smoke --output_dir "runs/smoke_gat_bs${bs}" \
    --model_type gatv2 --device cuda --epochs 1 --batch_size "$bs" --seed 0 2>&1 | tee "logs/smoke_gat_bs${bs}.log"; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    echo "batch_size=${bs}: PASS" | tee -a logs/batch_sweep.log
  else
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    echo "batch_size=${bs}: FAIL (see logs/smoke_gat_bs${bs}.log)" | tee -a logs/batch_sweep.log
  fi
done
.venv/bin/python scripts/verify_smoke_outputs.py | tee logs/smoke_validation.log
for model in mlp gat_bs4 gat_bs8 gat_bs16; do
  [[ -f "runs/smoke_${model}/model.pt" ]] || { echo "Skipping inference for ${model}: no checkpoint"; continue; }
  "$PY" run_minim_inference.py --data_root data_smoke --split_path "runs/smoke_${model}/split.json" \
    --checkpoint "runs/smoke_${model}/model.pt" --output_dir "runs/smoke_${model}" \
    --model_type "$(case "$model" in mlp) echo mlp;; *) echo gatv2;; esac)" --predict_split test \
    --device cuda --max_variants 3 2>&1 | tee "logs/smoke_infer_${model}.log"
done

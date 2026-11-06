#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
export MINIM_FEATURE_CACHE_DIR="$PWD/runs/minim/feature_cache"
PY=.venv/bin/python
COMMON=(--data_root data --split_path runs/minim/split.json --checkpoint runs/minim/model.pt \
  --model_type gatv2 --device cuda --seed 0)
mkdir -p logs results
if [[ ! -s "$MINIM_EMBEDDER_PATH/model.safetensors" ]]; then
  scripts/download_embedder.sh 2>&1 | tee logs/embedder_download.log
fi
for entry in \
  "table1:experiment_table1_main_results.py" \
  "table3:experiment_table3_task_semantics.py" \
  "table4:experiment_table4_abstraction_ablation.py" \
  "table10:experiment_table10_domain_specific.py" \
  "figure2:experiment_figure2_threshold_sweep.py" \
  "figure3:experiment_figure3_policy_distribution.py"; do
  name="${entry%%:*}"
  script="${entry#*:}"
  echo "===== ${name} =====" | tee "logs/${name}.log"
  "$PY" "$script" "${COMMON[@]}" 2>&1 | tee -a "logs/${name}.log"
done

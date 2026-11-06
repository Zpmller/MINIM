#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
export MINIM_FEATURE_CACHE_DIR="$PWD/runs/minim/feature_cache"
PY=.venv/bin/python
OUT="${RESULTS_DIR:-results/paper_aligned_selected}"
TAU_NEC="${TAU_NEC:-0.0}"
mkdir -p logs "$OUT"
COMMON=(--data_root data_paper_aligned --split_path runs/paper_aligned/split.json \
  --checkpoint runs/paper_aligned/model.pt --results_dir "$OUT" \
  --model_type gatv2 --device cuda --seed 0)
for entry in \
  "table1:experiment_table1_main_results.py" \
  "table3:experiment_table3_task_semantics.py" \
  "table4:experiment_table4_abstraction_ablation.py" \
  "table10:experiment_table10_domain_specific.py" \
  "figure2:experiment_figure2_threshold_sweep.py" \
  "figure3:experiment_figure3_policy_distribution.py"; do
  name="${entry%%:*}"
  script="${entry#*:}"
  logfile="logs/paper_aligned_selected_${name}.log"
  echo "===== paper-aligned selected ${name} (tau_nec=${TAU_NEC}) =====" | tee "$logfile"
  if [[ "$name" == "figure2" ]]; then
    "$PY" "$script" "${COMMON[@]}" --selected_tau "$TAU_NEC" 2>&1 | tee -a "$logfile"
  else
    "$PY" "$script" "${COMMON[@]}" --tau_nec "$TAU_NEC" 2>&1 | tee -a "$logfile"
  fi
done

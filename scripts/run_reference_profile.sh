#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
export MINIM_FEATURE_CACHE_DIR="$PWD/runs/minim/feature_cache"
run_dir=runs/reference_profile_20261104
mkdir -p "$run_dir" logs results/reference_profile_20261104
.venv/bin/python - <<'PY'
import json
from pathlib import Path
source = Path('scripts/reference_profile_protocol.json')
target = Path('runs/reference_profile_20261104/protocol.json')
if target.exists():
    assert json.loads(target.read_text()) == json.loads(source.read_text()), 'Existing protocol differs'
else:
    target.write_bytes(source.read_bytes())
PY
if [[ ! -s "$run_dir/model.pt" ]]; then
  .venv/bin/python -u train_minim_scorer.py \
    --data_root data --output_dir "$run_dir" --device cuda --model_type gatv2 \
    --gat_hidden 256 --gat_heads 4 --epochs 10 --batch_size 32 --seed 0 \
    --lr 3e-4 --weight_decay 1e-5 --alpha 1 --ci_lambda 1 \
    --task_encoding unique --loss_type ci_ce \
    2>&1 | tee logs/train_reference_profile_20261104.log
fi
.venv/bin/python -u scripts/evaluate_reference_profile.py \
  2>&1 | tee logs/evaluate_reference_profile_20261104.log

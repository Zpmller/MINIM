#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs runs/training_scale
if [[ ! -f runs/training_scale/official_paper_full/summary.json ]]; then
  .venv/bin/python -u scripts/run_training_scale.py --mode official --stage train > logs/training_scale_official.log 2>&1
fi
for trees in 6 15 30; do
  tag=$(printf '%02d' "$trees")
  if [[ ! -f "runs/training_scale/repair_${tag}/summary.json" ]]; then
    .venv/bin/python -u scripts/run_training_scale.py --mode repair --trees "$trees" --stage train > "logs/training_scale_repair_${tag}.log" 2>&1
  fi
done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('runs/training_scale')
names=['repair_06','repair_15','repair_30']
metrics={n:json.loads((root/n/'summary.json').read_text())['validation'] for n in names}
selected=max(names,key=lambda n:metrics[n]['selection_score'])
decision={'selected_run':selected,'validation_metrics':metrics,'selection_uses_only_validation':True}
path=root/'selection_before_test.json'
if path.exists():
    assert json.loads(path.read_text()) == decision, 'Saved validation decision changed; use a fresh run directory'
else:
    path.write_text(json.dumps(decision,indent=2)+'\n')
print('Selected before test:',selected)
PY
if [[ ! -f runs/training_scale/official_paper_full/test_metrics.json ]]; then
  .venv/bin/python -u scripts/run_training_scale.py --mode official --stage test > logs/training_scale_official_test.log 2>&1
fi
for trees in 6 15 30; do
  tag=$(printf '%02d' "$trees")
  if [[ ! -f "runs/training_scale/repair_${tag}/test_metrics.json" ]]; then
    .venv/bin/python -u scripts/run_training_scale.py --mode repair --trees "$trees" --stage test > "logs/training_scale_repair_${tag}_test.log" 2>&1
  fi
done
.venv/bin/python scripts/analyze_training_scale.py > logs/training_scale_analysis.log
.venv/bin/python scripts/export_training_scale.py > logs/training_scale_export.log
.venv/bin/python scripts/analyze_training_scale.py > logs/training_scale_analysis.log

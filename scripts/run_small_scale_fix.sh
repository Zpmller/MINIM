#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs
.venv/bin/python -u scripts/run_small_scale_fix.py --stage train 2>&1 | tee logs/pilot_fix_train.log
.venv/bin/python -u scripts/run_small_scale_fix.py --stage test 2>&1 | tee logs/pilot_fix_test.log
.venv/bin/python -u scripts/run_small_scale_fix.py --stage export
.venv/bin/python scripts/summarize_small_scale_fix.py

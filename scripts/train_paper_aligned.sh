#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
export MINIM_EMBEDDER_PATH="$PWD/data/models/all-MiniLM-L6-v2"
# Feature arrays are model-independent and identical for the retained rows;
# reuse the verified full-data cache created during the first run.
export MINIM_FEATURE_CACHE_DIR="$PWD/runs/minim/feature_cache"
mkdir -p logs runs/paper_aligned
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python train_minim_scorer.py \
  --data_root data_paper_aligned --output_dir runs/paper_aligned \
  --model_type gatv2 --device cuda --seed 0 \
  --train_frac 0.8 --val_frac 0.1 --epochs 10 --batch_size 32 \
  --lr 3e-4 --weight_decay 1e-5 --gat_hidden 256 --gat_heads 4 \
  --alpha 1.0 --ci_lambda 1.0 \
  --sens_high_boost 0.0 --nec_low_boost 0.0 --nec_high_boost 0.0 \
  --task_encoding legacy --loss_type ci_l1 \
  2>&1 | tee logs/train_paper_aligned.log
.venv/bin/python scripts/plot_training.py logs/train_paper_aligned.log
cp results/training_history.csv results/paper_aligned_training_history.csv
cp results/training_loss.png results/paper_aligned_training_loss.png

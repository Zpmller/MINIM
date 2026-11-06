#!/usr/bin/env python3
"""Run the released Table 1 scripts against the two separately labelled tracks."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
selected = json.loads((ROOT / 'results/training_scale/selection.json').read_text())['selected_run']
for name in ('official_paper_full', selected):
    run = ROOT / 'runs/training_scale' / name
    out = ROOT / 'results/training_scale' / name
    out.mkdir(exist_ok=True)
    common = ['--data_root', 'data', '--split_path', str(run / 'split.json'),
              '--checkpoint', str(run / 'model.pt'), '--results_dir', str(out), '--device', 'cuda']
    scripts = ['experiment_table1_main_results.py']
    if name == selected:
        scripts.append('experiment_table4_abstraction_ablation.py')
    for script in scripts:
        with (ROOT / 'logs' / f'training_scale_{name}_{script}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', script, *common], check=True, stdout=log, stderr=subprocess.STDOUT)
    measured = json.loads((run / 'test_metrics.json').read_text())
    row = next(r for r in csv.DictReader((out / 'table1.csv').open()) if r['method'] == 'MINIM')
    for field in ('tcnp', 'tcnp_i', 'tisl'):
        assert abs(float(row[field]) - measured[field]) < 1e-6
    print(f'{name}: released Table 1 matches independent evaluation')

#!/usr/bin/env python3
"""One predeclared CI-L1 fine-tuning epoch from a legacy-encoded CE checkpoint."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_training_scale import Nodes, official_snapshot
from run_small_scale_fix import evaluate
from train_minim_scorer import featurize_variant, set_seed

BASE = ROOT / 'runs/causal27_20261102/full_ci_ce'
RUN = ROOT / 'runs/official_l1_finetune_20261104'
OUT = ROOT / 'results/official_l1_finetune_20261104'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    os.chdir(ROOT)
    RUN.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    torch.set_num_threads(2)
    set_seed(0)
    device = torch.device('cuda')
    config = json.loads((BASE / 'train_config.json').read_text())
    assert config['task_encoding'] == 'legacy' and config['task_vocab'] is None and not config['prior_correction']
    split = json.loads((BASE / 'split.json').read_text())
    assert split == json.loads((ROOT / 'runs/paper_aligned/split.json').read_text())
    groups = {k: {(r['domain'], r['sample_id']) for r in v} for k, v in split.items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    base_hash = hashlib.sha256((BASE / 'model.pt').read_bytes()).hexdigest()
    protocol = {'date': '2026-11-04', 'initial_checkpoint': str(BASE / 'model.pt'),
                'initial_checkpoint_sha256': base_hash, 'epochs': 1, 'batch_size': 32,
                'lr': 3e-5, 'weight_decay': 1e-5, 'seed': 0, 'loss': 'frozen official ci_loss',
                'alpha': 1, 'ci_lambda': 1, 'extra_boosts': [0, 0, 0],
                'task_encoding': 'legacy', 'thresholds': [1, 5],
                'selection': 'fixed final epoch; no threshold or test-based selection',
                'purpose': 'restore the original objective and task encoding; accept actual results without targeting a lower score',
                'test_status': 'previously used exploratory split, not a fresh independent confirmation',
                'manifest_sha256': hashlib.sha256((ROOT / 'data_paper_aligned/manifest.jsonl').read_bytes()).hexdigest()}
    if (RUN / 'protocol.json').exists():
        assert json.loads((RUN / 'protocol.json').read_text()) == protocol
    else:
        write_json(RUN / 'protocol.json', protocol)
    frozen, source_hash = official_snapshot(RUN)
    model = frozen.build_scorer('gatv2', gat_hidden=256, gat_heads=4).to(device)
    model.load_state_dict(torch.load(BASE / 'model.pt', map_location=device)['model_state'], strict=True)
    nodes = Nodes()
    def variants(name):
        return (featurize_variant(r, 'data_paper_aligned', 'gatv2') for r in split[name])
    before = evaluate(model, variants('val'), nodes, device, scorer=frozen.score_variant)
    print('Before fine-tuning, validation:', before, flush=True)
    config.update(output_dir=str(RUN), epochs=1, lr=3e-5, loss_type='ci_l1',
                  init_checkpoint=str(BASE / 'model.pt'), init_checkpoint_sha256=base_hash,
                  training_phase='CE pretraining followed by one official CI-L1 fine-tuning epoch',
                  official_source_sha256=source_hash)
    if (RUN / 'model.pt').exists():
        existing = torch.load(RUN / 'model.pt', map_location=device)
        assert existing['config'] == config
        model.load_state_dict(existing['model_state'], strict=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
        order = list(range(len(split['train'])))
        random.Random(0).shuffle(order)
        history = []
        model.train()
        start_time = time.monotonic()
        for step, offset in enumerate(range(0, len(order), 32), 1):
            graphs = [featurize_variant(split['train'][i], 'data_paper_aligned', 'gatv2') for i in order[offset:offset+32]]
            batch = frozen.collate_variants(graphs, device)
            sh, nh = frozen.score_batch(model, batch)
            loss, ls, ln = frozen.ci_loss(batch['sens'], batch['nec'], sh, nh,
                alpha=1, ci_lambda=1, sens_high_boost=0, nec_low_boost=0, nec_high_boost=0)
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            history.append({'step': step, 'loss': float(loss.detach()), 'sensitivity_loss': float(ls.detach()),
                            'necessity_loss': float(ln.detach()), 'elapsed_seconds': time.monotonic()-start_time})
            if step % 20 == 0 or offset + 32 >= len(order):
                print(f'Official L1 fine-tune {step}/137 loss={history[-1]["loss"]:.6f}', flush=True)
        torch.save({'model_type': 'gatv2', 'model_state': model.state_dict(), 'config': config}, RUN / 'model.pt')
        write_json(RUN / 'train_config.json', config)
        write_json(RUN / 'split.json', split)
        with (OUT / 'history.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    validation = evaluate(model, variants('val'), nodes, device, scorer=frozen.score_variant)
    print('After fine-tuning, validation:', validation, flush=True)
    write_json(RUN / 'evaluation_lock.json', {'checkpoint_sha256': hashlib.sha256((RUN / 'model.pt').read_bytes()).hexdigest(),
                'policy': {'tau_nec': 1, 'tau_sens': 5}, 'validation': validation, 'test_not_used_for_selection': True})
    test = evaluate(model, variants('test'), nodes, device, scorer=frozen.score_variant)
    result = {'protocol': protocol, 'before_validation': before, 'after_validation': validation, 'test': test,
              'checkpoint_sha256': hashlib.sha256((RUN / 'model.pt').read_bytes()).hexdigest(),
              'official_source_sha256': source_hash}
    write_json(OUT / 'summary.json', result)
    del model
    torch.cuda.empty_cache()
    for entry in ('table1_main_results', 'table4_abstraction_ablation'):
        cmd = [sys.executable, '-u', f'experiment_{entry}.py', '--data_root', 'data_paper_aligned',
               '--split_path', str(RUN / 'split.json'), '--checkpoint', str(RUN / 'model.pt'),
               '--results_dir', str(OUT), '--device', 'cuda', '--tau_nec', '1', '--tau_sens', '5']
        with (ROOT / f'logs/official_l1_finetune_{entry}.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    table = next(r for r in csv.DictReader((OUT / 'table1.csv').open()) if r['method'] == 'MINIM')
    assert all(abs(float(table[k]) - test[k]) < 1e-6 for k in ('tcnp', 'tcnp_i', 'tisl'))
    assert hashlib.sha256((BASE / 'model.pt').read_bytes()).hexdigest() == base_hash
    write_json(OUT / 'verification.json', {'original_checkpoint_unchanged': True,
                'frozen_official_model_and_loss_used': True, 'official_table1_matches_independent_evaluation': True,
                'original_legacy_encoding_and_thresholds': True})
    print('Completed test:', test, flush=True)


if __name__ == '__main__':
    main()

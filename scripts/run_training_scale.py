#!/usr/bin/env python3
"""Paper-aligned official baseline and a separate repaired-data scaling study."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_minim_inference import build_model
from run_small_scale_fix import evaluate, prepare_split, write_json
from train_minim_scorer import (balanced_score_loss, build_scorer, collate_variants,
    featurize_variant, label_balance_weights, score_variant, set_seed)

COMMIT = '456c60b6fd47941d1e5849db573dbbcb488e3ae7'


class Nodes:
    @lru_cache(maxsize=16)
    def __getitem__(self, path):
        return json.loads(Path(path).read_text())['nodes']


def official_snapshot(out):
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:train_minim_scorer.py'], cwd=ROOT)
    path = out / 'official_train_snapshot.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('minim_official_frozen', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('official', 'repair'), required=True)
    parser.add_argument('--trees', type=int, default=0, help='0 = all original training trees')
    parser.add_argument('--stage', choices=('train', 'test'), default='train')
    parser.add_argument('--output_root', default='runs/training_scale')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    name = 'official_paper_full' if args.mode == 'official' else f'repair_{args.trees:02d}'
    out = ROOT / args.output_root / name
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / 'runs/minim/split.json').read_text())
    nodes = Nodes()
    if args.stage == 'test':
        config = json.loads((out / 'train_config.json').read_text())
        split = json.loads((out / 'split.json').read_text())
        model = build_model(str(out / 'model.pt'), device)
        data = (featurize_variant(row, 'data', 'gatv2', config['task_vocab']) for row in split['test'])
        scores = evaluate(model, data, nodes, device)
        scores['checkpoint_sha256'] = hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest()
        write_json(out / 'test_metrics.json', scores)
        print(json.dumps(scores, indent=2), flush=True)
        return
    if (out / 'summary.json').exists():
        raise RuntimeError(f'Completed run already exists: {out}; choose a new output root')
    official = args.mode == 'official'
    if official:
        split = original
        vocab = None
        epochs, effective_batch = 10, 32
        frozen, source_hash = official_snapshot(out)
    else:
        if args.trees <= 0 or args.trees % 3:
            raise ValueError('Repaired runs require a positive multiple of three training trees')
        split = prepare_split(args.trees // 3)
        split['val'] = [row for row in original['val'] if Path(row['tree_with_scores_path']).parent.name.startswith(
            ('variant_balanced_', 'variant_reading_'))]
        split['test'] = original['test']
        vocab = {g: i for i, g in enumerate(sorted({r['goal_id'] for r in split['train']}))}
        epochs, effective_batch = 40, 4
        source_hash = None
    groups = {k: {(r['domain'], r['sample_id']) for r in rows} for k, rows in split.items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    config = {'model_type': 'gatv2', 'gat_hidden': 256, 'gat_heads': 4, 'seed': 0,
        'epochs': epochs, 'batch_size': effective_batch, 'micro_batch_size': 4,
        'lr': 3e-4, 'weight_decay': 1e-5, 'alpha': 1., 'ci_lambda': 1.,
        'sens_high_boost': 0., 'nec_low_boost': 0., 'nec_high_boost': 0.,
        'task_vocab': vocab, 'task_encoding': 'legacy' if official else 'unique',
        'loss_type': 'ci_l1' if official else 'balanced_ce', 'prior_correction': not official,
        'training_tree_count': len(groups['train']), 'training_variants': len(split['train']),
        'validation_tree_count': len(groups['val']), 'validation_variants': len(split['val']),
        'test_tree_count': len(groups['test']), 'test_variants': len(split['test']),
        'selection': 'last epoch (official trainer behavior)' if official else 'max val micro TCNP - 0.5*TISL; earliest tie',
        'official_commit': COMMIT if official else None, 'official_source_sha256': source_hash,
        'manifest_sha256': hashlib.sha256((ROOT / 'data/manifest.jsonl').read_bytes()).hexdigest(),
        'original_model_sha256': hashlib.sha256((ROOT / 'runs/minim/model.pt').read_bytes()).hexdigest()}
    write_json(out / 'train_config.json', config)
    write_json(out / 'split.json', split)
    def features(row):
        return featurize_variant(row, 'data', 'gatv2', vocab)
    val_data = [features(row) for row in split['val']]
    train_data = None if official else [features(row) for row in split['train']]
    print(f"{name}: train={len(split['train'])} variants/{len(groups['train'])} trees; "
          f"val={len(val_data)} variants/{len(groups['val'])} trees; test excluded from training", flush=True)
    class_weights = None if official else label_balance_weights(train_data)
    if class_weights is not None:
        config['class_weights'] = [w.tolist() for w in class_weights]
    set_seed(0)
    model = (frozen.build_scorer('gatv2', gat_hidden=256, gat_heads=4) if official else
             build_scorer('gatv2', gat_hidden=256, gat_heads=4)).to(device)
    model.task_vocab = vocab
    if not official:
        model.set_class_prior_correction(class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scorer = frozen.score_variant if official else score_variant
    collate = frozen.collate_variants if official else collate_variants
    rng = random.Random(0)
    history, best_score, start_time = [], -float('inf'), time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(split['train'])))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), effective_batch):
            ids = order[start:start + effective_batch]
            window = [features(split['train'][i]) if official else train_data[i] for i in ids]
            total_nodes = sum(len(v.nec) for v in window)
            optimizer.zero_grad()
            window_loss = 0.
            for offset in range(0, len(window), 4):
                batch = collate(window[offset:offset + 4], device)
                if official:
                    sh, nh = frozen.score_batch(model, batch)
                    loss, _, _ = frozen.ci_loss(batch['sens'], batch['nec'], sh, nh,
                        alpha=1., ci_lambda=1., sens_high_boost=0., nec_low_boost=0., nec_high_boost=0.)
                else:
                    sl, nl = model.forward_logits(batch['node_features'], batch['edge_index'])
                    loss, _, _ = balanced_score_loss(batch['sens'], batch['nec'], sl, nl, class_weights)
                fraction = len(batch['nec']) / total_nodes
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                (loss * fraction).backward()
                window_loss += float(loss.detach()) * fraction
            optimizer.step()
            losses.append(window_loss)
        val = evaluate(model, val_data, nodes, device, scorer=scorer)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'elapsed_seconds': time.monotonic() - start_time, **val}
        history.append(row)
        print(f"{name} epoch {epoch}/{epochs}: loss={row['train_loss']:.5f}, "
              f"val micro={val['micro_tcnp']:.4f}, interactive={val['micro_tcnp_i']:.4f}, "
              f"TISL={val['tisl']:.4f}, elapsed={row['elapsed_seconds']:.0f}s", flush=True)
        checkpoint_config = {**config, 'selected_epoch': epoch}
        payload = {'model_type': 'gatv2', 'model_state': model.state_dict(), 'config': checkpoint_config}
        if official:
            torch.save(payload, out / f'epoch_{epoch:02d}.pt')
        if (official and epoch == epochs) or (not official and val['selection_score'] > best_score):
            torch.save(payload, out / 'model.pt')
            config_selected = checkpoint_config
            best_val = val
            best_score = val['selection_score']
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    write_json(out / 'train_config.json', config_selected)
    loaded = build_model(str(out / 'model.pt'), device)
    restored_val = evaluate(loaded, val_data, nodes, device)
    assert abs(restored_val['selection_score'] - best_val['selection_score']) < 1e-6
    result = {'config': config_selected, 'validation': restored_val,
              'last_validation': val, 'elapsed_seconds': time.monotonic() - start_time,
              'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    if not official:
        result['training'] = evaluate(loaded, train_data, nodes, device)
    write_json(out / 'summary.json', result)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()

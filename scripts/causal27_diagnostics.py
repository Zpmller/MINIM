#!/usr/bin/env python3
"""Isolated causal diagnostics for the 27-task scorer collapse.

No task re-encoding, test tuning, or overwriting of previous runs.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_training_scale import Nodes, official_snapshot
from run_small_scale_fix import evaluate, write_json
from train_minim_scorer import (MINILM_MODEL_NAME, featurize_variant,
                               label_balance_weights, set_seed)

RUN = ROOT / 'runs/causal27_20261102'
OUT = ROOT / 'results/causal27_20261102'


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def logits(model, batch):
    x = batch['node_features']
    norms = []
    for layer in model.gat_layers:
        x = F.elu(layer(x, batch['edge_index']))
        norms.append(float(x.detach().norm(dim=1).mean()))
    return model.sens_head(x), model.nec_head(x), x, norms


def expected(z):
    return (z.softmax(-1) * torch.arange(11, device=z.device)).sum(-1)


def objective(batch, sl, nl, kind, weights=None):
    terms = []
    for index, (target, z) in enumerate(((batch['sens'], sl), (batch['nec'], nl))):
        w = torch.ones_like(target)
        if weights is not None:
            w *= weights[index].to(z.device)[target.long()]
        if index == 1:
            w *= 1 + batch['sens'] / 10 * (1 - batch['nec'] / 10)
        error = (expected(z) - target).abs() if kind == 'l1' else F.cross_entropy(z, target.long(), reduction='none')
        # Common node-mean normalization in all four arms; no batch-dependent
        # renormalization of weights that could change the relative step size.
        terms.append((w * error).mean())
    return sum(terms)


def probe(model, variants, frozen, device):
    values = defaultdict(list)
    norms = []
    with torch.no_grad():
        for variant in variants:
            batch = frozen.collate_variants([variant], device)
            sl, nl, x, layer_norms = logits(model, batch)
            s, n = expected(sl), expected(nl)
            values['s'].append(s)
            values['n'].append(n)
            values['sp0'].append(sl.softmax(-1)[:, 0])
            values['np0'].append(nl.softmax(-1)[:, 0])
            values['critical_n'].append(n[batch['nec'] >= 7])
            values['irrelevant_n'].append(n[batch['nec'] == 0])
            norms.append(layer_norms)
    values = {k: torch.cat(v) for k, v in values.items()}
    return {'probe_s_mean': float(values['s'].mean()), 'probe_n_mean': float(values['n'].mean()),
        'probe_n_max': float(values['n'].max()), 'probe_remove_rate': float((values['n'] < 1).float().mean()),
        'probe_critical_n_mean': float(values['critical_n'].mean()),
        'probe_critical_recall': float((values['critical_n'] >= 1).float().mean()),
        'probe_irrelevant_n_mean': float(values['irrelevant_n'].mean()),
        'probe_s_bin0_probability': float(values['sp0'].mean()),
        'probe_n_bin0_probability': float(values['np0'].mean()),
        **{f'probe_layer{i+1}_norm': float(np.mean([x[i] for x in norms])) for i in range(3)}}


def trace(split, frozen, device):
    train_rows = split['train']
    # Select three positive-bearing training graphs, without inspecting test.
    candidates = {}
    for row in train_rows:
        digest = hashlib.sha256(f"gatv2\0{MINILM_MODEL_NAME}\0{row['tree_with_scores_path']}".encode()).hexdigest()
        with np.load(ROOT / 'runs/minim/feature_cache' / digest[:2] / (digest + '.npz')) as a:
            n, s = a['nec'], a['sens']
        if (n >= 7).sum() >= 3 and (s >= 5).any():
            if row['domain'] not in candidates or len(n) < candidates[row['domain']][0]:
                candidates[row['domain']] = (len(n), row)
    probes = [featurize_variant(candidates[d][1], 'data_paper_aligned', 'gatv2')
              for d in ('gmail', 'reddit', 'shopping')]
    write_json(RUN / 'trace_probe_rows.json', [candidates[d][1] for d in ('gmail', 'reddit', 'shopping')])
    set_seed(0)
    model = frozen.build_scorer('gatv2', gat_hidden=256, gat_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    order = list(range(len(train_rows)))
    random.Random(0).shuffle(order)
    history = [{'step': 0, 'loss': None, **probe(model, probes, frozen, device)}]
    details, onset, started = [], None, time.monotonic()
    for start in range(0, len(order), 32):
        model.train()
        variants = [featurize_variant(train_rows[j], 'data_paper_aligned', 'gatv2') for j in order[start:start+32]]
        batch = frozen.collate_variants(variants, device)
        sl, nl, _, _ = logits(model, batch)
        sh, nh = expected(sl), expected(nl)
        loss, _, _ = frozen.ci_loss(batch['sens'], batch['nec'], sh, nh,
            sens_high_boost=0., nec_low_boost=0., nec_high_boost=0.)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, target, z, pred, head, weight in (
                ('sens', batch['sens'], sl, sh, model.sens_head, torch.ones_like(sh)),
                ('nec', batch['nec'], nl, nh, model.nec_head, 1+batch['sens']/10*(1-batch['nec']/10))):
                p = z.softmax(-1)
                gradient = p * (torch.arange(11, device=device) - pred[:, None]) * (
                    weight * (pred - target).sign() / len(target))[:, None]
                error = float((gradient.sum(0) - head[-1].bias.grad).abs().max())
                assert torch.allclose(gradient.sum(0), head[-1].bias.grad, atol=2e-6, rtol=1e-4)
                for group, mask in [('zero', target == 0), ('positive', target > 0), ('high', target >= 7)]:
                    details.append({'step': len(history), 'head': name, 'group': group,
                        'nodes': int(mask.sum()), 'score_mean': float(pred[mask].mean()) if mask.any() else 0.,
                        'logit_grad_abs_sum': float(gradient[mask].abs().sum()),
                        'bias0_gradient': float(gradient[mask, 0].sum()),
                        'uniform_score_derivative': float((weight * (pred-target).sign())[mask].sum()/len(target)),
                        'analytic_autograd_max_difference': error})
        optimizer.step()
        row = {'step': len(history), 'loss': float(loss.detach()), **probe(model, probes, frozen, device)}
        history.append(row)
        if onset is None and row['probe_critical_recall'] == 0:
            onset = row['step']
        if row['step'] % 10 == 0 or row['step'] == 1:
            print(f"trace step={row['step']}: n={row['probe_n_mean']:.5f}, critical_recall={row['probe_critical_recall']:.3f}, "
                  f"p0={row['probe_n_bin0_probability']:.6f}, layer3_norm={row['probe_layer3_norm']:.2f}", flush=True)
    save_csv(OUT / 'trace.csv', history)
    save_csv(OUT / 'gradient_groups.csv', details)
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, RUN / 'trace_final.pt')
    write_json(OUT / 'trace_summary.json', {'steps': len(history)-1,
        'first_step_all_probe_critical_removed': onset, 'initial': history[0], 'final': history[-1],
        'elapsed_seconds': time.monotonic()-started})
    recovery(model, optimizer, probes, frozen, device)


def recovery(original, optimizer, probes, frozen, device):
    features, labels, sens = [], [], []
    with torch.no_grad():
        for variant in probes:
            batch = frozen.collate_variants([variant], device)
            _, _, x, _ = logits(original, batch)
            positive = batch['nec'] >= 7
            features.append(x[positive].detach())
            labels.append(batch['nec'][positive])
            sens.append(batch['sens'][positive])
    x, target, sensitivity = torch.cat(features), torch.cat(labels), torch.cat(sens)
    w = 1+sensitivity/10*(1-target/10)
    rows = []
    for kind in ('l1', 'ce'):
        for state in ('carry', 'fresh'):
            model = copy.deepcopy(original)
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.nec_head.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
            if state == 'carry':
                opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            for step in range(401):
                z = model.nec_head(x)
                pred = expected(z)
                if step in (0, 1, 10, 25, 50, 100, 200, 400):
                    rows.append({'loss': kind, 'optimizer': state, 'step': step, 'nodes': len(target),
                        'mean_score': float(pred.detach().mean()),
                        'critical_recall': float((pred.detach() >= 1).float().mean()),
                        'mae': float((pred.detach()-target).abs().mean())})
                if step == 400:
                    break
                error = (pred-target).abs() if kind == 'l1' else F.cross_entropy(z, target.long(), reduction='none')
                loss = (w * error).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f'recovery {kind}/{state}: {rows[-1]}', flush=True)
    save_csv(OUT / 'recovery.csv', rows)


def factorial(split, frozen, device):
    selection_rng = random.Random(20261031)
    selected_groups = set()
    for domain in ('gmail', 'reddit', 'shopping'):
        candidates = sorted({r['sample_id'] for r in split['train'] if r['domain'] == domain})
        selection_rng.shuffle(candidates)
        selected_groups.update((domain, sample) for sample in candidates[:5])
    train_rows = [r for r in split['train'] if (r['domain'], r['sample_id']) in selected_groups
                  and Path(r['tree_with_scores_path']).parent.name.startswith('variant_balanced_')]
    train_data = [featurize_variant(row, 'data_paper_aligned', 'gatv2') for row in train_rows]
    val_data = [featurize_variant(row, 'data_paper_aligned', 'gatv2') for row in split['val']]
    weights = label_balance_weights(train_data)
    write_json(RUN / 'factorial_split.json', {'train': train_rows, 'val': split['val']})
    write_json(RUN / 'factorial_config.json', {'train_trees': 15, 'train_variants': len(train_rows),
        'train_nodes': sum(len(v.nec) for v in train_data), 'val_variants': len(val_data),
        'model': 'frozen official GATv2, hidden256/heads4', 'task_encoding': 'original hash',
        'seed': 0, 'epochs': 40, 'batch_size': 4, 'lr': 3e-4,
        'loss_normalization': 'node mean in every arm', 'test_evaluated': False,
        'class_weights': [w.tolist() for w in weights]})
    nodes = Nodes()
    all_rows, summaries = [], {}
    for kind, balanced in (('l1', False), ('l1', True), ('ce', False), ('ce', True)):
        name = f'{kind}_{"balanced" if balanced else "natural"}'
        set_seed(0)
        model = frozen.build_scorer('gatv2', gat_hidden=256, gat_heads=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        rng = random.Random(0)
        best = -float('inf')
        for epoch in range(1, 41):
            model.train()
            order = list(range(len(train_data)))
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), 4):
                batch = frozen.collate_variants([train_data[i] for i in order[start:start+4]], device)
                sl, nl, _, _ = logits(model, batch)
                loss = objective(batch, sl, nl, kind, weights if balanced else None)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            if epoch == 1 or epoch % 5 == 0:
                val = evaluate(model, val_data, nodes, device, scorer=frozen.score_variant)
                row = {'arm': name, 'epoch': epoch, 'train_loss': float(np.mean(losses)), **val}
                all_rows.append(row)
                if val['selection_score'] > best:
                    best = val['selection_score']
                    best_val = {'epoch': epoch, **val}
                print(f'{name} {epoch}/40: loss={row["train_loss"]:.4f}, val_recall={val["micro_tcnp"]:.4f}, '
                      f'TISL={val["tisl"]:.4f}, R={val["remove_rate"]:.3f}', flush=True)
        train_metrics = evaluate(model, train_data, nodes, device, scorer=frozen.score_variant)
        summaries[name] = {'final_validation': val, 'best_validation': best_val, 'training': train_metrics}
        torch.save({'model_type': 'gatv2', 'model_state': model.state_dict(),
            'config': {'gat_hidden': 256, 'gat_heads': 4, 'task_vocab': None, 'prior_correction': False,
                       'objective': name}}, RUN / f'{name}.pt')
        write_json(OUT / 'factorial_summary.json', summaries)
        save_csv(OUT / 'factorial_history.csv', all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('trace', 'factorial'), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    torch.set_num_threads(2)
    split = json.loads((ROOT / 'runs/paper_aligned/split.json').read_text())
    assert len({r['goal_id'] for part in split.values() for r in part}) == 27
    frozen, source_hash = official_snapshot(RUN)
    write_json(RUN / 'provenance.json', {'source_sha256': source_hash,
        'manifest_sha256': hashlib.sha256((ROOT / 'data_paper_aligned/manifest.jsonl').read_bytes()).hexdigest(),
        'split_sha256': hashlib.sha256((ROOT / 'runs/paper_aligned/split.json').read_bytes()).hexdigest(),
        'date': '2026-11-02', 'all_new_checkpoints_are_diagnostic': True})
    if args.stage == 'trace':
        trace(split, frozen, torch.device('cuda'))
    else:
        factorial(split, frozen, torch.device('cuda'))


if __name__ == '__main__':
    main()

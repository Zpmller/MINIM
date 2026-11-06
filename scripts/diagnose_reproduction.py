#!/usr/bin/env python3
"""Read-only audit of the saved MINIM run; writes separate diagnostic artifacts."""
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_metrics import (compute_tcnp, compute_tcnp_i,
                             compute_tisl_denominator, compute_tisl_numerator,
                             is_interactive)
from experiment_table1_main_results import random_budget_actions
from run_minim_inference import build_model
from train_minim_scorer import (MINILM_MODEL_NAME, featurize_variant,
                               score_variant, stable_hash)


def main():
    import random
    os.chdir(ROOT)
    torch.set_num_threads(2)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    out = ROOT / 'results/diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / 'runs/minim/split.json').read_text())
    report = {'checkpoint_sha256': hashlib.sha256((ROOT / 'runs/minim/model.pt').read_bytes()).hexdigest(),
              'splits': {}, 'cache_label_checks': [], 'gradient_checks': []}
    per_graph = []
    domain_examples = {}
    for name, rows in split.items():
        sh, nh = Counter(), Counter()
        zero_loss = []
        all_s, all_n = [], []
        for row in rows:
            cache_id = hashlib.sha256(f"gatv2\0{MINILM_MODEL_NAME}\0{row['tree_with_scores_path']}".encode()).hexdigest()
            cache_path = ROOT / 'runs/minim/feature_cache' / cache_id[:2] / (cache_id + '.npz')
            with np.load(cache_path, allow_pickle=False) as a:
                s, n = a['sens'], a['nec']
            sh.update(s.tolist())
            nh.update(n.tolist())
            all_s.append(s)
            all_n.append(n)
            loss = float(np.mean(s + (1 + s / 10 * (1 - n / 10)) * n))
            zero_loss.append(loss)
            per_graph.append({'split': name, 'domain': row['domain'], 'sample_id': row['sample_id'],
                              'goal_id': row['goal_id'], 'path': row['tree_with_scores_path'],
                              'nodes': len(n), 'nec_positive': int((n > 0).sum()),
                              'critical_nodes': int((n >= 7).sum()), 'zero_predictor_loss': loss})
            if row['domain'] not in domain_examples:
                domain_examples[row['domain']] = row
        s, n = np.concatenate(all_s), np.concatenate(all_n)
        w = 1 + s / 10 * (1 - n / 10)
        sorted_n = np.argsort(n)
        weighted_median = float(n[sorted_n][np.searchsorted(np.cumsum(w[sorted_n], dtype=np.float64), w.sum(dtype=np.float64) / 2)])
        report['splits'][name] = {
            'variants': len(rows), 'samples': len({(r['domain'], r['sample_id']) for r in rows}),
            'domain_variants': dict(Counter(r['domain'] for r in rows)), 'nodes': len(n),
            'sens_histogram': dict(sorted(sh.items())), 'nec_histogram': dict(sorted(nh.items())),
            'sens_zero_fraction': float((s == 0).mean()), 'nec_zero_fraction': float((n == 0).mean()),
            'critical_fraction': float((n >= 7).mean()),
            'sens_constant_L1_median': float(np.median(s)),
            'nec_constant_CI_L1_weighted_median': weighted_median,
            'zero_predictor_loss_node_mean': float(np.mean(s + w * n)),
            'zero_predictor_loss_graph_mean': float(np.mean(zero_loss)),
            'graphs_without_critical_nodes': sum(x['critical_nodes'] == 0 for x in per_graph if x['split'] == name),
        }
        print(f"labels {name}: {report['splits'][name]}", flush=True)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        sa = {(r['domain'], r['sample_id']) for r in split[a]}
        sb = {(r['domain'], r['sample_id']) for r in split[b]}
        report[f'{a}_{b}_sample_overlap'] = len(sa & sb)
    goal_sets = {k: set(r['goal_id'] for r in v) for k, v in split.items()}
    report['goals'] = {k: sorted(v) for k, v in goal_sets.items()}
    report['unseen_test_goals'] = sorted(goal_sets['test'] - goal_sets['train'])
    buckets = defaultdict(list)
    for goal in set.union(*goal_sets.values()):
        buckets[stable_hash(goal) % 32].append(goal)
    report['goal_feature_hash_collisions'] = {str(k): sorted(v) for k, v in buckets.items() if len(v) > 1}
    for row in domain_examples.values():
        v = featurize_variant(row, 'data', 'gatv2')
        nodes = json.loads(Path(v.tree_path).read_text())['nodes']
        report['cache_label_checks'].append({'path': row['tree_with_scores_path'],
            'sens_equal': bool(np.array_equal(v.sens.numpy(), [x.get('sensitivity_score', 0) for x in nodes])),
            'nec_equal': bool(np.array_equal(v.nec.numpy(), [x.get('necessity_score', 0) for x in nodes])),
            'features_finite': bool(torch.isfinite(v.node_features).all()),
            'embedding_norm_mean': float(v.node_features[:, :384].norm(dim=1).mean())})
    with (out / 'per_graph.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_graph[0].keys())
        writer.writeheader()
        writer.writerows(per_graph)

    # Replay exactly the seeded shuffle and the trainer's batch-level reduction.
    train_graphs = [x for x in per_graph if x['split'] == 'train']
    history = list(csv.DictReader((ROOT / 'results/training_history.csv').open()))
    shuffle_rng = random.Random(0)
    zero_history = []
    for epoch, logged in enumerate(history, 1):
        order = list(range(len(train_graphs)))
        shuffle_rng.shuffle(order)
        batch_losses = []
        for start in range(0, len(order), 16):
            batch = [train_graphs[j] for j in order[start:start + 16]]
            batch_losses.append(sum(x['nodes'] * x['zero_predictor_loss'] for x in batch) / sum(x['nodes'] for x in batch))
        zero_history.append({'epoch': epoch, 'all_zero_batched_train_loss': float(np.mean(batch_losses)),
                             'logged_train_loss': float(logged['train_loss'])})
    (out / 'zero_baseline_training.json').write_text(json.dumps(zero_history, indent=2) + '\n')

    # A sample in each domain: identical GAT inputs must have identical outputs.
    input_conflicts = []
    for domain, example in domain_examples.items():
        selected = [r for r in split['train'] if r['domain'] == domain and r['sample_id'] == example['sample_id']]
        inputs = defaultdict(list)
        for row in selected:
            v = featurize_variant(row, 'data', 'gatv2')
            fingerprint = hashlib.sha256(v.node_features.numpy().tobytes() + v.edge_index.numpy().tobytes()).hexdigest()
            inputs[fingerprint].append((row, v.sens.numpy(), v.nec.numpy()))
        conflicts = []
        for items in inputs.values():
            r0, s0, n0 = items[0]
            for row, s, n in items[1:]:
                if np.any(s != s0) or np.any(n != n0):
                    conflicts.append({'path_a': r0['tree_with_scores_path'], 'path_b': row['tree_with_scores_path'],
                        'goal_a': r0['goal_id'], 'goal_b': row['goal_id'], 'nodes': len(n),
                        'different_sens_nodes': int((s != s0).sum()), 'different_nec_nodes': int((n != n0).sum()),
                        'different_critical_membership_nodes': int(((n >= 7) != (n0 >= 7)).sum())})
        input_conflicts.append({'domain': domain, 'sample_id': example['sample_id'],
            'variants_checked': len(selected), 'distinct_graph_inputs': len(inputs),
            'conflicts_with_group_representative': conflicts})
    (out / 'input_conflicts.json').write_text(json.dumps(input_conflicts, indent=2) + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report['diagnosis_device'] = str(device)
    model = build_model('runs/minim/model.pt', device)
    model.eval()
    report['checkpoint_parameters_finite'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    report['checkpoint_parameter_count'] = sum(p.numel() for p in model.parameters())
    s_all, n_all = [], []
    counts = Counter()
    metric_sums = defaultdict(lambda: Counter())
    critical_total = interactive_total = no_critical = no_interactive = 0
    from random import Random
    rng = Random(0)
    for i, row in enumerate(split['test'], 1):
        v = featurize_variant(row, 'data', 'gatv2')
        nodes = json.loads(Path(v.tree_path).read_text())['nodes']
        with torch.no_grad():
            sp, npred = score_variant(model, v, device)
        s, n = sp.cpu().numpy(), npred.cpu().numpy()
        s_all.append(s)
        n_all.append(n)
        actions = np.where(n < 1, 'R', np.where(s >= 5, 'A', 'K')).tolist()
        counts.update(actions)
        gs, gn = v.sens.tolist(), v.nec.tolist()
        oracle = ['R' if nn < 1 else ('A' if ss >= 5 else 'K') for ss, nn in zip(gs, gn)]
        policies = {'checkpoint': actions, 'all_remove': ['R'] * len(nodes), 'oracle': oracle,
                    'random_budget': random_budget_actions(len(nodes), .2, rng)}
        critical_total += int((v.nec >= 7).sum())
        inter = [j for j, x in enumerate(nodes) if gn[j] >= 7 and is_interactive(x)]
        interactive_total += len(inter)
        no_critical += not any(nn >= 7 for nn in gn)
        no_interactive += not inter
        for name, acts in policies.items():
            stats = metric_sums[name]
            stats['tcnp_sum'] += compute_tcnp(nodes, gn, acts)
            stats['tcnp_i_sum'] += compute_tcnp_i(nodes, gn, acts)
            stats['leak_num'] += compute_tisl_numerator(nodes, gs, gn, acts)
            stats['leak_den'] += compute_tisl_denominator(nodes, gs, gn)
            stats['critical_kept'] += sum(nn >= 7 and act != 'R' for nn, act in zip(gn, acts))
            stats['interactive_kept'] += sum(acts[j] != 'R' for j in inter)
        if i % 100 == 0 or i == len(split['test']):
            print(f"inference {i}/{len(split['test'])}", flush=True)
    report['test_action_counts'] = dict(counts)
    report['test_no_critical_graphs'] = no_critical
    report['test_no_critical_interactive_graphs'] = no_interactive
    report['test_critical_nodes'] = critical_total
    report['test_critical_interactive_nodes'] = interactive_total
    report['test_metrics'] = {name: {
        'macro_tcnp': x['tcnp_sum'] / len(split['test']),
        'macro_tcnp_i': x['tcnp_i_sum'] / len(split['test']),
        'tisl': x['leak_num'] / x['leak_den'] if x['leak_den'] else 0,
        'micro_critical_recall': x['critical_kept'] / critical_total,
        'micro_critical_interactive_recall': x['interactive_kept'] / interactive_total,
    } for name, x in metric_sums.items()}
    report['test_predictions'] = {}
    for name, values in [('sens', np.concatenate(s_all)), ('nec', np.concatenate(n_all))]:
        report['test_predictions'][name] = {'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'quantiles': dict(zip(['p01', 'p50', 'p99'], np.quantile(values, [.01, .5, .99]).tolist()))}
    # Compare gradients at the saved logits, using the same labels and no updates.
    for domain in ['gmail', 'reddit', 'shopping']:
        candidates = [x for x in per_graph if x['split'] == 'train' and x['domain'] == domain]
        best = max(candidates, key=lambda x: x['critical_nodes'] / x['nodes'])
        row = next(r for r in split['train'] if r['tree_with_scores_path'] == best['path'])
        v = featurize_variant(row, 'data', 'gatv2')
        with torch.no_grad():
            x = v.node_features.to(device)
            for layer in model.gat_layers:
                x = F.elu(layer(x, v.edge_index.to(device)))
        for head_name, head, target in [('sens', model.sens_head, v.sens), ('nec', model.nec_head, v.nec)]:
            logits = head(x).detach().requires_grad_(True)
            target = target.to(device)
            probs = logits.softmax(-1)
            pred = (probs * model.score_bins).sum(-1)
            weight = torch.ones_like(target) if head_name == 'sens' else 1 + v.sens.to(device) / 10 * (1 - target / 10)
            l1 = (weight * (pred - target).abs()).mean()
            ce = (weight * F.cross_entropy(logits, target.long(), reduction='none')).mean()
            grad_l1 = torch.autograd.grad(l1, logits, retain_graph=True)[0]
            grad_ce = torch.autograd.grad(ce, logits)[0]
            pos = target > 0
            report['gradient_checks'].append({'domain': domain, 'head': head_name, 'path': best['path'],
                'nodes': len(target), 'positive_nodes': int(pos.sum()),
                'bin0_probability_mean': float(probs[:, 0].mean().detach()),
                'positive_logit_gradient_l1_abs_mean': float(grad_l1[pos].abs().mean()),
                'positive_logit_gradient_ce_abs_mean': float(grad_ce[pos].abs().mean()),
                'l1_over_ce_positive_gradient_ratio': float(grad_l1[pos].abs().mean() / grad_ce[pos].abs().mean())})
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [x['epoch'] for x in zero_history]
    axes[0].plot(epochs, [x['logged_train_loss'] for x in zero_history], 'o-', label='Logged train loss')
    axes[0].plot(epochs, [x['all_zero_batched_train_loss'] for x in zero_history], 'x--', label='All-zero predictor, same batches')
    axes[0].set(title='After epoch 1, losses match to 4 decimals', xlabel='Epoch', ylabel='Batch-mean CI L1')
    axes[1].plot(epochs, [float(x['val_loss']) for x in history], 'o-', label='Logged validation loss')
    axes[1].axhline(report['splits']['val']['zero_predictor_loss_graph_mean'], linestyle='--', color='tab:orange', label='All-zero predictor')
    axes[1].set(title='Validation equals the all-zero baseline', xlabel='Epoch', ylabel='Graph-mean CI L1')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'zero_baseline_loss.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ['test_action_counts', 'test_metrics', 'test_predictions', 'gradient_checks']}, indent=2), flush=True)
    print(f'Saved {out}/audit.json and per_graph.csv', flush=True)


if __name__ == '__main__':
    main()

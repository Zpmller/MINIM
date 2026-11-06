#!/usr/bin/env python3
"""Controlled small-scale MINIM repair study with validation-only selection."""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_metrics import (compute_tcnp, compute_tcnp_i, compute_tisl_denominator,
                             compute_tisl_numerator, is_interactive)
from run_minim_inference import build_model
from train_minim_scorer import (balanced_score_loss, build_scorer, ci_loss,
    collate_variants, encode_task, featurize_variant, label_balance_weights,
    score_variant, set_seed)

ARMS = {
    'legacy_ci_l1': ('legacy', 'ci_l1'),
    'unique_ci_l1': ('unique', 'ci_l1'),
    'unique_balanced_l1': ('unique', 'balanced_l1'),
    'unique_balanced_ce': ('unique', 'balanced_ce'),
    'legacy_boosted_l1': ('legacy', 'boosted_l1'),
    'unique_balanced_ce_calibrated': ('unique', 'balanced_ce'),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare_split(train_trees):
    original = json.loads((ROOT / 'runs/minim/split.json').read_text())
    result = {}
    rng = random.Random(20261031)
    for name in ('train', 'val', 'test'):
        selected = []
        for domain in ('gmail', 'reddit', 'shopping'):
            groups = sorted({r['sample_id'] for r in original[name] if r['domain'] == domain})
            rng.shuffle(groups)
            chosen = set(groups[:train_trees if name == 'train' else 1])
            selected.extend(r for r in original[name] if r['domain'] == domain and r['sample_id'] in chosen
                and Path(r['tree_with_scores_path']).parent.name.startswith(('variant_balanced_', 'variant_reading_')))
        result[name] = selected
    sets = {k: {(r['domain'], r['sample_id']) for r in v} for k, v in result.items()}
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    return result


def evaluate(model, variants, nodes_by_path, device, scorer=score_variant):
    stats = Counter()
    all_n, all_s = [], []
    with torch.no_grad():
        model.eval()
        for v in variants:
            sh, nh = scorer(model, v, device)
            s, n = sh.cpu().numpy(), nh.cpu().numpy()
            gs, gn = v.sens.numpy(), v.nec.numpy()
            nodes = nodes_by_path[v.tree_path]
            actions = np.where(n < 1, 'R', np.where(s >= 5, 'A', 'K'))
            critical = gn >= 7
            interactive = np.array([is_interactive(x) for x in nodes]) & critical
            stats['graphs'] += 1
            stats['nodes'] += len(n)
            stats['tcnp_sum'] += compute_tcnp(nodes, gn, actions)
            stats['tcnp_i_sum'] += compute_tcnp_i(nodes, gn, actions)
            stats['critical'] += int(critical.sum())
            stats['critical_kept'] += int((critical & (actions != 'R')).sum())
            stats['interactive'] += int(interactive.sum())
            stats['interactive_kept'] += int((interactive & (actions != 'R')).sum())
            stats['leak_num'] += compute_tisl_numerator(nodes, gs, gn, actions)
            stats['leak_den'] += compute_tisl_denominator(nodes, gs, gn)
            for action in ('K', 'A', 'R'):
                stats[action] += int((actions == action).sum())
            stats['sens_tp'] += int(((gs >= 5) & (s >= 5)).sum())
            stats['sens_pos'] += int((gs >= 5).sum())
            stats['sens_pred_pos'] += int((s >= 5).sum())
            stats['hard_conflict'] += int(((gs >= 5) & critical).sum())
            stats['hard_conflict_abstract'] += int(((gs >= 5) & critical & (actions == 'A')).sum())
            stats['s_abs_error'] += float(np.abs(gs - s).sum())
            stats['n_abs_error'] += float(np.abs(gn - n).sum())
            all_n.append(n)
            all_s.append(s)
    n, s = np.concatenate(all_n), np.concatenate(all_s)
    def ratio(a, b):
        return float(stats[a] / stats[b]) if stats[b] else 0.
    result = {'tcnp': ratio('tcnp_sum', 'graphs'), 'tcnp_i': ratio('tcnp_i_sum', 'graphs'),
        'micro_tcnp': ratio('critical_kept', 'critical'), 'micro_tcnp_i': ratio('interactive_kept', 'interactive'),
        'tisl': ratio('leak_num', 'leak_den'),
        'keep_rate': ratio('K', 'nodes'), 'abstract_rate': ratio('A', 'nodes'), 'remove_rate': ratio('R', 'nodes'),
        'sensitivity_precision': ratio('sens_tp', 'sens_pred_pos'), 'sensitivity_recall': ratio('sens_tp', 'sens_pos'),
        'hard_conflict_abstract_recall': ratio('hard_conflict_abstract', 'hard_conflict'),
        'sens_mae': ratio('s_abs_error', 'nodes'), 'nec_mae': ratio('n_abs_error', 'nodes'),
        'sens_mean': float(s.mean()), 'sens_p99': float(np.quantile(s, .99)),
        'nec_mean': float(n.mean()), 'nec_p99': float(np.quantile(n, .99)),
        'critical_nodes': stats['critical'], 'critical_interactive_nodes': stats['interactive'], 'nodes': stats['nodes']}
    result['selection_score'] = result['micro_tcnp'] - .5 * result['tisl']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'test', 'export'), default='train')
    parser.add_argument('--output_dir', default='runs/pilot_fix')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_trees_per_domain', type=int, default=2)
    parser.add_argument('--gat_hidden', type=int, default=256)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument('--resume', action='store_true', help='Append arms without changing the saved split/protocol')
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'export':
        selected = json.loads((out / 'selected_model.json').read_text())
        results_dir = ROOT / 'results' / out.name
        results_dir.mkdir(parents=True, exist_ok=True)
        common = ['--data_root', 'data', '--split_path', str(out / 'split.json'),
                  '--checkpoint', selected['checkpoint'], '--device', str(device), '--seed', '0']
        for entry in ('table1_main_results', 'table3_task_semantics', 'table4_abstraction_ablation',
                      'table10_domain_specific', 'figure2_threshold_sweep', 'figure3_policy_distribution'):
            with (ROOT / 'logs' / f'{out.name}_{entry}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', f'experiment_{entry}.py', *common,
                                '--results_dir', str(results_dir)], check=True, stdout=log, stderr=subprocess.STDOUT)
        with (ROOT / 'logs' / f'{out.name}_inference.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'run_minim_inference.py', *common,
                            '--predict_split', 'test', '--output_dir', str(out / 'selected_inference')],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        return
    if args.stage == 'train' and not args.resume:
        split = prepare_split(args.train_trees_per_domain)
        config = vars(args).copy()
        config.update(lr=3e-4, weight_decay=1e-5, gat_heads=4, model_type='gatv2',
            selection_rule='max(validation micro TCNP - 0.5 * validation TISL); earliest tie',
            split_seed=20261031,
            original_checkpoint_sha256=hashlib.sha256((ROOT / 'runs/minim/model.pt').read_bytes()).hexdigest())
        config['task_vocab'] = {goal: i for i, goal in enumerate(sorted({r['goal_id'] for r in split['train']}))}
        write_json(out / 'pilot_config.json', config)
        write_json(out / 'split.json', split)
    else:
        config = json.loads((out / 'pilot_config.json').read_text())
        split = json.loads((out / 'split.json').read_text())
        if args.stage == 'train':
            for key in ('seed', 'epochs', 'gat_hidden', 'batch_size', 'train_trees_per_domain'):
                if getattr(args, key) != config[key]:
                    raise ValueError(f'Resumed arms must retain {key}={config[key]}')
            config['arms'] = list(dict.fromkeys(config['arms'] + args.arms))
            write_json(out / 'pilot_config.json', config)
    vocab = config['task_vocab']
    assert not {r['goal_id'] for name in split for r in split[name]} - set(vocab)
    names = ('train', 'val') if args.stage == 'train' else ('train', 'val', 'test')
    legacy, unique, nodes_by_path = {}, {}, {}
    for name in names:
        legacy[name] = [featurize_variant(row, 'data', 'gatv2') for row in split[name]]
        unique[name] = [encode_task(v, vocab) for v in legacy[name]]
        for v in legacy[name]:
            nodes_by_path[v.tree_path] = json.loads(Path(v.tree_path).read_text())['nodes']
        print(f'{name}: {len(legacy[name])} variants, {sum(len(v.nec) for v in legacy[name])} nodes', flush=True)
    if args.stage == 'test':
        summary = json.loads((out / 'validation_summary.json').read_text())
        all_rows = []
        for arm in config['arms']:
            model = build_model(str(out / arm / 'model.pt'), device)
            data = unique if model.task_vocab is not None else legacy
            scores = evaluate(model, data['test'], nodes_by_path, device)
            summary[arm]['test'] = scores
            scores['arm'] = arm
            all_rows.append(scores)
            print(f'TEST {arm}: {scores}', flush=True)
        old = build_model('runs/minim/model.pt', device)
        summary['original_failed_checkpoint'] = {'test': evaluate(old, legacy['test'], nodes_by_path, device)}
        write_json(out / 'summary.json', summary)
        with (out / 'test_comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        selected = max(config['arms'], key=lambda arm: summary[arm]['validation']['selection_score'])
        write_json(out / 'selected_model.json', {'arm': selected, 'checkpoint': str(out / selected / 'model.pt'),
            'selected_only_on_validation': True, 'test_metrics': summary[selected]['test']})
        return

    class_weights = label_balance_weights(legacy['train'])
    write_json(out / 'class_weights.json', [w.tolist() for w in class_weights])
    summary, history = {}, []
    if args.resume:
        summary = json.loads((out / 'validation_summary.json').read_text())
        history = list(csv.DictReader((out / 'history.csv').open()))
        for row in history:
            for key in row:
                if key != 'arm':
                    row[key] = int(row[key]) if key == 'epoch' else float(row[key])
    for arm in (args.arms if args.resume else config['arms']):
        if args.resume and arm in summary:
            raise ValueError(f'Arm {arm} already finished; use a new output directory to repeat it')
        encoding, loss_type = ARMS[arm]
        data = unique if encoding == 'unique' else legacy
        set_seed(config['seed'])
        model = build_scorer('gatv2', gat_hidden=config['gat_hidden'], gat_heads=4).to(device)
        model.task_vocab = vocab if encoding == 'unique' else None
        prior_correction = arm.endswith('_calibrated')
        if prior_correction:
            model.set_class_prior_correction(class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        rng = random.Random(config['seed'])
        run = out / arm
        run.mkdir(exist_ok=True)
        best_score = -float('inf')
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            order = list(range(len(data['train'])))
            rng.shuffle(order)
            losses, gradients = [], []
            for start in range(0, len(order), config['batch_size']):
                batch = collate_variants([data['train'][i] for i in order[start:start + config['batch_size']]], device)
                if loss_type in ('ci_l1', 'boosted_l1'):
                    sh, nh = model(batch['node_features'], batch['edge_index'])
                    boosts = (2, 2, 1) if loss_type == 'boosted_l1' else (0, 0, 0)
                    loss, _, _ = ci_loss(batch['sens'], batch['nec'], sh, nh,
                        sens_high_boost=boosts[0], nec_low_boost=boosts[1], nec_high_boost=boosts[2])
                else:
                    sl, nl = model.forward_logits(batch['node_features'], batch['edge_index'])
                    loss, _, _ = balanced_score_loss(batch['sens'], batch['nec'], sl, nl, class_weights, loss_type)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite training loss: {arm}, epoch {epoch}')
                optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                if not torch.isfinite(grad):
                    raise RuntimeError('Non-finite gradient')
                optimizer.step()
                losses.append(float(loss.detach()))
                gradients.append(float(grad))
            val = evaluate(model, data['val'], nodes_by_path, device)
            row = {'arm': arm, 'epoch': epoch, 'train_loss': float(np.mean(losses)),
                   'gradient_norm': float(np.mean(gradients)), **val}
            history.append(row)
            if val['selection_score'] > best_score:
                best_score = val['selection_score']
                run_config = {**config, 'task_vocab': model.task_vocab, 'task_encoding': encoding,
                              'loss_type': loss_type, 'selected_epoch': epoch,
                              'prior_correction': prior_correction,
                              'class_weights': [w.tolist() for w in class_weights]}
                torch.save({'model_type': 'gatv2', 'model_state': model.state_dict(), 'config': run_config}, run / 'model.pt')
                write_json(run / 'train_config.json', run_config)
                write_json(run / 'split.json', split)
                best_val = val
            if epoch % 5 == 0 or epoch == 1:
                print(f'{arm} {epoch}/{config["epochs"]}: loss={row["train_loss"]:.4f}, '
                      f'val micro_TCNP={val["micro_tcnp"]:.4f}, TISL={val["tisl"]:.4f}, '
                      f'K/A/R={val["keep_rate"]:.3f}/{val["abstract_rate"]:.3f}/{val["remove_rate"]:.3f}', flush=True)
        restored = build_model(str(run / 'model.pt'), device)
        best_train = evaluate(restored, data['train'], nodes_by_path, device)
        best_val_loaded = evaluate(restored, data['val'], nodes_by_path, device)
        assert abs(best_val_loaded['selection_score'] - best_val['selection_score']) < 1e-6
        summary[arm] = {'validation': best_val_loaded, 'training': best_train,
                        'selected_epoch': json.loads((run / 'train_config.json').read_text())['selected_epoch']}
        write_json(out / 'validation_summary.json', summary)
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f'BEST {arm}: {summary[arm]}', flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in config['arms']:
        rows = [r for r in history if r['arm'] == arm]
        for ax, metric in zip(axes, ('micro_tcnp', 'tisl')):
            ax.plot([r['epoch'] for r in rows], [r[metric] for r in rows], label=arm)
            ax.set(xlabel='Epoch', ylabel=metric, title=f'Validation {metric}')
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'validation_curves.png', dpi=180)
    plt.close(fig)
    print('Training complete; test labels/predictions have not been evaluated.', flush=True)


if __name__ == '__main__':
    main()

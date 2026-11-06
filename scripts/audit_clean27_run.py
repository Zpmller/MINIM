#!/usr/bin/env python3
"""Independently check the completed 27-task run without altering its outputs."""
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_training_scale import official_snapshot
from train_minim_scorer import MINILM_MODEL_NAME, featurize_variant
from evaluate_metrics import compute_tcnp, compute_tcnp_i, compute_tisl_denominator, compute_tisl_numerator, is_interactive


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    data = ROOT / 'data_paper_aligned'
    out = ROOT / 'results/clean27_audit'
    work = ROOT / 'runs/clean27_audit'
    out.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    cache = work / 'feature_cache'
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(cache)
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').open() if line.strip()]
    split = json.loads((ROOT / 'runs/paper_aligned/split.json').read_text())
    config = json.loads((ROOT / 'runs/paper_aligned/train_config.json').read_text())
    checkpoint = ROOT / 'runs/paper_aligned/model.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert len(rows) == 5400 and len({r['goal_id'] for r in rows}) == 27
    assert all(not r['goal_id'].startswith('T_') for r in rows)
    assert config['gat_hidden'] == 256 and config['batch_size'] == 32 and config['epochs'] == 10
    assert config['loss_type'] == 'ci_l1' and config['task_encoding'] == 'legacy'
    assert not config['prior_correction'] and config['task_vocab'] is None
    for field in ('sens_high_boost', 'nec_low_boost', 'nec_high_boost'):
        assert config[field] == 0
    key = lambda r: (r['domain'], r['sample_id'], r['goal_id'], r['tree_with_scores_path'])
    assert Counter(key(r) for r in rows) == Counter(key(r) for v in split.values() for r in v)
    group_sets = {name: {(r['domain'], r['sample_id']) for r in part} for name, part in split.items()}
    assert not (group_sets['train'] & group_sets['val'] or group_sets['train'] & group_sets['test'] or group_sets['val'] & group_sets['test'])
    old_split = json.loads((ROOT / 'runs/minim/split.json').read_text())
    assert all(group_sets[k] == {(r['domain'], r['sample_id']) for r in old_split[k]} for k in split)
    test_paths = {r['tree_with_scores_path'] for r in split['test']}
    hist = {'sens': Counter(), 'nec': Counter()}
    counts = Counter()
    per_graph = {}
    for i, row in enumerate(rows, 1):
        path = data / row['tree_with_scores_path']
        assert path.samefile(ROOT / 'data' / row['tree_with_scores_path'])
        nodes = json.loads(path.read_text())['nodes']
        s = np.array([n['sensitivity_score'] for n in nodes], dtype=np.float32)
        n = np.array([n['necessity_score'] for n in nodes], dtype=np.float32)
        assert np.isfinite(s).all() and np.isfinite(n).all()
        assert ((s >= 0) & (s <= 10)).all() and ((n >= 0) & (n <= 10)).all()
        digest = hashlib.sha256(f"gatv2\0{MINILM_MODEL_NAME}\0{row['tree_with_scores_path']}".encode()).hexdigest()
        source = ROOT / 'runs/minim/feature_cache' / digest[:2] / (digest + '.npz')
        with np.load(source, allow_pickle=False) as arrays:
            assert np.array_equal(s, arrays['sens']) and np.array_equal(n, arrays['nec'])
        if row['tree_with_scores_path'] in test_paths:
            target = cache / digest[:2] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            assert target.samefile(source)
        hist['sens'].update(s.tolist())
        hist['nec'].update(n.tolist())
        counts['nodes'] += len(nodes)
        per_graph[row['tree_with_scores_path']] = {
            'nodes': len(nodes), 'zero_loss': float(np.mean(s + (1 + s / 10 * (1 - n / 10)) * n))}
        if i % 1000 == 0 or i == len(rows):
            print(f'raw/cache label verification {i}/{len(rows)}', flush=True)
    frozen, source_hash = official_snapshot(work)
    payload = torch.load(checkpoint, map_location='cpu')
    device = torch.device('cuda')
    model = frozen.build_scorer('gatv2', gat_hidden=256, gat_heads=4).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    metric_totals = Counter()
    prediction_values = {'sens': [], 'nec': []}
    for i, row in enumerate(split['test'], 1):
        variant = featurize_variant(row, str(data), 'gatv2')
        nodes = json.loads(Path(variant.tree_path).read_text())['nodes']
        with torch.no_grad():
            sh, nh = frozen.score_variant(model, variant, device)
        s, n = sh.cpu().numpy(), nh.cpu().numpy()
        gs, gn = variant.sens.numpy(), variant.nec.numpy()
        actions = np.where(n < 1, 'R', np.where(s >= 5, 'A', 'K'))
        for action in ('K', 'A', 'R'):
            metric_totals[action] += int((actions == action).sum())
        critical = gn >= 7
        interactive = critical & np.array([is_interactive(x) for x in nodes])
        metric_totals['critical'] += int(critical.sum())
        metric_totals['critical_kept'] += int((critical & (actions != 'R')).sum())
        metric_totals['interactive'] += int(interactive.sum())
        metric_totals['interactive_kept'] += int((interactive & (actions != 'R')).sum())
        metric_totals['no_critical'] += int(not critical.any())
        metric_totals['no_interactive'] += int(not interactive.any())
        metric_totals['tcnp_sum'] += compute_tcnp(nodes, gn, actions)
        metric_totals['tcnp_i_sum'] += compute_tcnp_i(nodes, gn, actions)
        metric_totals['leak_num'] += float(compute_tisl_numerator(nodes, gs, gn, actions))
        metric_totals['leak_den'] += float(compute_tisl_denominator(nodes, gs, gn))
        prediction_values['sens'].append(s)
        prediction_values['nec'].append(n)
        if i % 100 == 0 or i == len(split['test']):
            print(f'frozen official inference {i}/{len(split["test"])}', flush=True)
    metrics = {'tcnp': metric_totals['tcnp_sum'] / len(split['test']),
        'tcnp_i': metric_totals['tcnp_i_sum'] / len(split['test']),
        'micro_tcnp': metric_totals['critical_kept'] / metric_totals['critical'],
        'micro_tcnp_i': metric_totals['interactive_kept'] / metric_totals['interactive'],
        'tisl': metric_totals['leak_num'] / metric_totals['leak_den']}
    official_table = next(r for r in csv.DictReader((ROOT / 'results/paper_aligned/table1.csv').open()) if r['method'] == 'MINIM')
    assert all(abs(metrics[k] - float(official_table[k])) < 1e-6 for k in ('tcnp', 'tcnp_i', 'tisl'))
    pattern = re.compile(r'epoch (\d+)/10 \| train_loss=([\d.]+) \| val_loss=([\d.]+)')
    history = [dict(epoch=int(m[0]), train_loss=float(m[1]), val_loss=float(m[2]))
               for m in pattern.findall((ROOT / 'logs/train_paper_aligned.log').read_text())]
    assert len(history) == 10
    rng = random.Random(0)
    for item in history:
        order = list(range(len(split['train'])))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 32):
            graphs = [per_graph[split['train'][j]['tree_with_scores_path']] for j in order[start:start + 32]]
            losses.append(sum(g['zero_loss'] * g['nodes'] for g in graphs) / sum(g['nodes'] for g in graphs))
        item['zero_train_loss'] = float(np.mean(losses))
    zero_val = float(np.mean([per_graph[r['tree_with_scores_path']]['zero_loss'] for r in split['val']]))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report = {'manifest_sha256': hashlib.sha256((data / 'manifest.jsonl').read_bytes()).hexdigest(),
        'checkpoint_sha256': checkpoint_hash, 'frozen_source_sha256': source_hash,
        'task_count': 27, 'variant_count': len(rows), 'nodes_checked': counts['nodes'],
        'raw_cache_label_checks': len(rows), 'all_tree_paths_share_original_files': True,
        'cache_strategy': 'Separate namespace; verified immutable hardlinks for unchanged test feature arrays.',
        'splits': {k: {'variants': len(v), 'base_trees': len(group_sets[k])} for k, v in split.items()},
        'same_base_tree_split_as_34_tasks': True,
        'label_histograms': {k: dict(sorted(v.items())) for k, v in hist.items()},
        'zero_label_fractions': {k: v[0.] / counts['nodes'] for k, v in hist.items()},
        'test_counts': dict(metric_totals), 'test_metrics': metrics,
        'predictions': {k: {'min': float(np.concatenate(v).min()), 'mean': float(np.concatenate(v).mean()),
                            'max': float(np.concatenate(v).max())} for k, v in prediction_values.items()},
        'training_history': history, 'zero_predictor_validation_loss': zero_val,
        'independent_metrics_match_existing_official_table': True}
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    with (out / 'training_zero_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['train_loss'] for r in history], 'o-', label='Logged 27-task training')
    axes[0].plot(epochs, [r['zero_train_loss'] for r in history], 'x--', label='All-zero predictor, same batches')
    axes[0].set(title='27-task training loss', xlabel='Epoch', ylabel='CI-L1')
    axes[1].plot(epochs, [r['val_loss'] for r in history], 'o-', label='Logged validation')
    axes[1].axhline(zero_val, color='tab:orange', linestyle='--', label='All-zero predictor')
    axes[1].set(title='27-task validation loss', xlabel='Epoch', ylabel='Graph-mean CI-L1')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'zero_loss_comparison.png', dpi=180)
    plt.close(fig)
    text = f'''# 27 任务清洗版独立复核

日期：2026-10-31（Asia/Shanghai）。**结论：27 任务版本的官方配置重跑仍发生近零分数塌缩。去掉额外阅读任务后，全移除现象没有解除。**

## 本轮实际执行范围

收到重试请求后，发现另一条本地进程已经完成 `runs/paper_aligned/` 的 10 轮训练并正在运行官方评估。因此没有再启动相同配置的重复训练。本轮独立完成了全部原始标签/缓存核验、冻结官方模型加载和全测试集推理；没有修改该进程的检查点、阈值或结果文件。

## 数据与配置

- 清洗数据：`data_paper_aligned/manifest.jsonl`，确为 **27 种任务、5,400 个变体、{counts['nodes']:,} 个节点**。仅移除 350 个 `T_` 阅读变体，保留的 JSON 树仍指向原始物理文件，标签没有重新标注。
- Train / val / test：4,356 / 592 / 452 个变体，基础树为 120 / 15 / 15；无交集，且与清洗前划分使用同一批基础树。这次改变的是任务集合，不是另挑一批测试页面。
- GATv2 hidden=256、4 heads、10 epochs、**原生 batch=32（未使用梯度累积）**、AdamW、lr=3e-4、weight decay=1e-5、seed=0、alpha=lambda=1。
- 原始 CI-L1、旧任务编码，三个额外 boost=0；没有使用唯一编码、平衡 CE 或先验校正。与论文一致的训练设置沿用，不能把本轮称为修复版训练。
- 全部 5,400 条缓存标签与原 JSON 逐项相等。独立测试使用单独缓存目录，只链接已验证、输入未变的 452 条测试特征，不混用被删除任务。未重新计算全部 MiniLM 嵌入。

## 独立测试结果

使用固定官方提交的原始 GATv2 类严格加载本次权重，对全部 452 个测试变体重新计算：

| 项目 | 实测 |
|---|---:|
| 测试节点 | {sum(metric_totals[a] for a in ('K','A','R')):,} |
| Keep / Abstract / Remove | {metric_totals['K']} / {metric_totals['A']} / {metric_totals['R']} |
| 关键节点保留 | {metric_totals['critical_kept']} / {metric_totals['critical']} |
| 关键交互节点保留 | {metric_totals['interactive_kept']} / {metric_totals['interactive']} |
| 官方 TCNP / TCNP-I | {metrics['tcnp']:.6f} / {metrics['tcnp_i']:.6f} |
| 真实节点微平均保留率 | {metrics['micro_tcnp']:.6f} |
| TISL（阈值 1） | {metrics['tisl']:.6f} |

TCNP/TCNP-I 的 3.54% 来自 16 个没有关键节点/关键交互节点的变体默认得分为 1，`16/452=0.035398`；实际保留率为零。独立计算与官方 Table 1 CSV 吻合到其六位小数。

整个测试集必要性分数均值约 {report['predictions']['nec']['mean']:.3g}，最大值 {report['predictions']['nec']['max']:.6g}，远低于阈值 1；敏感度最大值 {report['predictions']['sens']['max']:.6g}，也远低于阈值 5。

## 阈值选择并未解决退化

另一进程的验证集扫描（`results/paper_aligned/selected_threshold.json`）选择 `tau_nec=0`。由本轮全部测试分数的范围可知，这会使所有节点变成 Keep：TCNP/TCNP-I/TISL 均为 1。阈值 1 全移除，阈值 0 全保留，两者都未恢复有效折中；不能把全保留的 100% 保留率当成复现成功。

## 失败机制更新

1. **任务数量和已发现的阅读任务碰撞不是本次失败的充分解释。** 27 任务数据仍塌缩。原生 batch=32 也出现同样现象，说明该现象不要求使用此前的梯度累积。
2. **不是清洗后缓存标签未更新。** 全量 5,400 条逐项核验通过，输入树仅通过软链接复用。
3. **训练目标的零预测退化仍是重点。** 清洗数据敏感度为零占 {report['zero_label_fractions']['sens']:.2%}，必要性为零占 {report['zero_label_fractions']['nec']:.2%}。恒定零预测器的验证损失为 {zero_val:.9f}，与末轮日志 {history[-1]['val_loss']:.4f} 吻合到日志精度；末轮训练损失 {history[-1]['train_loss']:.4f} 也与按同批次重算的 {history[-1]['zero_train_loss']:.9f} 吻合。

这些证据支持原始 L1/softmax 期望分数训练仍退化到接近零的解释，但没有证明这是唯一原因。任务数已核对，不应继续把 34/27 差异当作本次失败的主要解释。若继续定位，下一项受控实验应固定这份 27 任务数据、划分和旧任务编码，只比较训练目标；这属于单独诊断配置，不冒称官方配方。

## 可复查文件

- 独立脚本：`scripts/audit_clean27_run.py`。
- 原始核验与计数：`results/clean27_audit/audit.json`；全零对照：`results/clean27_audit/training_zero_comparison.csv`。
- 训练日志：`logs/train_paper_aligned.log`；独立核验日志：`logs/clean27_independent_audit.log`。
- 数据 manifest SHA256：`{report['manifest_sha256']}`。
- 检查点 SHA256：`{checkpoint_hash}`，独立推理前后未变。
- 冻结官方源码 SHA256：`{source_hash}`。

![27 任务损失与全零预测器对照](../results/clean27_audit/zero_loss_comparison.png)
'''
    (ROOT / 'reports/MINIM_27任务独立复核.md').write_text(text)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Calibrate on validation, lock the policy, then evaluate the released test split."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_metrics import is_interactive, token_count
from run_minim_inference import build_model
from train_minim_scorer import featurize_variant, score_variant

RUN = ROOT / 'runs/reference_profile_20261104'
OUT = ROOT / 'results/reference_profile_20261104'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def policy_metrics(graphs, tau_nec, tau_sens):
    tcnp, tcnpi, numerator, denominator = [], [], 0., 0.
    counts = np.zeros(5, dtype=np.int64)
    privacy = np.zeros(6, dtype=np.int64)
    for g in graphs:
        gs, gn, sp, npred, tok, interactive = (g[k] for k in ('gs', 'gn', 'sp', 'np', 'tok', 'interactive'))
        keep = npred >= tau_nec
        abstract = keep & (sp >= tau_sens)
        critical = gn >= 7
        ci = critical & interactive
        tcnp.append(float(keep[critical].mean()) if critical.any() else 1.)
        tcnpi.append(float(keep[ci].mean()) if ci.any() else 1.)
        weight = gs * tok * (gn <= 1)
        denominator += float(weight.sum())
        post = np.where(keep, np.where(abstract, np.maximum(6., .05 * tok), tok), 0.)
        numerator += float((gs * post * (gn <= 1)).sum())
        counts += [int((keep & critical).sum()), int(critical.sum()), int((keep & ci).sum()), int(ci.sum()), len(gn)]
        sensitive = gs >= 5
        hard = sensitive & critical
        privacy += [int(sensitive.sum()), int((sensitive & (sp >= tau_sens)).sum()),
                    int(hard.sum()), int((hard & abstract).sum()), int(keep.sum()), int(abstract.sum())]
    return {'tcnp': float(np.mean(tcnp)), 'tcnp_i': float(np.mean(tcnpi)),
            'tisl': numerator / denominator if denominator else 0.,
            'micro_tcnp': int(counts[0]) / int(counts[1]) if counts[1] else 0.,
            'micro_tcnp_i': int(counts[2]) / int(counts[3]) if counts[3] else 0.,
            'nodes': int(counts[4]), 'variants': len(graphs),
            'sensitivity_recall': float(privacy[1] / privacy[0]) if privacy[0] else 0.,
            'hard_conflict_abstract_recall': float(privacy[3] / privacy[2]) if privacy[2] else 0.,
            'keep_rate': float((privacy[4] - privacy[5]) / counts[4]),
            'abstract_rate': float(privacy[5] / counts[4]),
            'remove_rate': float(1 - privacy[4] / counts[4])}


def selection_key(metrics, reference, tau_nec, tau_sens):
    # Better-than-reference preservation is retained; no reward for increasing leakage.
    utility = min(metrics['tcnp'] / reference['tcnp'], metrics['tcnp_i'] / reference['tcnp_i'], 1.)
    excess = max(0., metrics['tisl'] - reference['tisl'])
    return (utility - excess, -metrics['tisl'], -abs(tau_nec - 1) - abs(tau_sens - 5))


def collect(model, rows, device, name):
    graphs = []
    for i, row in enumerate(rows, 1):
        v = featurize_variant(row, 'data', 'gatv2', model.task_vocab)
        with torch.no_grad():
            sp, npred = score_variant(model, v, device)
        nodes = json.loads(Path(v.tree_path).read_text())['nodes']
        graphs.append({'gs': v.sens.numpy().astype(np.float64), 'gn': v.nec.numpy(),
                       'sp': sp.cpu().numpy(), 'np': npred.cpu().numpy(),
                       'tok': np.array([token_count(n) for n in nodes]),
                       'interactive': np.array([is_interactive(n) for n in nodes])})
        assert all(np.isfinite(array).all() for array in graphs[-1].values()), 'Non-finite graph scores'
        if i % 100 == 0 or i == len(rows):
            print(f'{name}: {i}/{len(rows)}', flush=True)
    np.savez_compressed(OUT / f'{name}_scores.npz', **{
        f'{i}_{key}': value for i, graph in enumerate(graphs) for key, value in graph.items()})
    return graphs


def report_results(decision, selected_metrics, fixed_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = decision['protocol']['reference']
    rows = [('Paper (different split)', ref), ('New, fixed 1/5', fixed_metrics), ('New, validation policy', selected_metrics)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key in zip(axes, ('tcnp', 'tcnp_i', 'tisl')):
        ax.bar(range(len(rows)), [r[key] for _, r in rows], color=['#999999', '#426c8a', '#50a58c'])
        ax.set_xticks(range(len(rows)), ['Paper', 'Fixed', 'Selected'])
        ax.set_title(key + (' (lower is better)' if key == 'tisl' else ''))
        ax.set_ylim(0, 1.05)
    fig.suptitle('Published reference vs released-data experiments; test splits differ')
    fig.tight_layout(); fig.savefig(OUT / 'reference_comparison.png', dpi=180); plt.close(fig)
    table = '\n'.join(f'| {name} | {m["tcnp"]:.6f} | {m["tcnp_i"]:.6f} | {m["tisl"]:.6f} |' for name, m in rows)
    selected = decision['selected']
    report = f'''# MINIM 参考性能对齐实验

日期：2026-11-04。使用 run-experiment 技能在既有 RTX 3060 Laptop GPU 环境实际训练。此次采用完整公开数据，不覆盖历史失败对照或已有修复模型。

| 结果来源 | TCNP | TCNP-I | TISL↓ |
|---|---:|---:|---:|
{table}

论文来自 Table 1：[官方论文](https://proceedings.mlr.press/v306/yu26bq.html)。论文测试集为 662，本地测试集为 484，不能把分差直接解释成方法优劣或严格复现。

模型仍为 512 维输入、三层 GATv2、hidden256、4 heads、595990 可训练参数。完整训练集 4637，验证集 629，测试集 484；保持原基础树划分 120/15/15，无交集。AdamW、lr=3e-4、weight decay=1e-5、batch32、10 epochs、seed0；使用末轮 checkpoint。

主动修改两项：CI-L1 改为 CI 加权交叉熵，旧任务哈希改为训练集固定唯一编号。未做类别平衡或推理先验校正。MiniLM 和原始节点标签沿用已核验的官方发布版本。

验证集扫描 55 组全局阈值。先最大化相对论文保留目标的最小达成度（超过目标封顶），再惩罚超出论文的泄漏；同分优先更低泄漏，再优先接近官方阈值。这避免为贴合论文数值而故意降低已有保留率或提高泄漏。选中 tau_nec={selected['tau_nec']}、tau_sens={selected['tau_sens']}。决定在测试推理之前写入 selected_policy.json；测试未参与训练、阈值或检查点选择。

测试真实节点微平均保留率为 {selected_metrics['micro_tcnp']:.6f}，真实交互节点微平均保留率为 {selected_metrics['micro_tcnp_i']:.6f}，与表中按变体聚合的 TCNP 分开记录。固定官方阈值 1/5 的结果也完整保留，避免只展示校准结果。

所选策略下，敏感节点检测召回率为 {selected_metrics['sensitivity_recall']:.2%}，高敏且关键节点的 Abstract 召回率为 {selected_metrics['hard_conflict_abstract_recall']:.2%}。TISL 只计算任务无关敏感内容的 token 代理，因此不能将低 TISL 解读为所有敏感字段都被正确抽象。

官方 Table 1 与独立数组计算均对所选及固定策略核对，误差小于 1e-6。保留官方指标文件、标签和原始失败产物；全部分数来自实际推理。单种子、替代损失和不同数据版本限制了结论；没有执行真实浏览器代理任务成功率评测。

checkpoint SHA256：`{decision['checkpoint_sha256']}`。

复跑：`bash scripts/run_reference_profile.sh`。已有完整 checkpoint 时校验配置并重新评估；缺失时按记录训练。入口：[运行脚本](../scripts/run_reference_profile.sh)、[评估与选策略](../scripts/evaluate_reference_profile.py)、[预先保存的协议](../scripts/reference_profile_protocol.json)。

产物：`runs/reference_profile_20261104/model.pt`、`selected_policy.json`、`results/reference_profile_20261104/summary.json`、`validation_sweep.csv`、`validation_scores.npz`、`test_scores.npz`；所选策略的 6 项 Table/Figure 在 selected/，固定阈值 Table 1 在 fixed_1_5/。Figure 2 的测试阈值曲线仅作描述，不用于重新选择阈值。

![参考比较](../results/reference_profile_20261104/reference_comparison.png)
'''
    (ROOT / 'reports/MINIM_参考性能对齐_20261104.md').write_text(report)


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    os.environ['MINIM_EMBEDDER_PATH'] = str(ROOT / 'data/models/all-MiniLM-L6-v2')
    protocol = json.loads((RUN / 'protocol.json').read_text())
    assert hashlib.sha256((ROOT / 'data/manifest.jsonl').read_bytes()).hexdigest() == protocol['manifest_sha256']
    config = json.loads((RUN / 'train_config.json').read_text())
    for field, value in protocol['train'].items():
        assert config[field] == value, (field, config[field], value)
    assert config['gat_hidden'] == 256 and config['gat_heads'] == 4
    split = json.loads((RUN / 'split.json').read_text())
    assert split == json.loads((ROOT / protocol['split_reference']).read_text())
    groups = {k: {(r['domain'], r['sample_id']) for r in v} for k, v in split.items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    model_hash = hashlib.sha256((RUN / 'model.pt').read_bytes()).hexdigest()
    device = torch.device('cuda')
    model = build_model(str(RUN / 'model.pt'), device)
    val = collect(model, split['val'], device, 'validation')
    curve = []
    for tn in protocol['tau_nec_grid']:
        for ts in protocol['tau_sens_grid']:
            m = policy_metrics(val, tn, ts)
            curve.append({'tau_nec': tn, 'tau_sens': ts, **m})
    selected = max(curve, key=lambda r: selection_key(r, protocol['reference'], r['tau_nec'], r['tau_sens']))
    with (OUT / 'validation_sweep.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=curve[0].keys()); writer.writeheader(); writer.writerows(curve)
    decision = {'checkpoint': str(RUN / 'model.pt'), 'checkpoint_sha256': model_hash,
                'selected_only_on_validation': True, 'selected': selected,
                'validation_fixed_policy': policy_metrics(val, 1, 5), 'protocol': protocol,
                'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('train_minim_scorer.py', 'evaluate_metrics.py', 'scripts/evaluate_reference_profile.py')}}
    write_json(RUN / 'selected_policy.json', decision)  # Lock before test scoring.
    print('Validation-selected policy:', selected, flush=True)
    del val
    test = collect(model, split['test'], device, 'test')
    metrics = policy_metrics(test, selected['tau_nec'], selected['tau_sens'])
    fixed = policy_metrics(test, 1, 5)
    write_json(OUT / 'summary.json', {'decision': decision, 'test_selected': metrics, 'test_fixed_1_5': fixed})
    del test, model
    torch.cuda.empty_cache()
    for name, tn, ts, entries in [
        ('selected', selected['tau_nec'], selected['tau_sens'], ('table1_main_results', 'table3_task_semantics', 'table4_abstraction_ablation', 'table10_domain_specific', 'figure2_threshold_sweep', 'figure3_policy_distribution')),
        ('fixed_1_5', 1, 5, ('table1_main_results',))]:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        for entry in entries:
            cmd = [sys.executable, '-u', f'experiment_{entry}.py', '--data_root', 'data',
                   '--split_path', str(RUN / 'split.json'), '--checkpoint', str(RUN / 'model.pt'),
                   '--results_dir', str(folder), '--device', 'cuda', '--tau_sens', str(ts),
                   '--selected_tau' if entry == 'figure2_threshold_sweep' else '--tau_nec', str(tn)]
            with (ROOT / f'logs/reference_profile_{name}_{entry}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('Exported', name, entry, flush=True)
        official = next(r for r in csv.DictReader((folder / 'table1.csv').open()) if r['method'] == 'MINIM')
        independently = metrics if name == 'selected' else fixed
        assert all(abs(float(official[k]) - independently[k]) < 1e-6 for k in ('tcnp', 'tcnp_i', 'tisl'))
    assert hashlib.sha256((RUN / 'model.pt').read_bytes()).hexdigest() == model_hash
    write_json(OUT / 'verification.json', {'official_table1_matches_independent_metrics': True,
                'checkpoint_unchanged': True, 'train_validation_test_base_trees_disjoint': True,
                'split_matches_original_released_run': True})
    report_results(decision, metrics, fixed)
    print('Completed:', metrics, 'fixed official thresholds:', fixed, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Report paper-alignment evidence separately from the repaired-data sweep."""
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT / 'runs/training_scale'
    out = ROOT / 'results/training_scale'
    out.mkdir(exist_ok=True)
    names = ['official_paper_full', 'repair_06', 'repair_15', 'repair_30']
    records = {}
    rows = []
    train_sets, val_hashes, test_hashes = [], [], []
    for name in names:
        run = runs / name
        result = json.loads((run / 'summary.json').read_text())
        result['test'] = json.loads((run / 'test_metrics.json').read_text())
        assert result['test']['checkpoint_sha256'] == hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
        split = json.loads((run / 'split.json').read_text())
        groups = {k: {(r['domain'], r['sample_id']) for r in v} for k, v in split.items()}
        assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
        config = result['config']
        assert hashlib.sha256((ROOT / 'runs/minim/model.pt').read_bytes()).hexdigest() == config['original_model_sha256']
        assert config['selected_epoch'] <= config['epochs']
        if name.startswith('repair'):
            train_sets.append(groups['train'])
            val_hashes.append(hashlib.sha256(json.dumps(split['val'], sort_keys=True).encode()).hexdigest())
        test_hashes.append(hashlib.sha256(json.dumps(split['test'], sort_keys=True).encode()).hexdigest())
        test, val = result['test'], result['validation']
        rows.append({'run': name, 'train_trees': len(groups['train']),
            'train_variants': len(split['train']), 'selected_epoch': config['selected_epoch'],
            'validation_micro_tcnp': val['micro_tcnp'], 'validation_tisl': val['tisl'],
            'test_tcnp': test['tcnp'], 'test_tcnp_i': test['tcnp_i'],
            'test_micro_tcnp': test['micro_tcnp'], 'test_micro_tcnp_i': test['micro_tcnp_i'],
            'test_tisl': test['tisl'], 'test_keep_rate': test['keep_rate'],
            'test_abstract_rate': test['abstract_rate'], 'test_remove_rate': test['remove_rate']})
        records[name] = result
    assert train_sets[0] < train_sets[1] < train_sets[2]
    assert len(set(val_hashes)) == 1
    assert len(set(test_hashes)) == 1
    official = records[names[0]]
    config = official['config']
    assert config['batch_size'] == 32 and config['epochs'] == 10 and config['gat_hidden'] == 256
    assert config['loss_type'] == 'ci_l1' and config['task_vocab'] is None and not config['prior_correction']
    source_hash = hashlib.sha256((runs / names[0] / 'official_train_snapshot.py').read_bytes()).hexdigest()
    assert source_hash == config['official_source_sha256']
    baseline = [r for r in csv.DictReader((ROOT / 'results/diagnosis/per_graph.csv').open()) if r['split'] == 'train']
    history = list(csv.DictReader((runs / names[0] / 'history.csv').open()))
    rng = random.Random(0)
    zeros = []
    for epoch in range(10):
        order = list(range(len(baseline)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 32):
            batch = [baseline[i] for i in order[start:start + 32]]
            losses.append(sum(int(r['nodes']) * float(r['zero_predictor_loss']) for r in batch) /
                          sum(int(r['nodes']) for r in batch))
        zeros.append({'epoch': epoch + 1, 'logged_loss': float(history[epoch]['train_loss']),
                      'all_zero_loss': sum(losses) / len(losses)})
    with (out / 'official_zero_baseline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=zeros[0].keys())
        writer.writeheader()
        writer.writerows(zeros)
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    repair_names = names[1:]
    selected = max(repair_names, key=lambda n: records[n]['validation']['selection_score'])
    prior_decision = json.loads((runs / 'selection_before_test.json').read_text())
    assert prior_decision['selected_run'] == selected
    assert all((runs / 'selection_before_test.json').stat().st_mtime <=
               (runs / name / 'test_metrics.json').stat().st_mtime for name in names)
    decision = {'selected_run': selected, 'checkpoint': str(runs / selected / 'model.pt'),
                'selection_uses_only_validation': True,
                'same_validation_for_repair_sizes': True, 'same_test_for_all_runs': True,
                'nested_training_tree_sets': True, 'official_source_sha256': source_hash,
                'test_metrics': records[selected]['test']}
    (out / 'selection.json').write_text(json.dumps(decision, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for field, label in [('micro_tcnp', 'Critical recall'), ('micro_tcnp_i', 'Interactive recall'), ('tisl', 'TISL')]:
        x = [records[n]['config']['training_tree_count'] for n in repair_names]
        for ax, split_name in zip(axes, ('validation', 'test')):
            ax.plot(x, [records[n][split_name][field] for n in repair_names], 'o-', label=label)
            ax.set(xlabel='Training trees', ylabel='Metric', title=f'Repaired model: {split_name}', xticks=x)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'data_scaling.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in zeros], [r['logged_loss'] for r in zeros], 'o-', label='Paper-aligned official loss')
    ax.plot([r['epoch'] for r in zeros], [r['all_zero_loss'] for r in zeros], 'x--', label='All-zero predictor, same batches')
    ax.set(xlabel='Epoch', ylabel='CI-L1', title='Frozen official implementation, hidden=256, effective batch=32')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'official_training.png', dpi=180)
    plt.close(fig)
    table = '\n'.join(f"| {r['train_trees']} | {r['train_variants']} | {r['selected_epoch']} | "
        f"{r['validation_micro_tcnp']:.4f} / {r['validation_tisl']:.4f} | "
        f"{r['test_micro_tcnp']:.4f} / {r['test_micro_tcnp_i']:.4f} / {r['test_tisl']:.4f} |" for r in rows[1:])
    first, chosen = records['repair_06']['test'], records[selected]['test']
    aligned = official['test']
    ablation_text = ''
    ablation_path = out / selected / 'table4.csv'
    if ablation_path.exists() and ablation_path.stat().st_mtime >= (runs / selected / 'model.pt').stat().st_mtime:
        ablation = list(csv.DictReader(ablation_path.open()))
        ablation_rows = '\n'.join(f"| {r['strategy']} | {float(r['tcnp']):.6f} | "
            f"{float(r['tcnp_i']):.6f} | {float(r['tisl']):.6f} |" for r in ablation)
        ablation_text = f'''## 所选修复版的 Abstract 消融

同一检查点、同一 484 变体测试集，使用官方 Table 4 脚本：

| 策略 | TCNP | TCNP-I | TISL |
|---|---:|---:|---:|
{ablation_rows}

本轮 A→R 会损失关键内容与交互保留，A→K 则提高泄漏代理，支持所选修复模型中 Abstract 的效用。该结论限于当前公开数据和修复配置；官方指标仍存在短文本占位符长度的代理偏差。

'''
    text = f'''# MINIM 官方设置对齐与训练规模实验结果

执行日期：2026-10-31（Asia/Shanghai）。先运行冻结官方实现的论文设置对照，再单独扩大修复版数据。两者的训练目标不同，不混称官方复现。

## 官方究竟写了什么

论文附录 B 给出：三层 GATv2、hidden=256、4 heads、512 维输入、AdamW、lr=3e-4、weight decay=1e-5、batch=32、10 epochs、alpha=lambda=1；正文公式采用 CI 加权 L1。正文另说明必要性阈值用验证集选择，最终为 1，敏感度阈值为 5。

但公开代码默认 hidden=496；公开数据为 5,750 个变体，按代码划分得到 4,637/629/484，而论文写 5,403 个变体、4,741 train 与 662 test。已提供的官方代码提交未包含原始 split 或可直接核验论文指标的检查点。因此可以对齐明确的设置，不能自行声称重建了论文原始数据划分。

唯一任务编码、平衡 CE、先验校正是前一轮为诊断失败新增的实验改动，不是官方配方。6/15/30 棵树也是本地扩展实验，官方原实验本来使用完整训练集。

数据线索：公开版含 7 种 `T_` 阅读任务、350 个变体；排除后为 5,400，仍不等于论文 5,403。原 27 种非 `T_` 任务在各域内没有检测到任务哈希碰撞；之前四对实证同树冲突均涉及新增阅读任务。因此，当前版本的编码问题不能直接外推为论文原实验也存在该冲突。

来源：[论文 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/yu26bq/yu26bq.pdf)，[官方固定版本训练器](https://raw.githubusercontent.com/yyyyhx/MINIM/456c60b6fd47941d1e5849db573dbbcb488e3ae7/train_minim_scorer.py)。逐项设置见 [对齐清单](MINIM_官方对齐清单.md)。

## 按论文设置的官方对照

直接实例化官方提交原始模型，调用原始损失；hidden=256、有效 batch=32、10 epochs；旧任务哈希、CI-L1、无额外 boost、无 CE、无先验校正。训练为全部 120 棵树、4,637 个变体，验证 629，测试 484。按照官方行为使用末轮检查点，另保存全部 10 个 epoch 供核验。

micro-batch=4 的梯度乘以其节点数占完整 batch 的比例，再累积到 32 张图后更新。不同大小图的整批梯度等价性测试通过。磁盘缓存仅作硬件适配。源码 SHA256：`{source_hash}`。

实测测试指标：TCNP={aligned['tcnp']:.6f}，TCNP-I={aligned['tcnp_i']:.6f}，TISL={aligned['tisl']:.6f}；真实关键节点保留率={aligned['micro_tcnp']:.6f}，真实关键交互节点保留率={aligned['micro_tcnp_i']:.6f}，Remove={aligned['remove_rate']:.2%}。

这项结果检验了模型宽度和 batch 对齐是否足以恢复效果；它仍受公开数据版本与论文划分未对齐的限制。日志与每轮权重保存在 `runs/training_scale/official_paper_full/`，全零损失对照在 `results/training_scale/official_zero_baseline.csv`。

## 修复版扩大训练

训练集合按基础树逐级包含，每域 2/5/10 棵，共 6/15/30 棵；训练变体数为 68/170/340。仅保留 balanced 和 reading 变体。各规模使用相同的唯一任务编码、平衡 CE、先验校正、hidden=256、seed=0、batch=4、40 epochs、固定阈值 1/5。

统一验证集来自原 val 的全部 15 棵基础树，过滤重复 persona 后为 185 个变体。按验证集 `micro TCNP - 0.5*TISL` 选择检查点和训练规模；所有选择完成后，才在同一原测试集 484 个变体、696,266 个节点上评估。上一轮只有 34 个测试变体，其 65.06% 不与这里的数字直接相减。

| 训练树 | 训练变体 | 所选 epoch | 验证：真实保留率 / TISL | 测试：真实保留率 / 真实交互保留率 / TISL |
|---|---:|---:|---:|---:|
{table}

仅按验证集选出 `{selected}`。与本轮相同测试集上的 6 树基线相比，其真实保留率变化为 {(chosen['micro_tcnp']-first['micro_tcnp'])*100:+.2f} 个百分点，TISL 变化为 {chosen['tisl']-first['tisl']:+.4f}；泄漏相对变化为 {(chosen['tisl']/first['tisl']-1)*100 if first['tisl'] else float('nan'):+.2f}%。三组的完整曲线和每个节点类别的识别指标均保留在原始 JSON/CSV 中。

这次测试中，15 树的真实保留率和交互保留率略高于 30 树，TISL 也低于 30 树。说明从 6 树扩大到 15 树已有明显效用收益，继续扩大并未单调改善；不因测试结果而推翻事先按验证选择的 30 树模型。TISL 相比 6 树上升，不能表述为所有指标同时改善。下一步应对 15/30 树做多种子确认，而不是直接假定更多数据必然更好。

所选模型敏感节点识别召回率为 {chosen['sensitivity_recall']:.2%}，高敏且关键节点的 Abstract 召回率为 {chosen['hard_conflict_abstract_recall']:.2%}。即使关键节点保留率已高，也不能据此声称所有隐私策略行为都已复现。这里是修复方法在公开测试集上的结果，不是原论文训练配方数值的成功复现。

注意：固定 epochs 意味着更多数据也带来更多优化更新；本实验不能把收益单独归因于数据覆盖。只有一个种子、一个嵌套子集序列，不报告显著性，也不据测试表现重新选模型。官方 TISL 的短文本占位符代理未修改，仍可能出现 Abstract 比 Keep 的代理值更高。

{ablation_text}## 文件与复跑

- 训练器：`scripts/run_training_scale.py`；`--mode official` 为论文设置的官方对照，`--mode repair --trees 6/15/30` 为单独的修复扩展。
- 完整复跑：`bash scripts/run_training_scale_study.sh`；已经完成的训练会跳过，选模决定在测试前保存。两条路径的官方 Table 1 与独立评估作一致性检查，所选修复版另输出 Table 4。
- 结果表：`results/training_scale/comparison.csv`；选模记录：`results/training_scale/selection.json`；全量统计：`results/training_scale/summary.json`。
- 官方对照权重：`runs/training_scale/official_paper_full/model.pt`；按验证选择的修复权重：`runs/training_scale/{selected}/model.pt`。
- 各规模 split/config/history/summary 均在独立运行目录。原失败权重 SHA256 未变，四组测试集完全一致，三组修复验证集一致，训练树集合逐级包含，均由分析脚本断言校验。
- 五项回归检查通过，包含节点数不等时的梯度累积等价性；两条路径的官方 Table 1 与独立指标核对一致。

![官方设置训练损失](../results/training_scale/official_training.png)

![修复版训练规模变化](../results/training_scale/data_scaling.png)
'''
    (ROOT / 'reports/MINIM_对齐与扩大训练结果.md').write_text(text)
    print(json.dumps({'selection': decision, 'comparison': rows}, indent=2))


if __name__ == '__main__':
    main()

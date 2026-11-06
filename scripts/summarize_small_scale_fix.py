#!/usr/bin/env python3
"""Verify pilot artifacts and render its measured results without retraining."""
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_metrics import token_count, tok_post
from train_minim_scorer import featurize_variant


def main():
    os.chdir(ROOT)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    run = ROOT / 'runs/pilot_fix'
    out = ROOT / 'results/pilot_fix'
    out.mkdir(exist_ok=True)
    config = json.loads((run / 'pilot_config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    selected = json.loads((run / 'selected_model.json').read_text())
    metrics = selected['test_metrics']
    split = json.loads((run / 'split.json').read_text())
    assert selected['arm'] == max(config['arms'], key=lambda k: summary[k]['validation']['selection_score'])
    assert hashlib.sha256((ROOT / 'runs/minim/model.pt').read_bytes()).hexdigest() == config['original_checkpoint_sha256']
    groups = {k: {(r['domain'], r['sample_id']) for r in rows} for k, rows in split.items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    assert set(config['task_vocab']) == {r['goal_id'] for r in split['train']}
    checks = []
    conflicts = json.loads((ROOT / 'results/diagnosis/input_conflicts.json').read_text())
    for group in conflicts:
        for pair in group['conflicts_with_group_representative']:
            variants = []
            for side in ('a', 'b'):
                row = {'domain': group['domain'], 'sample_id': group['sample_id'],
                       'goal_id': pair[f'goal_{side}'], 'tree_with_scores_path': pair[f'path_{side}']}
                variants.append(featurize_variant(row, 'data', 'gatv2', config['task_vocab']))
            distinct = not torch.equal(variants[0].node_features, variants[1].node_features)
            assert distinct
            checks.append({'goals': [pair['goal_a'], pair['goal_b']], 'fixed_inputs_distinct': distinct})
    counts, variants_seen = Counter(), 0
    short_sensitive_abstract = 0
    leak_official = leak_capped = denominator = 0.
    for line in (run / 'selected_inference/predictions_test.jsonl').open():
        record = json.loads(line)
        row = record['meta']
        tree = json.loads((ROOT / 'data' / row['tree_with_scores_path']).read_text())
        nodes = tree['nodes']
        sanitized = {str(n['nodeId']): n for n in record['sanitized_tree']['nodes']}
        variants_seen += 1
        for node, score in zip(nodes, record['scores']):
            action = score['action']
            counts[action] += 1
            node_id = str(node['nodeId'])
            assert (node_id in sanitized) == (action != 'R')
            if action == 'A':
                for field in ('name', 'value', 'description'):
                    if isinstance(node.get(field), dict) and 'value' in node[field]:
                        assert sanitized[node_id][field]['value'] == '[REDACTED]'
            if node['necessity_score'] <= 1:
                tok = token_count(node)
                sens = node['sensitivity_score']
                denominator += sens * tok
                post = tok_post(tok, action)
                leak_official += sens * post
                leak_capped += sens * min(tok, post)
                short_sensitive_abstract += action == 'A' and sens > 0 and tok < 6
    assert variants_seen == len(split['test'])
    assert sum(counts.values()) == metrics['nodes']
    assert all(counts[a] > 0 for a in ('K', 'A', 'R'))
    table = list(csv.DictReader((out / 'table1.csv').open()))
    official = next(x for x in table if x['method'] == 'MINIM')
    for field in ('tcnp', 'tcnp_i', 'tisl'):
        assert abs(float(official[field]) - metrics[field]) < 1e-6
    verification = {'all_checks_passed': True, 'test_variants': variants_seen,
        'test_action_counts': dict(counts), 'original_checkpoint_unchanged': True,
        'fixed_task_pairs': checks, 'short_sensitive_irrelevant_abstract_nodes': short_sensitive_abstract,
        'official_tisl': leak_official / denominator, 'auxiliary_capped_tisl': leak_capped / denominator,
        'auxiliary_metric_note': 'Cap placeholder token proxy at original token count; official tables are unchanged.'}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False) + '\n')
    for name in ('test_comparison.csv', 'history.csv', 'summary.json', 'validation_curves.png'):
        shutil.copy2(run / name, out / name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Legacy L1', 'Unique L1', 'Balanced L1', 'Balanced CE', 'Boosts (2,2,1)', 'Balanced CE + prior']
    arms = config['arms']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field, title in zip(axes, ('micro_tcnp', 'tisl'), ('Test critical-node recall (higher is better)', 'Test TISL (lower is better)')):
        values = [summary[k]['test'][field] for k in arms]
        ax.barh(labels, values, color=['#9AA4AF'] * 5 + ['#287D8E'])
        ax.set(title=title, xlim=(0, 1.1))
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value + .01, i, f'{value:.3f}', va='center', fontsize=9)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / 'test_comparison.png', dpi=180)
    plt.close(fig)
    rows = []
    for arm in arms:
        x = summary[arm]
        val, test = x['validation'], x['test']
        rows.append(f"| {arm} | {x['selected_epoch']} | {val['micro_tcnp']:.4f} / {val['tisl']:.4f} | {test['micro_tcnp']:.4f} / {test['tisl']:.4f} |")
    val = summary[selected['arm']]['validation']
    training = summary[selected['arm']]['training']
    ablation = {r['strategy']: r for r in csv.DictReader((out / 'table4.csv').open())}
    def ablation_pair(strategy):
        row = ablation[strategy]
        return f"{float(row['tcnp']):.6f}/{float(row['tisl']):.6f}"
    selected_path = Path(selected['checkpoint']).relative_to(ROOT)
    text = f'''# MINIM 小规模修复与验证报告

> 后续：[官方设置对齐与扩大训练结果](MINIM_对齐与扩大训练结果.md)另行保存。本报告保留第一轮 34 变体小测试的记录，不与后续 484 变体测试数字直接相减。

执行日期：2026-10-31（Asia/Shanghai）。状态：代码修复、六组小规模训练、验证集选择、独立小测试集评估、官方六个评估脚本及清洗推理均完成。**模型已解除全移除塌缩，但尚未达到论文效用水平。**

## 修复内容

1. 任务条件改为训练集确定的唯一编号。{len(config['task_vocab'])} 个任务不再压缩到 32 个哈希桶。编号映射写入检查点，推理和所有评估脚本自动恢复；未知任务明确报错。已对之前发现的四对真实碰撞输入检查，修复后全部可区分。
2. 保留原始 CI-L1，新增平衡 L1 与平衡交叉熵。类别权重仅用训练集计数拟合，已观察类别的权重为 `N / (有效类别数 × 类别计数)`；必要性继续使用 CI 权重。每个头按当前批次权重和归一化。
3. 平衡 CE 会改变类别分布，直接对其 softmax 求期望会推高分数。额外对照在推理 logits 上减去 `log(训练类别权重)`，抵消这部分分布偏移；训练从未出现的类别屏蔽。该校正不参与梯度更新，不使用验证/测试标签估计先验。
4. 原始特征缓存继续保存旧布局，唯一任务特征在读取后替换；原检查点和原正式结果表保留。新选项默认关闭，旧运行可以继续按原配置复查。

## 实验协议

- 公开数据不变。训练用三域各 {config['train_trees_per_domain']} 棵树，{len(split['train'])} 个变体、{training['nodes']:,} 个节点；验证用各 1 棵树，{len(split['val'])} 个变体、{val['nodes']:,} 个节点；测试用各 1 棵树，{len(split['test'])} 个变体、{metrics['nodes']:,} 个节点。三个集合的基础树两两不重叠，分别来自原 train/val/test。
- 每个任务保留 balanced 标注变体与 reading 变体，避免同任务四种 persona 的重复副本；这属于诊断子集，不代表全量数据分布。
- 三层 GATv2、hidden={config['gat_hidden']}、4 heads、512 维输入；seed={config['seed']}、batch={config['batch_size']}、AdamW、lr=3e-4、weight decay=1e-5，每组 {config['epochs']} epochs。各组相同初始权重和批次顺序。CUDA 图聚合仍可能带来数值非确定性。
- 所有策略阈值固定 necessity=1、sensitivity=5。仅以 `验证集真实关键节点保留率 - 0.5 × 验证集 TISL` 选择检查点与方法，平分保留更早轮次。先完成全部选择，再运行小测试集；未根据测试结果改参数。
- 后补的 `(2,2,1)` 和先验校正对照仍使用同一子集和协议。只有一个随机种子，测试仅 3 棵基础树，不能据此声称统计显著或完成论文复现。

## 六组实测结果

下表保留验证集选出的检查点，保留率均为节点微平均，排除了空关键集合的默认满分影响。

| 配置 | 所选 epoch | 验证：真实保留率 / TISL | 测试：真实保留率 / TISL |
|---|---:|---:|---:|
{chr(10).join(rows)}

**本次原始 L1、唯一编码 L1、`(2,2,1)` 加权 L1 在第 5 轮时均已退化为全移除。** 表中三个配置显示 100% 保留，是因为验证选择保留了第 1 轮的“全保留”检查点，同时 TISL=1；不能将其当成有效方法。`(2,2,1)` 在本次受控小规模对照中不足以单独解决失败，尚未做该配置的全量重训。

平衡 L1 和未经先验校正的 CE 恢复了非零分数，但隐私效用折中仍差。验证集最终选择 `{selected['arm']}` 的第 {summary[selected['arm']]['selected_epoch']} 轮。验证集真实保留率为 {val['micro_tcnp']:.2%}，TISL={val['tisl']:.4f}。

## 所选修复版的独立小测试结果

| 指标 | 实测 |
|---|---:|
| 官方 TCNP（变体宏平均） | {metrics['tcnp']:.6f} |
| 官方 TCNP-I（变体宏平均） | {metrics['tcnp_i']:.6f} |
| 真实关键节点保留率（微平均） | {metrics['micro_tcnp']:.6f} |
| 真实关键交互节点保留率（微平均） | {metrics['micro_tcnp_i']:.6f} |
| TISL | {metrics['tisl']:.6f} |
| 敏感节点识别 precision / recall | {metrics['sensitivity_precision']:.4f} / {metrics['sensitivity_recall']:.4f} |
| 高敏且关键节点的 Abstract 召回率 | {metrics['hard_conflict_abstract_recall']:.4f} |
| Keep / Abstract / Remove 节点数 | {counts['K']} / {counts['A']} / {counts['R']} |

相对完整观察，官方泄漏代理下降 {(1-metrics['tisl']):.2%}。相对原失败检查点的真实保留率 0%，现为 {metrics['micro_tcnp']:.2%}，提升 {100*metrics['micro_tcnp']:.2f} 个百分点；零基线不计算相对倍数。共有 {metrics['critical_nodes']} 个关键节点，保留 {round(metrics['critical_nodes']*metrics['micro_tcnp'])} 个；{metrics['critical_interactive_nodes']} 个关键交互节点，保留 {round(metrics['critical_interactive_nodes']*metrics['micro_tcnp_i'])} 个。

**限制：仍删除约 35% 的关键节点，不能作为已达到论文水平或可直接部署的隐私过滤器。** 高敏关键节点 Abstract 召回率也偏低，后续需要增加训练树覆盖并检查任务条件学习。工程修复与小规模有效学习已验证，泛化与论文数值复现仍待完成。

## Abstract 消融与指标异常

官方脚本实测 Adaptive 的 TCNP/TISL={ablation_pair('Adaptive')}；A→R 为 {ablation_pair('Abs->Rm')}，删除抽象节点会损失效用。A→K 为 {ablation_pair('Abs->Kp')}，泄漏代理反而更低，不能据此声称当前消融完整支持 Abstract 的隐私优势。

原因是官方公式 `tok_post(A)=max(6, 0.05*tok)` 对不足 6 token 的节点可能高于原文本长度。本测试有 {short_sensitive_abstract} 个敏感且无关的 Abstract 节点属于此情形。仅作为诊断，将该代理上限截为原 token 数得到 TISL={verification['auxiliary_capped_tisl']:.6f}；**未修改官方指标或用辅助指标挑模型**。后续应单独审计占位符代理与真实清洗文本的对应关系。

## 验证与复查

- 四项回归测试通过：真实碰撞形式可区分且文本/结构保留、未知任务拒绝、饱和 logits 下 CE 恢复梯度、先验校正不抬高缺失类别。
- 主训练 CLI 的 unique + balanced CE + prior correction 已完成一轮 GPU 集成检查；该一轮仅验证接口，不计入方法表现。
- 所选权重重新加载后指标一致；Table 1/3/4/10、Figure 2/3 和全 34 变体清洗推理通过。清洗输出的删除和三个文本字段抽象行为与动作一致。
- `results/pilot_fix/verification.json` 记录数据隔离、旧权重 SHA256、四对真实输入修复、动作计数和官方表一致性检查。

复跑小规模六组训练与验证选择：`bash scripts/run_small_scale_fix.sh`。完整结果在 `runs/pilot_fix/summary.json`，日志为 `logs/pilot_fix_train.log`、`logs/pilot_fix_extra_train.log`、`logs/pilot_fix_test.log`。所选检查点：`{selected_path}`。

![验证集学习曲线](../results/pilot_fix/validation_curves.png)

![独立小测试集比较](../results/pilot_fix/test_comparison.png)
'''
    (ROOT / 'reports/MINIM_小规模修复报告.md').write_text(text)
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit official TISL provenance, arithmetic, and proxy limits separately."""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_minim_inference import build_model, sanitize_node
from train_minim_scorer import featurize_variant, score_variant

COMMIT = '456c60b6fd47941d1e5849db573dbbcb488e3ae7'


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT / 'runs/minim/feature_cache')
    out = ROOT / 'results/tisl_audit_20261102'
    out.mkdir(exist_ok=True)
    upstream = subprocess.check_output(['git', 'show', f'{COMMIT}:evaluate_metrics.py'])
    local = (ROOT / 'evaluate_metrics.py').read_bytes()
    assert local == upstream
    snapshot = out / 'official_evaluate_metrics_snapshot.py'
    snapshot.write_bytes(upstream)
    spec = importlib.util.spec_from_file_location('official_metrics_frozen', snapshot)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    checkpoint = ROOT / 'runs/causal27_20261102/full_ci_ce/model.pt'
    split = json.loads((checkpoint.parent / 'split.json').read_text())
    device = torch.device('cuda')
    model = build_model(str(checkpoint), device)
    total = Counter()
    per_variant = []
    for index, row in enumerate(split['test'], 1):
        variant = featurize_variant(row, 'data_paper_aligned', 'gatv2')
        nodes = json.loads(Path(variant.tree_path).read_text())['nodes']
        with torch.no_grad():
            sh, nh = score_variant(model, variant, device)
        actions = np.where(nh.cpu().numpy() < 1, 'R', np.where(sh.cpu().numpy() >= 5, 'A', 'K'))
        assert set(actions) <= {'K', 'A', 'R'}
        gs, gn = variant.sens.tolist(), variant.nec.tolist()
        # Frozen upstream functions; use Python floats just as the Table 1 script.
        official_num = metric.compute_tisl_numerator(nodes, gs, gn, actions)
        official_den = metric.compute_tisl_denominator(nodes, gs, gn)
        tokens = np.array([metric.token_count(node) for node in nodes], dtype=np.float64)
        relevant_weights = np.array(gs, dtype=np.float64) * (np.array(gn) <= 1)
        post = np.where(actions == 'K', tokens, np.where(actions == 'A', np.maximum(6., .05*tokens), 0.))
        reference_num = float(np.dot(relevant_weights, post))
        reference_den = float(np.dot(relevant_weights, tokens))
        assert abs(official_num-reference_num) <= 1e-9 * max(1, official_den)
        assert abs(official_den-reference_den) <= 1e-9 * max(1, official_den)
        capped_num = float(np.dot(relevant_weights, np.minimum(tokens, post)))
        total['official_num'] += official_num
        total['official_den'] += official_den
        total['reference_num'] += reference_num
        total['reference_den'] += reference_den
        total['capped_num'] += capped_num
        short = (actions == 'A') & (tokens < 6) & (relevant_weights > 0)
        total['short_sensitive_irrelevant_A'] += int(short.sum())
        total['abstract_nodes'] += int((actions == 'A').sum())
        total['zero_denominator_variants'] += int(official_den == 0)
        for node, act, weight in zip(nodes, actions, relevant_weights):
            if act == 'A' and node.get('properties'):
                total['A_with_properties'] += 1
                total['sensitive_irrelevant_A_with_properties'] += int(weight > 0)
                assert sanitize_node(node, 'A')['properties'] == node['properties']
        ratio = official_num/official_den if official_den else 0.
        per_variant.append({'domain':row['domain'], 'sample_id':row['sample_id'], 'goal_id':row['goal_id'],
            'path':row['tree_with_scores_path'], 'nodes':len(nodes),
            'numerator':official_num, 'denominator':official_den,
            'per_variant_ratio':ratio, 'capped_numerator':capped_num,
            'short_sensitive_irrelevant_A':int(short.sum())})
        if index % 100 == 0 or index == len(split['test']):
            print(f'TISL arithmetic audit {index}/{len(split["test"])}', flush=True)
    official = total['official_num']/total['official_den']
    reference = total['reference_num']/total['reference_den']
    reported = next(r for r in csv.DictReader((ROOT / 'results/causal27_20261102/full_ci_ce/table1.csv').open()) if r['method']=='MINIM')
    assert abs(official-float(reported['tisl'])) < 5e-7
    # Edge cases demonstrate the *official definition*, not a replacement metric.
    toy={'role':{'value':'text'},'name':{'value':'secret'}}
    keep=metric.aggregate_dataset_metrics([([toy],[10.],[0.],['K'])])['tisl']
    remove=metric.aggregate_dataset_metrics([([toy],[10.],[0.],['R'])])['tisl']
    abstract=metric.aggregate_dataset_metrics([([toy],[10.],[0.],['A'])])['tisl']
    necessary=metric.aggregate_dataset_metrics([([toy],[10.],[10.],['K'])])['tisl']
    long={'name':{'value':' '.join(['word']*100)}}
    unequal=metric.aggregate_dataset_metrics([([toy],[10.],[0.],['K']),([long],[10.],[0.],['R'])])['tisl']
    assert keep==1 and remove==0 and abstract==6 and necessary==0
    assert abs(unequal-10/1010)<1e-12
    result={'upstream_commit':COMMIT,'evaluate_metrics_bytes_match':True,
        'evaluate_metrics_sha256':hashlib.sha256(upstream).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'test_variants':len(per_variant),'sums':dict(total),
        'official_dataset_tisl':official,'independent_dot_product_tisl':reference,
        'existing_table1_tisl':float(reported['tisl']),
        'mean_per_variant_tisl':float(np.mean([r['per_variant_ratio'] for r in per_variant])),
        'auxiliary_capped_tisl':total['capped_num']/total['official_den'],
        'auxiliary_only_not_official':True,
        'edge_cases':{'all_keep':keep,'all_remove':remove,'one_token_abstract':abstract,
            'necessary_secret_excluded_from_metric':necessary,'unequal_scenes_ratio_of_sums':unequal,
            'unequal_scenes_mean_of_ratios':.5},
        'property_note':'Unchanged properties do not by themselves prove sensitive content leakage; field-level ground truth is absent.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'per_variant.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=per_variant[0].keys());writer.writeheader();writer.writerows(per_variant)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

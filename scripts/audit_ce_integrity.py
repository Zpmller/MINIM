#!/usr/bin/env python3
"""Deterministic evidence: duplicate inputs, train-only lookup baseline, fresh scoring."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import defaultdict
import types
import subprocess
from dataclasses import replace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_minim_scorer import (featurize_variant, node_text, build_depth_map,
                               ui_structural_features, build_edge_index)
from evaluate_metrics import is_interactive, token_count

OUT = ROOT / 'results/ce_integrity_20261105'


def metrics(graphs):
    macro, macroi, nonempty, nonemptyi = [], [], [], []
    counts = np.zeros(9, dtype=np.int64)
    numerator = denominator = 0.
    maes = np.zeros(2)
    for g in graphs:
        s, n, gs, gn, tok, inter = (g[k] for k in ('s', 'n', 'gs', 'gn', 'tok', 'inter'))
        k, a, c = n >= 1, (n >= 1) & (s >= 5), gn >= 7
        ci = c & inter
        r = float(k[c].mean()) if c.any() else 1.
        ri = float(k[ci].mean()) if ci.any() else 1.
        macro.append(r); macroi.append(ri)
        if c.any(): nonempty.append(r)
        if ci.any(): nonemptyi.append(ri)
        counts += [int((c & k).sum()), int(c.sum()), int((ci & k).sum()), int(ci.sum()),
                   len(n), int(k.sum()), int(a.sum()), int(((gs >= 5) & (s >= 5)).sum()), int((gs >= 5).sum())]
        weight = gs * (gn <= 1)
        denominator += float((weight * tok).sum())
        post = np.where(k, np.where(a, np.maximum(6., .05 * tok), tok), 0.)
        numerator += float((weight * post).sum())
        maes += [float(abs(s-gs).sum()), float(abs(n-gn).sum())]
    return {'tcnp': float(np.mean(macro)), 'tcnp_i': float(np.mean(macroi)),
            'nonempty_macro_tcnp': float(np.mean(nonempty)), 'nonempty_macro_tcnp_i': float(np.mean(nonemptyi)),
            'micro_tcnp': int(counts[0])/int(counts[1]), 'micro_tcnp_i': int(counts[2])/int(counts[3]),
            'tisl': numerator/denominator, 'sens_mae': maes[0]/counts[4], 'nec_mae': maes[1]/counts[4],
            'nodes': int(counts[4]), 'critical': int(counts[1]), 'critical_kept': int(counts[0]),
            'critical_interactive': int(counts[3]), 'critical_interactive_kept': int(counts[2]),
            'keep': int(counts[5]-counts[6]), 'abstract': int(counts[6]), 'remove': int(counts[4]-counts[5]),
            'sensitivity_recall': int(counts[7])/int(counts[8]),
            'empty_critical_variants': len(graphs)-len(nonempty), 'variants': len(graphs)}


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    os.environ['MINIM_FEATURE_CACHE_DIR'] = str(ROOT/'runs/minim/feature_cache')
    run = ROOT/'runs/causal27_20261102/full_ci_ce'
    split = json.loads((run/'split.json').read_text())
    group_sets = {name: {(r['domain'],r['sample_id']) for r in rows} for name,rows in split.items()}
    overlaps = {a+'_'+b: len(group_sets[a]&group_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    # Canonicalize actual visible text, UI scalars and edges, without task or labels.
    fingerprints = defaultdict(list)
    for name,rows in split.items():
        first = {}
        for row in rows: first.setdefault((row['domain'],row['sample_id']), row)
        for group,row in first.items():
            nodes=json.loads((ROOT/'data'/row['tree_with_scores_path']).read_text())['nodes']
            depth=build_depth_map(nodes); max_depth=max(depth.values())
            canonical=[]
            for nd in nodes:
                role=(nd.get('role') or {}).get('value','none').lower()
                ui=ui_structural_features(nd,row,role,depth[str(nd['nodeId'])],max_depth)
                ui[64:96]=0  # task conditioning excluded from page identity
                canonical.append([node_text(nd),ui.tolist()])
            digest=hashlib.sha256(json.dumps([canonical,build_edge_index(nodes).tolist()],sort_keys=True).encode()).hexdigest()
            fingerprints[digest].append({'partition':name,'domain':group[0],'sample_id':group[1]})
    cross=[v for v in fingerprints.values() if len({x['partition'] for x in v})>1]
    print('Cross-partition identical visible graph groups:',len(cross),flush=True)

    # Fit only training-label means. Collapse redundant persona copies by tree/task.
    text_stats=defaultdict(lambda: np.zeros(3,dtype=np.float64))
    role_stats=defaultdict(lambda: np.zeros(3,dtype=np.float64))
    first={}
    for row in split['train']:
        first.setdefault((row['domain'],row['sample_id'],row['goal_id']),row)
    for i,row in enumerate(first.values(),1):
        nodes=json.loads((ROOT/'data'/row['tree_with_scores_path']).read_text())['nodes']
        for nd in nodes:
            role=(nd.get('role') or {}).get('value','none').lower()
            key=(row['domain'],row['goal_id'],role)
            value=np.array([nd['sensitivity_score'],nd['necessity_score'],1.])
            role_stats[key]+=value
            text_stats[(*key,node_text(nd))]+=value
        if i%300==0:print('Lookup training unique tree/task graphs:',i,flush=True)
    def predict(row,nd,use_text):
        role=(nd.get('role') or {}).get('value','none').lower()
        key=(row['domain'],row['goal_id'],role)
        stat=text_stats.get((*key,node_text(nd))) if use_text else None
        if stat is None:stat=role_stats.get(key)
        return stat[:2]/stat[2] if stat is not None else np.zeros(2)

    source=subprocess.check_output(['git','show','456c60b6fd47941d1e5849db573dbbcb488e3ae7:train_minim_scorer.py'])
    frozen=types.ModuleType('ce_audit_frozen');sys.modules[frozen.__name__]=frozen
    exec(compile(source,'<frozen_official>', 'exec'),frozen.__dict__)
    device=torch.device('cuda')
    model=frozen.build_scorer('gatv2',gat_hidden=256,gat_heads=4).to(device)
    model.load_state_dict(torch.load(run/'model.pt',map_location=device)['model_state'],strict=True);model.eval()
    baseline={'role_goal_mean':[],'text_role_goal_lookup':[],'fresh_ce_predictions':[]}
    hits=critical_hits=critical_total=0
    cache_checks=0;label_blind_checks=0;label_blind_evidence=[]
    details=[]
    for i,row in enumerate(split['test'],1):
        nodes=json.loads((ROOT/'data'/row['tree_with_scores_path']).read_text())['nodes']
        gs=np.array([n['sensitivity_score'] for n in nodes],dtype=np.float64)
        gn=np.array([n['necessity_score'] for n in nodes],dtype=np.float64)
        common={'gs':gs,'gn':gn,'tok':np.array([token_count(n) for n in nodes]),'inter':np.array([is_interactive(n) for n in nodes])}
        v=featurize_variant(row,'data_paper_aligned','gatv2')
        assert np.array_equal(gs,v.sens.numpy()) and np.array_equal(gn,v.nec.numpy())
        cache_checks+=1
        with torch.no_grad():
            s,n=frozen.score_variant(model,v,device)
            if label_blind_checks<3 and row['domain'] not in [x['domain'] for x in details if x.get('label_blind_check')]:
                altered=replace(v,sens=torch.full_like(v.sens,10),nec=torch.full_like(v.nec,10))
                ss,nn=frozen.score_variant(model,altered,device)
                cs,cn=frozen.score_variant(model,v,device)
                # CUDA scatter reductions can vary across identical forwards.
                # Judge label dependence on deterministic CPU forwards instead.
                gpu_altered_error=max(float((s-ss).abs().max()),float((n-nn).abs().max()))
                gpu_repeat_error=max(float((s-cs).abs().max()),float((n-cn).abs().max()))
                model.cpu()
                cpu_s,cpu_n=frozen.score_variant(model,v,torch.device('cpu'))
                altered_s,altered_n=frozen.score_variant(model,altered,torch.device('cpu'))
                assert torch.equal(cpu_s,altered_s) and torch.equal(cpu_n,altered_n)
                model.to(device)
                label_blind_evidence.append({'domain':row['domain'],'cpu_scores_exactly_unchanged':True,
                    'gpu_altered_label_max_abs_error':gpu_altered_error,'gpu_identical_repeat_max_abs_error':gpu_repeat_error})
                label_blind_checks+=1; checked=True
            else:checked=False
        baseline['fresh_ce_predictions'].append({**common,'s':s.cpu().numpy().astype(np.float64),'n':n.cpu().numpy().astype(np.float64)})
        for kind,use_text in [('role_goal_mean',False),('text_role_goal_lookup',True)]:
            pred=np.array([predict(row,nd,use_text) for nd in nodes])
            baseline[kind].append({**common,'s':pred[:,0],'n':pred[:,1]})
        seen=[]
        for nd in nodes:
            role=(nd.get('role') or {}).get('value','none').lower()
            seen.append((row['domain'],row['goal_id'],role,node_text(nd)) in text_stats)
        seen=np.array(seen);hits+=int(seen.sum());critical_hits+=int((seen & (gn>=7)).sum());critical_total+=int((gn>=7).sum())
        details.append({'domain':row['domain'],'sample_id':row['sample_id'],'goal_id':row['goal_id'],
                        'label_blind_check':checked,'nodes':len(nodes),'text_lookup_hit_nodes':int(seen.sum())})
        if i%100==0 or i==len(split['test']):print('Fresh CE and independent baseline scoring:',i,flush=True)
    outcomes={k:metrics(v) for k,v in baseline.items()}
    np.savez_compressed(OUT/'fresh_ce_scores.npz', **{
        f'{i}_{key}':value for i,g in enumerate(baseline['fresh_ce_predictions']) for key,value in g.items()})
    visible_test_hashes=[]
    for row in split['test']:
        nodes=json.loads((ROOT/'data'/row['tree_with_scores_path']).read_text())['nodes']
        stripped=[{k:v for k,v in node.items() if k not in ('sensitivity_score','necessity_score')} for node in nodes]
        visible_test_hashes.append(hashlib.sha256(json.dumps([row['goal_id'],stripped],sort_keys=True).encode()).hexdigest())
    near_path=OUT/'independent_near_clone_evidence.json'
    ablations={}
    if near_path.exists():
        clones={tuple(x) for x in json.loads(near_path.read_text())['near_clone_heldout_samples']}
        indices=[i for i,r in enumerate(split['test']) if (r['domain'],r['sample_id']) not in clones]
        ablations={name:metrics([graphs[i] for i in indices]) for name,graphs in baseline.items()}
    expected=json.loads((ROOT/'results/causal27_20261102/full_loss_only_comparison.json').read_text())['ci_ce_test']
    differences={k:outcomes['fresh_ce_predictions'][k]-expected[k] for k in ('tcnp','tcnp_i','tisl','micro_tcnp','micro_tcnp_i')}
    assert max(abs(v) for v in differences.values())<1e-6
    report={'date':'2026-11-05','split_variant_counts':{k:len(v) for k,v in split.items()},
            'base_tree_overlaps':overlaps,'canonical_duplicate_page_groups_cross_partition':cross,
            'unique_canonical_pages':len(fingerprints),'training_lookup_unique_tree_goal_graphs':len(first),
            'lookup_key_count':len(text_stats),'test_text_lookup_hit_fraction':hits/outcomes['fresh_ce_predictions']['nodes'],
            'critical_node_text_lookup_hit_fraction':critical_hits/critical_total,'test_cache_label_checks':cache_checks,
            'label_blind_forward_checks':label_blind_checks,'metrics':outcomes,'fresh_minus_existing_metrics':differences,
            'label_blind_evidence':label_blind_evidence,
            'unique_exact_visible_test_inputs_with_goal':len(set(visible_test_hashes)),
            'posthoc_excluding_near_clone_pages':ablations,
            'model_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
            'lookup_protocol':'train-only empirical score means; unique tree/task representatives; exact text-role-goal lookup with role-goal fallback; no test tuning'}
    (OUT/'deterministic_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'per_variant.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=details[0].keys());writer.writeheader();writer.writerows(details)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()

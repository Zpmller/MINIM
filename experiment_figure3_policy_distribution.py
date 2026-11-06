import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_minim_inference import build_model
from train_minim_scorer import DEFAULT_MODEL_TYPE, MODEL_TYPES, featurize_variant, read_manifest, score_variant, set_seed


def load_split(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def row_key(row: Dict) -> str:
    return f'{row["domain"]}|{row["sample_id"]}|{row["goal_id"]}|{row["tree_with_scores_path"]}'


def collect_test_rows(data_root: str, split_path: str) -> List[Dict]:
    split = load_split(split_path)["test"]
    manifest = read_manifest(data_root)
    idx = {row_key(r): r for r in manifest}
    rows = [idx[row_key(r)] for r in split if row_key(r) in idx]
    return rows


def minim_actions(s_hat: List[float], n_hat: List[float], tau_nec: float = 1.0, tau_sens: float = 5.0) -> List[str]:
    actions = []
    for s, n in zip(s_hat, n_hat):
        if n < tau_nec:
            actions.append("R")
        elif s >= tau_sens:
            actions.append("A")
        else:
            actions.append("K")
    return actions


def bucket_label(gt_n: float, gt_s: float, tau_crit: float, tau_irr: float, tau_sens: float) -> str:
    if gt_n >= tau_crit and gt_s >= tau_sens:
        return "Hard Conflict"
    if gt_n >= tau_crit and gt_s < tau_sens:
        return "Safe Utility"
    if gt_n < tau_irr and gt_s >= tau_sens:
        return "Privacy Risk"
    if gt_n < tau_irr and gt_s < tau_sens:
        return "Irrelevant"
    return "Other"


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)
    rows = collect_test_rows(args.data_root, args.split_path)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    buckets = {
        "Hard Conflict": {"K": 0, "A": 0, "R": 0},
        "Safe Utility": {"K": 0, "A": 0, "R": 0},
        "Privacy Risk": {"K": 0, "A": 0, "R": 0},
        "Irrelevant": {"K": 0, "A": 0, "R": 0},
    }

    print("Step 1/3: collecting policy actions by semantic bucket")
    for i, row in enumerate(rows, start=1):
        variant = featurize_variant(row, args.data_root, model_type, getattr(model, "task_vocab", None))
        with torch.no_grad():
            s_hat, n_hat = score_variant(model, variant, device)
        actions = minim_actions(s_hat.cpu().tolist(), n_hat.cpu().tolist(), args.tau_nec, args.tau_sens)
        for gt_n, gt_s, action in zip(variant.nec.tolist(), variant.sens.tolist(), actions):
            b = bucket_label(gt_n, gt_s, args.tau_crit, args.tau_irr, args.tau_sens)
            if b in buckets:
                buckets[b][action] += 1
        if i % 50 == 0 or i == len(rows):
            print(f"Step 1/3: progress {i}/{len(rows)}")

    print("Step 2/3: computing proportions")
    labels = ["Hard Conflict", "Safe Utility", "Privacy Risk", "Irrelevant"]
    k_vals, a_vals, r_vals = [], [], []
    for label in labels:
        total = float(sum(buckets[label].values()))
        if total <= 0:
            k_vals.append(0.0)
            a_vals.append(0.0)
            r_vals.append(0.0)
        else:
            k_vals.append(buckets[label]["K"] / total)
            a_vals.append(buckets[label]["A"] / total)
            r_vals.append(buckets[label]["R"] / total)

    print("Step 3/3: rendering figure")
    csv_path = os.path.join(args.results_dir, "figure3_policy_distribution.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("bucket,n,keep,abstract,remove\n")
        for label in labels:
            counts = buckets[label]
            total = sum(counts.values())
            denom = max(1, total)
            f.write(f'{label},{total},{counts["K"] / denom:.8f},{counts["A"] / denom:.8f},{counts["R"] / denom:.8f}\n')
    print(f"Saved {csv_path}")
    y = np.arange(len(labels))
    plt.figure(figsize=(7, 4.2))
    plt.barh(y, k_vals, label="Keep")
    plt.barh(y, a_vals, left=k_vals, label="Abstract")
    plt.barh(y, r_vals, left=(np.array(k_vals) + np.array(a_vals)), label="Remove")
    plt.yticks(y, labels)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Action Proportion")
    plt.title("Policy Action Distribution")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, "figure3_policy_distribution.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"Saved {fig_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Figure 3 policy distribution.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
    parser.add_argument("--checkpoint", type=str, default="runs/minim/model.pt")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--tau_nec", type=float, default=1.0)
    parser.add_argument("--tau_sens", type=float, default=5.0)
    parser.add_argument("--tau_crit", type=float, default=7.0)
    parser.add_argument("--tau_irr", type=float, default=1.0)
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

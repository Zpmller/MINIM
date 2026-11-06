import argparse
import json
import os
from typing import Dict, List

import torch

from evaluate_metrics import aggregate_dataset_metrics
from run_minim_inference import build_model
from train_minim_scorer import DEFAULT_MODEL_TYPE, MODEL_TYPES, featurize_variant, read_manifest, score_variant, set_seed


DOMAINS = ["gmail", "reddit", "shopping"]


def load_split(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def row_key(row: Dict) -> str:
    return f'{row["domain"]}|{row["sample_id"]}|{row["goal_id"]}|{row["tree_with_scores_path"]}'


def collect_split_rows(data_root: str, split_path: str, split_name: str) -> List[Dict]:
    split = load_split(split_path)[split_name]
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


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)
    rows = collect_split_rows(args.data_root, args.split_path, args.split_name)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    grouped = {domain: [] for domain in DOMAINS}
    print("Step 1/3: running predictions and grouping by domain")
    for i, row in enumerate(rows, start=1):
        variant = featurize_variant(row, args.data_root, model_type, getattr(model, "task_vocab", None))
        with open(variant.tree_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        with torch.no_grad():
            s_hat, n_hat = score_variant(model, variant, device)
        actions = minim_actions(s_hat.cpu().tolist(), n_hat.cpu().tolist(), args.tau_nec, args.tau_sens)
        grouped[row["domain"]].append((nodes, variant.sens.tolist(), variant.nec.tolist(), actions))
        if i % 50 == 0 or i == len(rows):
            print(f"Step 1/3: progress {i}/{len(rows)}")

    print("Step 2/3: aggregating metrics")
    metrics = {domain: aggregate_dataset_metrics(payload) for domain, payload in grouped.items()}

    print("Step 3/3: writing table")
    print("Domain,Test N,TCNP,TISL")
    for domain in DOMAINS:
        m = metrics[domain]
        print(f'{domain},{len(grouped[domain])},{100.0*m["tcnp"]:.2f}%,{100.0*m["tisl"]:.2f}%')

    out_csv = os.path.join(args.results_dir, "table10.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("domain,test_n,tcnp_percent,tisl_percent\n")
        for domain in DOMAINS:
            m = metrics[domain]
            f.write(f'{domain},{len(grouped[domain])},{100.0*m["tcnp"]:.6f},{100.0*m["tisl"]:.6f}\n')
    print(f"Saved {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Table 10 domain-specific performance.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
    parser.add_argument("--split_name", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default="runs/minim/model.pt")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--tau_nec", type=float, default=1.0)
    parser.add_argument("--tau_sens", type=float, default=5.0)
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

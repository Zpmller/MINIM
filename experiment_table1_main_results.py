import argparse
import json
import os
import random
from typing import Dict, List

import torch

from evaluate_metrics import aggregate_dataset_metrics
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


def base_prediction_for_variant(model, row: Dict, data_root: str, device: torch.device, model_type: str):
    variant = featurize_variant(row, data_root, model_type, getattr(model, "task_vocab", None))
    with open(variant.tree_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)["nodes"]
    with torch.no_grad():
        s_hat, n_hat = score_variant(model, variant, device)
    return {
        "nodes": nodes,
        "gt_s": variant.sens.tolist(),
        "gt_n": variant.nec.tolist(),
        "s_hat": s_hat.cpu().tolist(),
        "n_hat": n_hat.cpu().tolist(),
    }


def topk_actions(scores: List[float], keep_ratio: float, reverse: bool) -> List[str]:
    n = len(scores)
    k = max(1, int(round(n * keep_ratio)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=reverse)
    keep_idx = set(order[:k])
    return ["K" if i in keep_idx else "R" for i in range(n)]


def random_budget_actions(n: int, keep_ratio: float, rng: random.Random) -> List[str]:
    k = max(1, int(round(n * keep_ratio)))
    idx = list(range(n))
    rng.shuffle(idx)
    keep_idx = set(idx[:k])
    return ["K" if i in keep_idx else "R" for i in range(n)]


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
    print("Step 1/4: loading model and test rows")
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)
    rows = collect_test_rows(args.data_root, args.split_path)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    print(f"Step 2/4: computing node predictions for {len(rows)} variants")
    base = [base_prediction_for_variant(model, r, args.data_root, device, model_type) for r in rows]

    print("Step 3/4: evaluating policies")
    rng = random.Random(args.seed)
    methods = {
        "Full Observation": [],
        "Random Budget": [],
        "Sensitivity-Only": [],
        "Necessity-Only": [],
        "MINIM": [],
    }

    for item in base:
        n_nodes = len(item["nodes"])
        full_actions = ["K"] * n_nodes
        rand_actions = random_budget_actions(n_nodes, args.keep_ratio, rng)
        sens_actions = topk_actions(item["s_hat"], args.keep_ratio, reverse=False)
        nec_actions = topk_actions(item["n_hat"], args.keep_ratio, reverse=True)
        minim = minim_actions(item["s_hat"], item["n_hat"], args.tau_nec, args.tau_sens)

        methods["Full Observation"].append((item["nodes"], item["gt_s"], item["gt_n"], full_actions))
        methods["Random Budget"].append((item["nodes"], item["gt_s"], item["gt_n"], rand_actions))
        methods["Sensitivity-Only"].append((item["nodes"], item["gt_s"], item["gt_n"], sens_actions))
        methods["Necessity-Only"].append((item["nodes"], item["gt_s"], item["gt_n"], nec_actions))
        methods["MINIM"].append((item["nodes"], item["gt_s"], item["gt_n"], minim))

    results = {name: aggregate_dataset_metrics(payload) for name, payload in methods.items()}

    print("Step 4/4: done. Table 1 metrics")
    print("Method,TCNP,TCNP-I,TISL")
    for method in ["Full Observation", "Random Budget", "Sensitivity-Only", "Necessity-Only", "MINIM"]:
        m = results[method]
        print(f'{method},{m["tcnp"]:.4f},{m["tcnp_i"]:.4f},{m["tisl"]:.4f}')

    out_csv = os.path.join(args.results_dir, "table1.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("method,tcnp,tcnp_i,tisl\n")
        for method in ["Full Observation", "Random Budget", "Sensitivity-Only", "Necessity-Only", "MINIM"]:
            m = results[method]
            f.write(f'{method},{m["tcnp"]:.6f},{m["tcnp_i"]:.6f},{m["tisl"]:.6f}\n')
    print(f"Saved {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce MINIM Table 1 main results.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
    parser.add_argument("--checkpoint", type=str, default="runs/minim/model.pt")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--tau_nec", type=float, default=1.0)
    parser.add_argument("--tau_sens", type=float, default=5.0)
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

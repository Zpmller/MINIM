import argparse
import json
import os
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


def map_actions(actions: List[str], map_a_to: str) -> List[str]:
    if map_a_to == "A":
        return actions
    return [map_a_to if a == "A" else a for a in actions]


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)
    rows = collect_test_rows(args.data_root, args.split_path)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    methods = {"MINIM (Ours)": [], "Conservative (A->R)": [], "Radical (A->K)": []}
    print("Step 1/3: running inference and policy variants")
    for i, row in enumerate(rows, start=1):
        variant = featurize_variant(row, args.data_root, model_type, getattr(model, "task_vocab", None))
        with open(variant.tree_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        with torch.no_grad():
            s_hat, n_hat = score_variant(model, variant, device)
        base = minim_actions(s_hat.cpu().tolist(), n_hat.cpu().tolist(), args.tau_nec, args.tau_sens)
        methods["MINIM (Ours)"].append((nodes, variant.sens.tolist(), variant.nec.tolist(), map_actions(base, "A")))
        methods["Conservative (A->R)"].append((nodes, variant.sens.tolist(), variant.nec.tolist(), map_actions(base, "R")))
        methods["Radical (A->K)"].append((nodes, variant.sens.tolist(), variant.nec.tolist(), map_actions(base, "K")))
        if i % 50 == 0 or i == len(rows):
            print(f"Step 1/3: progress {i}/{len(rows)}")

    print("Step 2/3: aggregating metrics")
    metrics = {k: aggregate_dataset_metrics(v) for k, v in methods.items()}

    print("Step 3/3: writing table")
    print("Policy Variant,Strategy,TCNP,TCNP-I,TISL")
    for name, strategy in [
        ("MINIM (Ours)", "Adaptive"),
        ("Conservative (A->R)", "Abs->Rm"),
        ("Radical (A->K)", "Abs->Kp"),
    ]:
        m = metrics[name]
        print(f'{name},{strategy},{m["tcnp"]:.4f},{m["tcnp_i"]:.4f},{m["tisl"]:.4f}')

    out_csv = os.path.join(args.results_dir, "table4.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("policy_variant,strategy,tcnp,tcnp_i,tisl\n")
        for name, strategy in [
            ("MINIM (Ours)", "Adaptive"),
            ("Conservative (A->R)", "Abs->Rm"),
            ("Radical (A->K)", "Abs->Kp"),
        ]:
            m = metrics[name]
            f.write(f'{name},{strategy},{m["tcnp"]:.6f},{m["tcnp_i"]:.6f},{m["tisl"]:.6f}\n')
    print(f"Saved {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Table 4 abstraction ablation.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
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

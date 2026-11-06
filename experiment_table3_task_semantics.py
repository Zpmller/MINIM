import argparse
import json
import os
from typing import Dict, List

import torch

from evaluate_metrics import aggregate_dataset_metrics
from run_minim_inference import build_model
from train_minim_scorer import DEFAULT_MODEL_TYPE, MODEL_TYPES, featurize_variant, read_manifest, score_variant, set_seed


TASK_SEMANTICS = {
    "Informational": {
        "read_email",
        "read_post",
        "read_comments",
        "read_slack_message",
        "search_email",
        "search_product",
        "search_discussion",
        "view_product_details",
        "view_cart",
        "navigate_forums",
        "read_reviews",
        "compare_products",
        "T_read_subject",
        "T_identify_sender",
        "T_read_body",
        "T_read_price",
        "T_compare_items",
        "T_read_post",
        "T_find_detail",
    },
    "Sensitive/Admin": {"complete_2fa", "extract_2fa_code", "save_password"},
    "Transactional": {
        "compose_email",
        "reply_email",
        "forward_email",
        "delete_email",
        "reply_slack",
        "add_to_cart",
        "vote_content",
        "reply_comment",
        "access_bookmark",
        "switch_tab",
        "add_to_wishlist",
        "open_slack_app",
    },
}

_WARNED_UNKNOWN_GOALS: set = set()


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


def semantic_bucket(goal_id: str) -> str:
    for name, goals in TASK_SEMANTICS.items():
        if goal_id in goals:
            return name
    if goal_id not in _WARNED_UNKNOWN_GOALS:
        _WARNED_UNKNOWN_GOALS.add(goal_id)
        print(f"[warn] goal_id '{goal_id}' not in TASK_SEMANTICS; defaulting to Informational")
    return "Informational"


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
    rows = collect_test_rows(args.data_root, args.split_path)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    grouped = {"Informational": [], "Sensitive/Admin": [], "Transactional": []}
    print("Step 1/3: running predictions and grouping by semantics")
    for i, row in enumerate(rows, start=1):
        variant = featurize_variant(row, args.data_root, model_type, getattr(model, "task_vocab", None))
        with open(variant.tree_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        with torch.no_grad():
            s_hat, n_hat = score_variant(model, variant, device)
        actions = minim_actions(s_hat.cpu().tolist(), n_hat.cpu().tolist(), args.tau_nec, args.tau_sens)
        group = semantic_bucket(row["goal_id"])
        grouped[group].append((nodes, variant.sens.tolist(), variant.nec.tolist(), actions))
        if i % 50 == 0 or i == len(rows):
            print(f"Step 1/3: progress {i}/{len(rows)}")

    print("Step 2/3: aggregating metrics")
    out = {k: aggregate_dataset_metrics(v) for k, v in grouped.items()}

    print("Step 3/3: writing table")
    print("Task Semantics,Keep,Abstract,TCNP,TCNP-I")
    order = ["Informational", "Sensitive/Admin", "Transactional"]
    for k in order:
        v = out[k]
        print(f'{k},{100.0*v["keep_rate"]:.2f}%,{100.0*v["abstract_rate"]:.2f}%,{v["tcnp"]:.4f},{v["tcnp_i"]:.4f}')

    out_csv = os.path.join(args.results_dir, "table3.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("task_semantics,keep_percent,abstract_percent,tcnp,tcnp_i\n")
        for k in order:
            v = out[k]
            f.write(f'{k},{100.0*v["keep_rate"]:.4f},{100.0*v["abstract_rate"]:.4f},{v["tcnp"]:.6f},{v["tcnp_i"]:.6f}\n')
    print(f"Saved {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Table 3 task semantics analysis.")
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

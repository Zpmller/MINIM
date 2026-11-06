#!/usr/bin/env python3
"""Select tau_nec on the paper-aligned validation split."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_metrics import aggregate_dataset_metrics
from run_minim_inference import build_model
from train_minim_scorer import featurize_variant, read_manifest, score_variant, set_seed


def key(row):
    return f'{row["domain"]}|{row["sample_id"]}|{row["goal_id"]}|{row["tree_with_scores_path"]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data_paper_aligned")
    parser.add_argument("--split_path", default="runs/paper_aligned/split.json")
    parser.add_argument("--checkpoint", default="runs/paper_aligned/model.pt")
    parser.add_argument("--results_dir", default="results/paper_aligned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device, "gatv2")
    model_type = getattr(model, "model_type", "gatv2")
    split = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
    manifest = read_manifest(args.data_root)
    by_key = {key(row): row for row in manifest}
    rows = [by_key[key(row)] for row in split["val"] if key(row) in by_key]
    base = []
    print(f"Scoring {len(rows)} validation variants")
    for i, row in enumerate(rows, 1):
        variant = featurize_variant(row, args.data_root, model_type, getattr(model, "task_vocab", None))
        with torch.no_grad():
            s_hat, n_hat = score_variant(model, variant, device)
        with open(variant.tree_path, encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        base.append((nodes, variant.sens.tolist(), variant.nec.tolist(),
                     s_hat.cpu().tolist(), n_hat.cpu().tolist()))
        if i % 100 == 0 or i == len(rows):
            print(f"Progress {i}/{len(rows)}")

    curve = []
    for tau in [i / 2 for i in range(21)]:
        payload = []
        for nodes, gs, gn, sp, npred in base:
            actions = ["R" if n < tau else ("A" if s >= 5.0 else "K") for s, n in zip(sp, npred)]
            payload.append((nodes, gs, gn, actions))
        metrics = aggregate_dataset_metrics(payload)
        curve.append({"tau_nec": tau, "tcnp": metrics["tcnp"], "tcnp_i": metrics["tcnp_i"],
                      "tisl": metrics["tisl"], "objective": metrics["tcnp"] - 0.5 * metrics["tisl"]})
    best_value = max(row["objective"] for row in curve)
    best = next(row for row in curve if abs(row["objective"] - best_value) < 1e-12)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    (out / "selected_threshold.json").write_text(json.dumps({
        "selection_rule": "maximize validation TCNP - 0.5*TISL; ties choose lower tau",
        "selected": best,
        "validation_variants": len(rows),
        "tau_sens": 5.0,
        "candidates": [row["tau_nec"] for row in curve],
    }, indent=2) + "\n", encoding="utf-8")
    print("Selected validation operating point:", json.dumps(best))


if __name__ == "__main__":
    main()

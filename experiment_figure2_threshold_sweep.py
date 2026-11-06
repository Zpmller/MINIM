import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def minim_actions(s_hat: List[float], n_hat: List[float], tau_nec: float, tau_sens: float = 5.0) -> List[str]:
    actions = []
    for s, n in zip(s_hat, n_hat):
        if n < tau_nec:
            actions.append("R")
        elif s >= tau_sens:
            actions.append("A")
        else:
            actions.append("K")
    return actions


def load_metrics_csv(csv_path: str) -> List[Tuple[float, float, float]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(float(row["tau_nec"]), float(row["tcnp"]), float(row["tisl"])) for row in reader]


def plot_threshold_sweep(metrics: List[Tuple[float, float, float]], args: argparse.Namespace) -> str:
    xs = [x[0] for x in metrics]
    ys_tcnp = [x[1] for x in metrics]
    ys_tisl = [min(x[2], 1.0) for x in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        xs,
        ys_tcnp,
        color="#2ca02c",
        marker="o",
        markersize=5,
        linewidth=3,
        label="Utility (TCNP)",
    )
    ax.plot(
        xs,
        ys_tisl,
        color="#d62728",
        marker="o",
        markersize=5,
        linewidth=3,
        label="Sensitive Data Exposure",
    )
    ax.axvline(args.selected_tau, color="#9e9e9e", linestyle="--", linewidth=2)
    ax.text(
        args.selected_tau + 0.1,
        0.54,
        rf"Selected $\tau$ = {args.selected_tau:.1f}" + "\n(Sweet Spot)",
        ha="left",
        va="center",
        fontsize=11,
    )

    ax.set_title("MINIM Threshold Sweep: Utility vs. Privacy", fontsize=16, pad=12)
    ax.set_xlabel(r"Necessity Threshold ($\tau$)", fontsize=13)
    ax.set_ylabel("Rate (Normalized)", fontsize=13)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color="#d9d9d9", linestyle=":", linewidth=0.9)
    ax.legend(loc="center", frameon=False, fontsize=11)
    fig.tight_layout()

    fig_path = os.path.join(args.results_dir, "figure2_threshold_sweep.png")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "pareto_test_sweep.csv")
    if args.plot_only:
        print("Step 1/2: loading existing threshold sweep CSV")
        metrics = load_metrics_csv(csv_path)
        print("Step 2/2: drawing Figure 2")
        fig_path = plot_threshold_sweep(metrics, args)
        print(f"Saved {fig_path}")
        return

    device = torch.device(args.device)

    print("Step 1/4: loading model and test split")
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)
    rows = collect_test_rows(args.data_root, args.split_path)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    print("Step 2/4: computing base predictions")
    base = [base_prediction_for_variant(model, r, args.data_root, device, model_type) for r in rows]

    taus = [i * 0.5 for i in range(21)]
    metrics = []
    print("Step 3/4: sweeping tau_nec")
    for i, tau in enumerate(taus, start=1):
        payload = []
        for item in base:
            actions = minim_actions(item["s_hat"], item["n_hat"], tau_nec=tau, tau_sens=args.tau_sens)
            payload.append((item["nodes"], item["gt_s"], item["gt_n"], actions))
        agg = aggregate_dataset_metrics(payload)
        metrics.append((tau, agg["tcnp"], agg["tisl"]))
        print(f"Step 3/4: {i}/{len(taus)} tau={tau:.1f} tcnp={agg['tcnp']:.4f} tisl={agg['tisl']:.4f}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tau_nec,tcnp,tisl\n")
        for tau, tcnp, tisl in metrics:
            f.write(f"{tau:.1f},{tcnp:.6f},{tisl:.6f}\n")

    fig_path = plot_threshold_sweep(metrics, args)

    print("Step 4/4: done")
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 threshold sweep.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
    parser.add_argument("--checkpoint", type=str, default="runs/minim/model.pt")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--tau_sens", type=float, default=5.0)
    parser.add_argument("--selected_tau", type=float, default=1.0)
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot_only", action="store_true", help="Redraw Figure 2 from an existing pareto_test_sweep.csv.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

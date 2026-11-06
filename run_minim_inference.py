import argparse
import copy
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from train_minim_scorer import (
    DEFAULT_GAT_HEADS,
    DEFAULT_GAT_HIDDEN,
    DEFAULT_MODEL_TYPE,
    MODEL_TYPES,
    build_scorer,
    featurize_variant,
    read_manifest,
    score_variant,
    set_seed,
)


def load_split(split_path: str) -> Dict[str, List[Dict]]:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def row_key(row: Dict) -> str:
    return f'{row["domain"]}|{row["sample_id"]}|{row["goal_id"]}|{row["tree_with_scores_path"]}'


def index_manifest(rows: List[Dict]) -> Dict[str, Dict]:
    return {row_key(r): r for r in rows}


def action_from_scores(s_hat: float, n_hat: float, tau_nec: float, tau_sens: float) -> str:
    if n_hat < tau_nec:
        return "R"
    if s_hat >= tau_sens:
        return "A"
    return "K"


def sanitize_node(node: Dict, action: str) -> Dict:
    node = copy.deepcopy(node)
    if action != "A":
        return node
    for k in ("name", "value", "description"):
        field = node.get(k)
        if isinstance(field, dict) and "value" in field:
            field["value"] = "[REDACTED]"
    return node


def predict_variant(
    model: nn.Module,
    row: Dict,
    data_root: str,
    device: torch.device,
    tau_nec: float,
    tau_sens: float,
    model_type: str,
) -> Dict:
    variant = featurize_variant(row, data_root, model_type, getattr(model, "task_vocab", None))

    with torch.no_grad():
        s_hat, n_hat = score_variant(model, variant, device)
    s_hat_l = s_hat.cpu().tolist()
    n_hat_l = n_hat.cpu().tolist()
    actions = [action_from_scores(s, n, tau_nec, tau_sens) for s, n in zip(s_hat_l, n_hat_l)]

    with open(variant.tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)
    nodes = tree["nodes"]
    sanitized_nodes = [sanitize_node(node, action) for node, action in zip(nodes, actions) if action != "R"]

    return {
        "meta": {
            "domain": row["domain"],
            "sample_id": row["sample_id"],
            "goal_id": row["goal_id"],
            "tree_with_scores_path": row["tree_with_scores_path"],
        },
        "scores": [{"nodeId": str(nodes[i].get("nodeId")), "s_hat": s_hat_l[i], "n_hat": n_hat_l[i], "action": actions[i]} for i in range(len(nodes))],
        "sanitized_tree": {"nodes": sanitized_nodes},
    }


def build_model(ckpt_path: str, device: torch.device, model_type: str = DEFAULT_MODEL_TYPE) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = {}
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_type = checkpoint.get("model_type", model_type)
        config = checkpoint.get("config", {})
        state = checkpoint["model_state"]
    else:
        state = checkpoint
        if isinstance(state, dict) and "role_emb.weight" in state:
            model_type = "mlp"
    model = build_scorer(
        model_type,
        hidden=int(config.get("hidden", 128)),
        gat_hidden=int(config.get("gat_hidden", DEFAULT_GAT_HIDDEN)),
        gat_heads=int(config.get("gat_heads", DEFAULT_GAT_HEADS)),
    )
    model.model_type = model_type
    model.task_vocab = config.get("task_vocab")
    model.to(device)
    model.load_state_dict(state)
    if config.get("prior_correction", False):
        model.set_class_prior_correction([torch.tensor(w) for w in config["class_weights"]])
    model.eval()
    return model


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("Step 1/4: loading model")
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device, args.model_type)
    model_type = getattr(model, "model_type", args.model_type)

    print("Step 2/4: resolving split rows")
    manifest_rows = read_manifest(args.data_root)
    manifest_idx = index_manifest(manifest_rows)
    split_data = load_split(args.split_path)
    selected = split_data[args.predict_split]
    rows: List[Dict] = []
    for small in selected:
        k = row_key(small)
        if k in manifest_idx:
            rows.append(manifest_idx[k])
    if args.max_variants > 0:
        rows = rows[: args.max_variants]

    print(f"Step 3/4: running inference on {len(rows)} variants")
    out_jsonl = os.path.join(args.output_dir, f"predictions_{args.predict_split}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            pred = predict_variant(
                model=model,
                row=row,
                data_root=args.data_root,
                device=device,
                tau_nec=args.tau_nec,
                tau_sens=args.tau_sens,
                model_type=model_type,
            )
            f.write(json.dumps(pred) + "\n")
            if i % max(1, args.progress_every) == 0 or i == len(rows):
                print(f"Step 3/4: progress {i}/{len(rows)}")

    if args.export_first_tree and rows:
        with open(out_jsonl, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        with open(os.path.join(args.output_dir, "sanitized_tree.json"), "w", encoding="utf-8") as out_f:
            json.dump(first["sanitized_tree"], out_f, indent=2)

    print("Step 4/4: done")
    print(f"Predictions saved to {out_jsonl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MINIM inference from annotated AX trees.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split_path", type=str, default="runs/minim/split.json")
    parser.add_argument("--checkpoint", type=str, default="runs/minim/model.pt")
    parser.add_argument("--output_dir", type=str, default="runs/minim")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--predict_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--tau_nec", type=float, default=1.0)
    parser.add_argument("--tau_sens", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--export_first_tree", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

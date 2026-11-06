import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_TYPE = "gatv2"
MODEL_TYPES = ("gatv2", "mlp")
DEFAULT_GAT_HIDDEN = 496
DEFAULT_GAT_HEADS = 4
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384
GAT_NODE_DIM = 512
UI_STRUCT_DIM = GAT_NODE_DIM - MINILM_DIM
_TEXT_EMBEDDER = None


INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "checkbox",
    "combobox",
    "tab",
    "menuitem",
    "searchbox",
    "switch",
    "slider",
    "radio",
    "listbox",
}


@dataclass
class VariantData:
    variant_id: str
    goal_id: str
    domain: str
    tree_path: str
    role_ids: torch.Tensor
    goal_ids: torch.Tensor
    name_hash: torch.Tensor
    extras: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    sens: torch.Tensor
    nec: torch.Tensor


class MinimScorer(nn.Module):
    def __init__(self, role_buckets: int = 256, goal_buckets: int = 128, hidden: int = 128):
        super().__init__()
        self.role_emb = nn.Embedding(role_buckets, 16)
        self.goal_emb = nn.Embedding(goal_buckets, 16)
        self.mlp = nn.Sequential(
            nn.Linear(164, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.temp_s = nn.Parameter(torch.tensor(1.0))
        self.temp_n = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        role_ids: torch.Tensor,
        goal_ids: torch.Tensor,
        name_hash: torch.Tensor,
        extras: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        role_vec = self.role_emb(role_ids)
        goal_vec = self.goal_emb(goal_ids)
        x = torch.cat([role_vec, goal_vec, name_hash, extras], dim=-1)
        logits = self.mlp(x)
        temp_s = torch.clamp(self.temp_s, min=0.1)
        temp_n = torch.clamp(self.temp_n, min=0.1)
        s_hat = 10.0 * torch.sigmoid(logits[:, 0] / temp_s)
        n_hat = 10.0 * torch.sigmoid(logits[:, 1] / temp_n)
        return s_hat, n_hat


class GATv2Scorer(nn.Module):
    def __init__(
        self,
        input_dim: int = GAT_NODE_DIM,
        hidden: int = DEFAULT_GAT_HIDDEN,
        heads: int = DEFAULT_GAT_HEADS,
        out_bins: int = 11,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError as exc:
            raise ImportError(
                "GATv2Scorer requires torch-geometric. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        if hidden % heads != 0:
            raise ValueError(f"hidden size {hidden} must be divisible by heads {heads}")
        head_dim = hidden // heads
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(input_dim, head_dim, heads=heads, concat=True),
                GATv2Conv(hidden, head_dim, heads=heads, concat=True),
                GATv2Conv(hidden, head_dim, heads=heads, concat=True),
            ]
        )
        self.sens_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, out_bins))
        self.nec_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, out_bins))
        self.register_buffer("score_bins", torch.arange(out_bins, dtype=torch.float32))
        self.register_buffer("inference_log_weights", torch.zeros(2, out_bins), persistent=False)

    def set_class_prior_correction(self, class_weights):
        weights = torch.stack(class_weights).to(self.score_bins.device)
        if weights.shape != self.inference_log_weights.shape or not (weights > 0).any(dim=1).all():
            raise ValueError("Each scoring head needs at least one observed training class")
        # Classes absent from training must not be boosted by log(0).
        log_weights = torch.where(weights > 0, weights.log(), torch.full_like(weights, float('inf')))
        self.inference_log_weights.copy_(log_weights)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_logits, n_logits = self.forward_logits(node_features, edge_index)
        s_probs = torch.softmax(s_logits - self.inference_log_weights[0], dim=-1)
        n_probs = torch.softmax(n_logits - self.inference_log_weights[1], dim=-1)
        s_hat = torch.sum(s_probs * self.score_bins, dim=-1)
        n_hat = torch.sum(n_probs * self.score_bins, dim=-1)
        return s_hat, n_hat

    def forward_logits(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = node_features
        for layer in self.gat_layers:
            x = F.elu(layer(x, edge_index))
        return self.sens_head(x), self.nec_head(x)


def build_scorer(
    model_type: str,
    hidden: int = 128,
    gat_hidden: int = DEFAULT_GAT_HIDDEN,
    gat_heads: int = DEFAULT_GAT_HEADS,
) -> nn.Module:
    if model_type == "mlp":
        return MinimScorer(hidden=hidden)
    if model_type == "gatv2":
        return GATv2Scorer(hidden=gat_hidden, heads=gat_heads)
    raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {MODEL_TYPES}.")


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest(data_root: str) -> List[Dict]:
    manifest_path = os.path.join(data_root, "manifest.jsonl")
    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(
    rows: List[Dict],
    train_frac: float,
    val_frac: float,
    seed: int,
    group_by_sample: bool = True,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    if not group_by_sample:
        rows = rows[:]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_rows = rows[:n_train]
        val_rows = rows[n_train : n_train + n_val]
        test_rows = rows[n_train + n_val :]
        return {"train": train_rows, "val": val_rows, "test": test_rows}

    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (row["domain"], row["sample_id"])
        groups.setdefault(key, []).append(row)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    n_groups = len(group_items)
    n_train_g = int(n_groups * train_frac)
    n_val_g = int(n_groups * val_frac)

    def flatten(items: List[Tuple[Tuple[str, str], List[Dict]]]) -> List[Dict]:
        out: List[Dict] = []
        for _, vals in items:
            out.extend(vals)
        return out

    train_rows = flatten(group_items[:n_train_g])
    val_rows = flatten(group_items[n_train_g : n_train_g + n_val_g])
    test_rows = flatten(group_items[n_train_g + n_val_g :])
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def node_text(node: Dict) -> str:
    name = (node.get("name") or {}).get("value", "")
    value = (node.get("value") or {}).get("value", "")
    desc = (node.get("description") or {}).get("value", "")
    props = node.get("properties") or []
    prop_texts: List[str] = []
    for p in props:
        pname = str(p.get("name", ""))
        pval = p.get("value") or {}
        if isinstance(pval, dict):
            pval = str(pval.get("value", ""))
        else:
            pval = str(pval)
        prop_texts.append(f"{pname} {pval}")
    prop_blob = " ".join(prop_texts)
    return f"{name} {value} {desc} {prop_blob}".strip().lower()


def tokenize(text: str) -> List[str]:
    clean = "".join(c if c.isalnum() else " " for c in text)
    toks = [t for t in clean.split() if t]
    return toks


def hash_bow(tokens: List[str], dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        idx = stable_hash(tok) % dim
        vec[idx] += 1.0
    vec /= float(len(tokens))
    return vec


def get_text_embedder():
    global _TEXT_EMBEDDER
    if _TEXT_EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "GATv2 featurization requires sentence-transformers. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        model_name = os.environ.get("MINIM_EMBEDDER_PATH", MINILM_MODEL_NAME)
        _TEXT_EMBEDDER = SentenceTransformer(model_name)
    return _TEXT_EMBEDDER


def minilm_embeddings(texts: List[str]) -> np.ndarray:
    embedder = get_text_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def ui_structural_features(node: Dict, row: Dict, role: str, depth: int, max_depth: int) -> np.ndarray:
    vec = np.zeros(UI_STRUCT_DIM, dtype=np.float32)
    vec[stable_hash(role) % 64] = 1.0
    vec[64 + (stable_hash(row["goal_id"]) % 32)] = 1.0
    children = node.get("childIds") or []
    name = (node.get("name") or {}).get("value", "")
    value = (node.get("value") or {}).get("value", "")
    desc = (node.get("description") or {}).get("value", "")
    parent = node.get("parentId")
    scalars = [
        float(depth) / float(max_depth + 1e-6),
        min(1.0, math.log1p(len(children)) / 8.0),
        1.0 if role in INTERACTIVE_ROLES else 0.0,
        1.0 if value else 0.0,
        1.0 if name else 0.0,
        1.0 if desc else 0.0,
        1.0 if parent is None else 0.0,
        1.0 if not children else 0.0,
    ]
    vec[96 : 96 + len(scalars)] = np.array(scalars, dtype=np.float32)
    return vec


def build_edge_index(nodes: List[Dict]) -> torch.Tensor:
    node_ids = {str(node.get("nodeId")): i for i, node in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    for src, node in enumerate(nodes):
        for child in node.get("childIds") or []:
            dst = node_ids.get(str(child))
            if dst is not None:
                edges.append((src, dst))
                edges.append((dst, src))
    if not edges:
        edges = [(i, i) for i in range(len(nodes))]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_depth_map(nodes: List[Dict]) -> Dict[str, int]:
    id_to_node = {str(n.get("nodeId")): n for n in nodes}
    memo: Dict[str, int] = {}

    def depth_of(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        node = id_to_node.get(node_id)
        if node is None:
            memo[node_id] = 0
            return 0
        parent = node.get("parentId")
        if parent is None:
            memo[node_id] = 0
            return 0
        parent = str(parent)
        d = 1 + depth_of(parent)
        memo[node_id] = d
        return d

    for nid in id_to_node:
        depth_of(nid)
    return memo


def _featurize_variant_uncached(row: Dict, data_root: str, model_type: str = DEFAULT_MODEL_TYPE) -> VariantData:
    rel = row["tree_with_scores_path"]
    tree_path = os.path.join(data_root, rel)
    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = tree["nodes"]
    depth_map = build_depth_map(nodes)
    max_depth = max(depth_map.values()) if depth_map else 1

    role_ids: List[int] = []
    goal_ids: List[int] = []
    hashes: List[np.ndarray] = []
    extras: List[List[float]] = []
    sens: List[float] = []
    nec: List[float] = []
    node_texts: List[str] = []
    ui_features: List[np.ndarray] = []

    goal_bucket = stable_hash(row["goal_id"]) % 128
    for node in nodes:
        role = (node.get("role") or {}).get("value", "none").lower()
        role_bucket = stable_hash(role) % 256
        text = node_text(node)
        tokens = tokenize(text)
        bow = hash_bow(tokens, 128)

        nid = str(node.get("nodeId"))
        depth = depth_map.get(nid, 0)
        children = node.get("childIds") or []
        has_value = 1.0 if ((node.get("value") or {}).get("value")) else 0.0
        is_inter = 1.0 if role in INTERACTIVE_ROLES else 0.0
        extra = [
            float(depth) / float(max_depth + 1e-6),
            math.log1p(len(children)),
            is_inter,
            has_value,
        ]

        role_ids.append(role_bucket)
        goal_ids.append(goal_bucket)
        hashes.append(bow)
        extras.append(extra)
        node_texts.append(text)
        ui_features.append(ui_structural_features(node, row, role, depth, max_depth))
        sens.append(float(node.get("sensitivity_score", 0.0)))
        nec.append(float(node.get("necessity_score", 0.0)))

    if model_type == "gatv2":
        text_features = minilm_embeddings(node_texts)
        graph_features = np.concatenate([text_features, np.stack(ui_features)], axis=1)
    else:
        graph_features = np.zeros((len(nodes), GAT_NODE_DIM), dtype=np.float32)

    variant_id = f'{row["domain"]}/{row["sample_id"]}/{os.path.basename(os.path.dirname(rel))}'
    return VariantData(
        variant_id=variant_id,
        goal_id=row["goal_id"],
        domain=row["domain"],
        tree_path=tree_path,
        role_ids=torch.tensor(role_ids, dtype=torch.long),
        goal_ids=torch.tensor(goal_ids, dtype=torch.long),
        name_hash=torch.tensor(np.stack(hashes), dtype=torch.float32),
        extras=torch.tensor(np.array(extras), dtype=torch.float32),
        node_features=torch.tensor(graph_features, dtype=torch.float32),
        edge_index=build_edge_index(nodes),
        sens=torch.tensor(sens, dtype=torch.float32),
        nec=torch.tensor(nec, dtype=torch.float32),
    )


def encode_task(variant: VariantData, task_vocab: Dict[str, int] = None) -> VariantData:
    """Replace legacy task hashes with checkpoint-owned unique task IDs.

    Disk caches always contain the original feature layout. Re-encoding only
    the 128 UI columns preserves cached text embeddings and old checkpoints.
    """
    if task_vocab is None:
        return variant
    if sorted(task_vocab.values()) != list(range(len(task_vocab))):
        raise ValueError("Task vocabulary IDs must be consecutive from zero")
    if len(task_vocab) > UI_STRUCT_DIM - 64 - 8:
        raise ValueError("Task vocabulary exceeds available UI feature dimensions")
    if variant.goal_id not in task_vocab:
        raise ValueError(f"Unknown task {variant.goal_id!r}; checkpoint task vocabulary is fixed")
    features = variant.node_features.clone()
    scalars = features[:, MINILM_DIM + 96:MINILM_DIM + 104].clone()
    features[:, MINILM_DIM + 64:] = 0
    features[:, MINILM_DIM + 64 + task_vocab[variant.goal_id]] = 1
    offset = MINILM_DIM + 64 + len(task_vocab)
    features[:, offset:offset + 8] = scalars
    return replace(variant, node_features=features)


def featurize_variant(row: Dict, data_root: str, model_type: str = DEFAULT_MODEL_TYPE,
                      task_vocab: Dict[str, int] = None) -> VariantData:
    """Featurize one graph, optionally storing exact arrays in a disk cache.

    The cache is opt-in so the official small-data path stays unchanged. It
    lets full runs stream graph tensors instead of retaining all node features
    in host RAM.
    """
    cache_root = os.environ.get("MINIM_FEATURE_CACHE_DIR")
    if not cache_root:
        return encode_task(_featurize_variant_uncached(row, data_root, model_type), task_vocab)
    cache_id = hashlib.sha256(
        f"{model_type}\0{MINILM_MODEL_NAME}\0{row['tree_with_scores_path']}".encode("utf-8")
    ).hexdigest()
    cache_path = os.path.join(cache_root, cache_id[:2], cache_id + ".npz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as arrays:
            tensors = {name: torch.from_numpy(arrays[name].copy()) for name in (
                "role_ids", "goal_ids", "name_hash", "extras", "node_features",
                "edge_index", "sens", "nec"
            )}
        rel = row["tree_with_scores_path"]
        return encode_task(VariantData(
            variant_id=f"{row['domain']}/{row['sample_id']}/{os.path.basename(os.path.dirname(rel))}",
            goal_id=row["goal_id"], domain=row["domain"],
            tree_path=os.path.join(data_root, rel), **tensors,
        ), task_vocab)

    variant = _featurize_variant_uncached(row, data_root, model_type)
    np.savez(
        cache_path,
        role_ids=variant.role_ids.numpy(), goal_ids=variant.goal_ids.numpy(),
        name_hash=variant.name_hash.numpy(), extras=variant.extras.numpy(),
        node_features=variant.node_features.numpy(), edge_index=variant.edge_index.numpy(),
        sens=variant.sens.numpy(), nec=variant.nec.numpy(),
    )
    return encode_task(variant, task_vocab)


def collate_variants(batch: List[VariantData], device: torch.device) -> Dict[str, torch.Tensor]:
    role_ids = torch.cat([x.role_ids for x in batch]).to(device)
    goal_ids = torch.cat([x.goal_ids for x in batch]).to(device)
    name_hash = torch.cat([x.name_hash for x in batch]).to(device)
    extras = torch.cat([x.extras for x in batch]).to(device)
    node_features = torch.cat([x.node_features for x in batch]).to(device)
    edge_parts: List[torch.Tensor] = []
    offset = 0
    for variant in batch:
        edge_parts.append(variant.edge_index + offset)
        offset += int(variant.node_features.shape[0])
    edge_index = torch.cat(edge_parts, dim=1).to(device)
    sens = torch.cat([x.sens for x in batch]).to(device)
    nec = torch.cat([x.nec for x in batch]).to(device)
    return {
        "role_ids": role_ids,
        "goal_ids": goal_ids,
        "name_hash": name_hash,
        "extras": extras,
        "node_features": node_features,
        "edge_index": edge_index,
        "sens": sens,
        "nec": nec,
    }


def score_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, GATv2Scorer):
        return model(batch["node_features"], batch["edge_index"])
    return model(batch["role_ids"], batch["goal_ids"], batch["name_hash"], batch["extras"])


def score_variant(model: nn.Module, variant: VariantData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = {
        "role_ids": variant.role_ids.to(device),
        "goal_ids": variant.goal_ids.to(device),
        "name_hash": variant.name_hash.to(device),
        "extras": variant.extras.to(device),
        "node_features": variant.node_features.to(device),
        "edge_index": variant.edge_index.to(device),
    }
    return score_batch(model, batch)


def ci_loss(
    s_true: torch.Tensor,
    n_true: torch.Tensor,
    s_hat: torch.Tensor,
    n_hat: torch.Tensor,
    alpha: float = 1.0,
    ci_lambda: float = 1.0,
    sens_high_boost: float = 2.0,
    nec_low_boost: float = 2.0,
    nec_high_boost: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_weights = 1.0 + sens_high_boost * (s_true >= 5.0).float()
    l_sens = torch.mean(s_weights * torch.abs(s_true - s_hat))
    weights = 1.0 + ci_lambda * (s_true / 10.0) * (1.0 - (n_true / 10.0))
    weights = weights + nec_low_boost * (n_true <= 1.0).float() + nec_high_boost * (n_true >= 7.0).float()
    l_nec = torch.mean(weights * torch.abs(n_true - n_hat))
    total = l_sens + alpha * l_nec
    return total, l_sens, l_nec


def ci_cross_entropy(s_true, n_true, s_logits, n_logits, alpha=1.0, ci_lambda=1.0,
                     sens_high_boost=0.0, nec_low_boost=0.0, nec_high_boost=0.0):
    """Diagnostic alternative: same CI weights/node mean, categorical targets.

    This is not the loss reported in the paper. It deliberately omits class
    balancing and does not require inference prior correction.
    """
    for labels in (s_true, n_true):
        valid = torch.isfinite(labels) & (labels >= 0) & (labels <= 10) & (labels == labels.round())
        if not bool(valid.all()):
            raise ValueError("CI cross entropy requires integer labels in [0, 10]")
    s_weights = 1 + sens_high_boost * (s_true >= 5).float()
    n_weights = 1 + ci_lambda * s_true / 10 * (1 - n_true / 10)
    n_weights = n_weights + nec_low_boost * (n_true <= 1).float() + nec_high_boost * (n_true >= 7).float()
    ls = (s_weights * F.cross_entropy(s_logits, s_true.long(), reduction="none")).mean()
    ln = (n_weights * F.cross_entropy(n_logits, n_true.long(), reduction="none")).mean()
    return ls + alpha * ln, ls, ln


def label_balance_weights(variants: Iterable[VariantData]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse class frequencies fitted on training labels, never validation/test."""
    histograms = [torch.zeros(11), torch.zeros(11)]
    for variant in variants:
        for index, attr in enumerate(("sens", "nec")):
            labels = getattr(variant, attr)
            if not torch.equal(labels, labels.round()) or labels.min() < 0 or labels.max() > 10:
                raise ValueError("Balanced losses require integer labels in [0, 10]")
            histograms[index] += torch.bincount(labels.long().cpu(), minlength=11)
    result = []
    for counts in histograms:
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts.sum() / (present.sum() * counts[present])
        result.append(weights)
    return tuple(result)


def balanced_score_loss(s_true, n_true, s_logits, n_logits, class_weights,
                        loss_type="balanced_ce", alpha=1.0, ci_lambda=1.0):
    """Balanced CE or L1 using the same train-fitted per-class weights."""
    if loss_type not in ("balanced_ce", "balanced_l1"):
        raise ValueError(f"Unsupported balanced objective: {loss_type}")
    terms = []
    for index, (labels, logits) in enumerate(((s_true, s_logits), (n_true, n_logits))):
        target = labels.long()
        weights = class_weights[index].to(logits.device)[target]
        if index == 1:
            weights = weights * (1 + ci_lambda * s_true / 10 * (1 - n_true / 10))
        if loss_type == "balanced_ce":
            errors = F.cross_entropy(logits, target, reduction="none")
        else:
            bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
            predictions = (logits.softmax(-1) * bins).sum(-1)
            errors = (predictions - labels).abs()
        terms.append((weights * errors).sum() / weights.sum().clamp_min(1e-12))
    return terms[0] + alpha * terms[1], terms[0], terms[1]


def evaluate_variant_policy(variant: VariantData, s_hat: torch.Tensor, n_hat: torch.Tensor) -> Tuple[float, float]:
    tau_crit = 7.0
    tau_irr = 1.0
    tau_nec = 1.0
    tau_sens = 5.0
    tok_ph = 6.0
    p_abs = 0.05

    with open(variant.tree_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)["nodes"]

    gt_n = variant.nec.cpu().numpy()
    gt_s = variant.sens.cpu().numpy()
    s_pred = s_hat.detach().cpu().numpy()
    n_pred = n_hat.detach().cpu().numpy()

    actions: List[str] = []
    for sp, npred in zip(s_pred, n_pred):
        if npred < tau_nec:
            actions.append("R")
        elif sp >= tau_sens:
            actions.append("A")
        else:
            actions.append("K")

    critical = np.where(gt_n >= tau_crit)[0]
    if len(critical) == 0:
        tcnp = 1.0
    else:
        kept = sum(actions[i] != "R" for i in critical)
        tcnp = kept / float(len(critical))

    def node_tok(node: Dict) -> int:
        t = node_text(node)
        return max(1, len(t.split())) if t else 1

    leak = 0.0
    full = 0.0
    for i, node in enumerate(nodes):
        if gt_n[i] <= tau_irr:
            tok = float(node_tok(node))
            full += gt_s[i] * tok
            if actions[i] == "K":
                post = tok
            elif actions[i] == "A":
                post = max(tok_ph, p_abs * tok)
            else:
                post = 0.0
            leak += gt_s[i] * post
    norm_tisl = 0.0 if full <= 1e-9 else (leak / full)
    return tcnp, norm_tisl


def save_split(split: Dict[str, List[Dict]], run_dir: str) -> None:
    split_out = {
        key: [
            {
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "goal_id": row["goal_id"],
                "tree_with_scores_path": row["tree_with_scores_path"],
            }
            for row in rows
        ]
        for key, rows in split.items()
    }
    with open(os.path.join(run_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_out, f, indent=2)


def train(args: argparse.Namespace) -> None:
    print("Step 1/5: loading manifest")
    rows = read_manifest(args.data_root)
    if args.max_variants > 0:
        rows = rows[: args.max_variants]
    split = split_rows(
        rows,
        args.train_frac,
        args.val_frac,
        args.seed,
        group_by_sample=True,
    )
    tr_samples = {(r["domain"], r["sample_id"]) for r in split["train"]}
    te_samples = {(r["domain"], r["sample_id"]) for r in split["test"]}
    overlap = len(tr_samples & te_samples)
    assert overlap == 0, f"train-test sample overlap {overlap} > 0; refusing to train under leakage."
    print(f"Step 1/5: sample overlap train-test = {overlap}")
    if args.task_encoding == "unique" and args.model_type != "gatv2":
        raise ValueError("Unique task features are currently implemented for GATv2")
    if args.loss_type != "ci_l1" and args.model_type != "gatv2":
        raise ValueError("Balanced logit losses require GATv2")
    if args.prior_correction and args.loss_type != "balanced_ce":
        raise ValueError("Class-prior correction is only defined for balanced CE")
    args.task_vocab = ({goal: i for i, goal in enumerate(sorted({r['goal_id'] for r in split['train']}))}
                       if args.task_encoding == "unique" else None)
    def features(row):
        return featurize_variant(row, args.data_root, args.model_type, args.task_vocab)

    print("Step 2/5: featurizing variants")
    disk_cache = bool(os.environ.get("MINIM_FEATURE_CACHE_DIR"))
    if disk_cache:
        train_data = split["train"]
        val_data = split["val"]
        all_rows = train_data + val_data
        for i, row in enumerate(all_rows, 1):
            features(row)
            if i % 100 == 0 or i == len(all_rows):
                print(f"Step 2/5: cached {i}/{len(all_rows)} train/validation graphs", flush=True)
    else:
        train_data = [features(r) for r in split["train"]]
        val_data = [features(r) for r in split["val"]]

    device = torch.device(args.device)
    model = build_scorer(args.model_type, hidden=args.hidden, gat_hidden=args.gat_hidden, gat_heads=args.gat_heads).to(device)
    model.task_vocab = args.task_vocab
    class_weights = None
    if args.loss_type in ("balanced_l1", "balanced_ce"):
        class_weights = label_balance_weights((features(r) for r in train_data) if disk_cache else train_data)
        args.class_weights = [w.tolist() for w in class_weights]
        if args.prior_correction:
            model.set_class_prior_correction(class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Step 3/5: training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_data)))
        random.shuffle(order)
        train_losses: List[float] = []

        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            batch_variants = [
                features(train_data[j])
                if disk_cache else train_data[j]
                for j in idx
            ]
            batch = collate_variants(batch_variants, device)
            if args.loss_type != "ci_l1":
                s_logits, n_logits = model.forward_logits(batch["node_features"], batch["edge_index"])
                if args.loss_type == "ci_ce":
                    loss, _, _ = ci_cross_entropy(batch["sens"], batch["nec"], s_logits, n_logits,
                        args.alpha, args.ci_lambda, args.sens_high_boost, args.nec_low_boost, args.nec_high_boost)
                else:
                    loss, _, _ = balanced_score_loss(batch["sens"], batch["nec"], s_logits, n_logits,
                                                     class_weights, args.loss_type, args.alpha, args.ci_lambda)
            else:
                s_hat, n_hat = score_batch(model, batch)
                loss, _, _ = ci_loss(
                    batch["sens"], batch["nec"], s_hat, n_hat,
                    alpha=args.alpha, ci_lambda=args.ci_lambda,
                    sens_high_boost=args.sens_high_boost,
                    nec_low_boost=args.nec_low_boost,
                    nec_high_boost=args.nec_high_boost,
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: List[float] = []
        val_tcnp: List[float] = []
        val_tisl: List[float] = []
        with torch.no_grad():
            for item in val_data:
                variant = features(item) if disk_cache else item
                s_true = variant.sens.to(device)
                n_true = variant.nec.to(device)
                s_hat, n_hat = score_variant(model, variant, device)
                if args.loss_type != "ci_l1":
                    s_logits, n_logits = model.forward_logits(variant.node_features.to(device), variant.edge_index.to(device))
                    if args.loss_type == "ci_ce":
                        loss, _, _ = ci_cross_entropy(s_true, n_true, s_logits, n_logits,
                            args.alpha, args.ci_lambda, args.sens_high_boost, args.nec_low_boost, args.nec_high_boost)
                    else:
                        loss, _, _ = balanced_score_loss(s_true, n_true, s_logits, n_logits,
                                                         class_weights, args.loss_type, args.alpha, args.ci_lambda)
                else:
                    loss, _, _ = ci_loss(
                        s_true, n_true, s_hat, n_hat,
                        alpha=args.alpha, ci_lambda=args.ci_lambda,
                        sens_high_boost=args.sens_high_boost,
                        nec_low_boost=args.nec_low_boost,
                        nec_high_boost=args.nec_high_boost,
                    )
                val_losses.append(float(loss.detach().cpu()))
                vtcnp, vtisl = evaluate_variant_policy(variant, s_hat, n_hat)
                val_tcnp.append(vtcnp)
                val_tisl.append(vtisl)

        print(
            f"Step 3/5: epoch {epoch}/{args.epochs} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_loss={np.mean(val_losses):.4f} | "
            f"val_tcnp={np.mean(val_tcnp):.4f} | "
            f"val_tisl={np.mean(val_tisl):.4f}"
        )

    print("Step 4/5: saving checkpoint and split")
    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)
    torch.save(
        {
            "model_type": args.model_type,
            "model_state": model.state_dict(),
            "config": vars(args),
        },
        os.path.join(run_dir, "model.pt"),
    )
    save_split(split, run_dir)
    with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Step 5/5: done")
    print(f"Saved checkpoint: {os.path.join(run_dir, 'model.pt')}")
    print(f"Saved split: {os.path.join(run_dir, 'split.json')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MINIM scorer from annotated AX-tree manifest.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="runs/minim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--gat_hidden", type=int, default=DEFAULT_GAT_HIDDEN)
    parser.add_argument("--gat_heads", type=int, default=DEFAULT_GAT_HEADS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ci_lambda", type=float, default=1.0)
    parser.add_argument("--sens_high_boost", type=float, default=0.0)
    parser.add_argument("--nec_low_boost", type=float, default=0.0)
    parser.add_argument("--nec_high_boost", type=float, default=0.0)
    parser.add_argument("--max_variants", type=int, default=0)
    parser.add_argument("--task_encoding", choices=("legacy", "unique"), default="legacy")
    parser.add_argument("--loss_type", choices=("ci_l1", "ci_ce", "balanced_l1", "balanced_ce"), default="ci_l1")
    parser.add_argument("--prior_correction", action="store_true",
                        help="Undo train class weights in CE logits before policy scoring")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    set_seed(args.seed)
    train(args)

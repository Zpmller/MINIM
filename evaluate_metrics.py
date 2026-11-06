import math
from typing import Dict, List, Sequence, Tuple


TAU_CRIT = 7.0
TAU_IRR = 1.0
TOK_PH = 6.0
P_ABS = 0.05
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
    return f"{name} {value} {desc} {prop_blob}".strip()


def token_count(node: Dict) -> float:
    text = node_text(node)
    if not text:
        return 1.0
    return float(max(1, len(text.split())))


def is_interactive(node: Dict) -> bool:
    role = ((node.get("role") or {}).get("value", "none") or "none").lower()
    return role in INTERACTIVE_ROLES


def tok_post(tok: float, action: str, tok_ph: float = TOK_PH, p_abs: float = P_ABS) -> float:
    if action == "K":
        return tok
    if action == "A":
        return max(tok_ph, p_abs * tok)
    return 0.0


def compute_tcnp(nodes: Sequence[Dict], gt_nec: Sequence[float], actions: Sequence[str], tau_crit: float = TAU_CRIT) -> float:
    crit_idx = [i for i, n in enumerate(gt_nec) if n >= tau_crit]
    if not crit_idx:
        return 1.0
    keep = sum(1 for i in crit_idx if actions[i] != "R")
    return keep / float(len(crit_idx))


def compute_tcnp_i(
    nodes: Sequence[Dict],
    gt_nec: Sequence[float],
    actions: Sequence[str],
    tau_crit: float = TAU_CRIT,
) -> float:
    crit_inter_idx = [i for i, n in enumerate(gt_nec) if n >= tau_crit and is_interactive(nodes[i])]
    if not crit_inter_idx:
        return 1.0
    keep = sum(1 for i in crit_inter_idx if actions[i] != "R")
    return keep / float(len(crit_inter_idx))


def compute_tisl_numerator(
    nodes: Sequence[Dict],
    gt_sens: Sequence[float],
    gt_nec: Sequence[float],
    actions: Sequence[str],
    tau_irr: float = TAU_IRR,
    tok_ph: float = TOK_PH,
    p_abs: float = P_ABS,
) -> float:
    leakage = 0.0
    for i, node in enumerate(nodes):
        if gt_nec[i] <= tau_irr and actions[i] != "R":
            tok = token_count(node)
            leakage += gt_sens[i] * tok_post(tok, actions[i], tok_ph=tok_ph, p_abs=p_abs)
    return leakage


def compute_tisl_denominator(
    nodes: Sequence[Dict],
    gt_sens: Sequence[float],
    gt_nec: Sequence[float],
    tau_irr: float = TAU_IRR,
) -> float:
    full = 0.0
    for i, node in enumerate(nodes):
        if gt_nec[i] <= tau_irr:
            full += gt_sens[i] * token_count(node)
    return full


def aggregate_dataset_metrics(
    per_variant: Sequence[Tuple[Sequence[Dict], Sequence[float], Sequence[float], Sequence[str]]]
) -> Dict[str, float]:
    tcnp_vals: List[float] = []
    tcnp_i_vals: List[float] = []
    tisl_num = 0.0
    tisl_den = 0.0
    keep_rates: List[float] = []
    abs_rates: List[float] = []

    for nodes, gt_s, gt_n, actions in per_variant:
        n_nodes = max(1, len(nodes))
        tcnp_vals.append(compute_tcnp(nodes, gt_n, actions))
        tcnp_i_vals.append(compute_tcnp_i(nodes, gt_n, actions))
        tisl_num += compute_tisl_numerator(nodes, gt_s, gt_n, actions)
        tisl_den += compute_tisl_denominator(nodes, gt_s, gt_n)
        keep_rates.append(sum(1 for a in actions if a == "K") / float(n_nodes))
        abs_rates.append(sum(1 for a in actions if a == "A") / float(n_nodes))

    norm_tisl = 0.0 if tisl_den <= 1e-9 else (tisl_num / tisl_den)
    return {
        "tcnp": sum(tcnp_vals) / float(max(1, len(tcnp_vals))),
        "tcnp_i": sum(tcnp_i_vals) / float(max(1, len(tcnp_i_vals))),
        "tisl": norm_tisl,
        "keep_rate": sum(keep_rates) / float(max(1, len(keep_rates))),
        "abstract_rate": sum(abs_rates) / float(max(1, len(abs_rates))),
    }

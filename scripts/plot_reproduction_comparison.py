#!/usr/bin/env python3
"""Plot reproduced versus paper Table 1 metrics from saved CSV artifacts."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
refs = {}
with (root / "results/paper_reference.csv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["table"] == "Table1":
            refs.setdefault(row["method_or_group"], {})[row["metric"].lower().replace("-", "_")] = float(row["value"])
actual = {}
with (root / "results/table1.csv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        actual[row["method"]] = {k: float(row[k]) for k in ("tcnp", "tcnp_i", "tisl")}

methods = ["Random Budget", "Sensitivity-Only", "Necessity-Only", "MINIM"]
metrics = ["tcnp", "tcnp_i", "tisl"]
labels = ["TCNP", "TCNP-I", "TISL"]
x = np.arange(len(metrics))
width = 0.18
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, method in enumerate(methods):
    paper = [refs[method][m] for m in metrics]
    reproduced = [actual[method][m] for m in metrics]
    offset = (i - (len(methods) - 1) / 2) * width
    ax.bar(x + offset - width * .24, paper, width * .48, color=f"C{i}", alpha=.38,
           label=f"Paper {method}" if i == 0 else None)
    ax.bar(x + offset + width * .24, reproduced, width * .48, color=f"C{i}", alpha=.95,
           label=f"Reproduction {method}" if i == 0 else None)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Rate")
ax.set_title("MINIM Table 1: paper values and full-data reproduction")
ax.grid(axis="y", alpha=.25)
ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, -0.12))
fig.tight_layout()
out = root / "results/table1_paper_vs_reproduction.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)

#!/usr/bin/env python3
"""Extract epoch metrics from the official trainer log and plot loss curves."""
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

log = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/train_full.log")
pattern = re.compile(r"epoch (\d+)/(\d+) \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| val_tcnp=([\d.]+) \| val_tisl=([\d.]+)")
rows = []
for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    match = pattern.search(line)
    if match:
        epoch, total, train, val, tcnp, tisl = match.groups()
        rows.append({"epoch": int(epoch), "epochs": int(total), "train_loss": float(train),
                     "val_loss": float(val), "val_tcnp": float(tcnp), "val_tisl": float(tisl)})
if not rows:
    raise SystemExit(f"No epoch metrics found in {log}")
out = Path("results")
out.mkdir(exist_ok=True)
with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], marker="o", label="Train loss")
ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], marker="s", label="Validation loss")
ax.set(xlabel="Epoch", ylabel="CI weighted L1 loss", title="MINIM GATv2 training")
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(out / "training_loss.png", dpi=180)
print(f"Saved {out / 'training_history.csv'} and {out / 'training_loss.png'}")

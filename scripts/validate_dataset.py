#!/usr/bin/env python3
"""Validate MINIM manifest paths, score labels, and per-domain coverage."""
import collections
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
rows = [json.loads(line) for line in (root / "manifest.jsonl").open(encoding="utf-8") if line.strip()]
domains = collections.Counter(r["domain"] for r in rows)
missing, bad_json, bad_labels = [], [], []
score_counts = {"sensitivity_score": collections.Counter(), "necessity_score": collections.Counter()}
node_counts = []
for row in rows:
    path = root / row["tree_with_scores_path"]
    if not path.is_file():
        missing.append(str(path))
        continue
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
        nodes = tree["nodes"]
    except Exception as exc:
        bad_json.append((str(path), str(exc)))
        continue
    node_counts.append(len(nodes))
    for i, node in enumerate(nodes):
        for label in score_counts:
            try:
                value = float(node[label])
                if not math.isfinite(value) or not 0 <= value <= 10:
                    raise ValueError(f"out of range: {value}")
                score_counts[label][value] += 1
            except Exception as exc:
                bad_labels.append((str(path), i, label, str(exc)))

report = {
    "manifest_rows": len(rows), "unique_tree_paths": len({r["tree_with_scores_path"] for r in rows}),
    "domain_counts": dict(domains), "nodes_validated": sum(node_counts),
    "node_count_mean": (sum(node_counts) / len(node_counts) if node_counts else 0),
    "score_label_histograms": {k: dict(sorted(v.items())) for k, v in score_counts.items()},
    "missing_files": missing[:20], "missing_count": len(missing),
    "bad_json_count": len(bad_json), "bad_label_count": len(bad_labels),
}
out = Path("results/dataset_validation.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if missing or bad_json or bad_labels:
    raise SystemExit("Dataset validation failed")

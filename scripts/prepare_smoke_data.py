#!/usr/bin/env python3
"""Create a 30-variant, domain-balanced smoke manifest from the full data."""
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data"
target = root / "data_smoke"
rows = [json.loads(line) for line in (source / "manifest.jsonl").open(encoding="utf-8") if line.strip()]
available = [r for r in rows if (source / r["tree_with_scores_path"]).is_file()]
selected = []
groups = defaultdict(set)
for row in available:
    domain, key = row["domain"], row["sample_id"]
    if len(groups[domain]) < 10 and key not in groups[domain]:
        selected.append(row)
        groups[domain].add(key)
if any(len(groups[d]) < 10 for d in ("gmail", "reddit", "shopping")):
    selected, groups = [], defaultdict(set)
    for row in available:
        domain, key = row["domain"], row["sample_id"]
        if key not in groups[domain]:
            selected.append(row)
            groups[domain].add(key)
        if len(selected) == 30:
            break
if len(selected) < 10:
    raise SystemExit(f"Need at least 10 downloaded sample groups; found {len(selected)}")
target.mkdir(exist_ok=True)
if not (target / "samples").exists():
    (target / "samples").symlink_to(source / "samples", target_is_directory=True)
(target / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected), encoding="utf-8")
print(f"Prepared {len(selected)} smoke variants from downloaded data: {dict((d, len(v)) for d, v in groups.items())}")

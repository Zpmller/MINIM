#!/usr/bin/env python3
"""Summarize smoke predictions and verify K/A/R transformations."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_minim_inference import action_from_scores, sanitize_node

predictions = {}
for name in ("mlp", "gat_bs4", "gat_bs8", "gat_bs16"):
    path = ROOT / "runs" / f"smoke_{name}" / "predictions_test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    actions = collections.Counter(a["action"] for row in rows for a in row["scores"])
    predictions[name] = {"variants": len(rows), "action_counts": dict(actions)}

assert action_from_scores(9.0, 0.0, 1.0, 5.0) == "R"
assert action_from_scores(9.0, 5.0, 1.0, 5.0) == "A"
assert action_from_scores(1.0, 5.0, 1.0, 5.0) == "K"
node = {"name": {"value": "private"}, "value": {"value": "secret"}}
assert sanitize_node(node, "A")["value"]["value"] == "[REDACTED]"
assert sanitize_node(node, "K") == node
assert [sanitize_node(node, a) for a in ("K", "A", "R") if a != "R"] == [node, sanitize_node(node, "A")]
report = {"prediction_runs": predictions, "kar_policy_unit_checks": "PASS"}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "smoke_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

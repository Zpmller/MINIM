#!/usr/bin/env python3
"""Create a non-destructive 27-task view of the public MINIM dataset."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data"
target = ROOT / "data_paper_aligned"
manifest = source / "manifest.jsonl"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
kept = [row for row in rows if not str(row["goal_id"]).startswith("T_")]
dropped = [row for row in rows if str(row["goal_id"]).startswith("T_")]
if len(kept) != 5400 or len({r["goal_id"] for r in kept}) != 27:
    raise SystemExit(f"Unexpected public dataset contents: kept={len(kept)} tasks={len({r['goal_id'] for r in kept})}")
target.mkdir(exist_ok=True)
(target / "manifest.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
)
sample_link = target / "samples"
expected = Path("../data/samples")
if sample_link.is_symlink():
    if sample_link.readlink() != expected:
        raise SystemExit(f"Refusing to replace unexpected symlink: {sample_link} -> {sample_link.readlink()}")
elif sample_link.exists():
    raise SystemExit(f"Refusing to replace existing path: {sample_link}")
else:
    sample_link.symlink_to(expected, target_is_directory=True)

meta = {
    "source": "data/manifest.jsonl",
    "filter": "retain rows whose goal_id does not start with 'T_'",
    "public_manifest_rows": len(rows),
    "retained_rows": len(kept),
    "retained_task_ids": sorted({r["goal_id"] for r in kept}),
    "retained_rows_by_domain": dict(collections.Counter(r["domain"] for r in kept)),
    "excluded_rows": len(dropped),
    "excluded_task_ids": sorted({r["goal_id"] for r in dropped}),
    "excluded_rows_by_task": dict(collections.Counter(r["goal_id"] for r in dropped)),
    "note": "Raw data/ is unchanged. Filtered rows reference tree files through a relative symlink.",
}
(target / "cleaning_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("public_manifest_rows", "retained_rows", "retained_rows_by_domain", "excluded_rows")}, ensure_ascii=False))

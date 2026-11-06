#!/usr/bin/env python3
"""Download the pinned public MINIM dataset snapshot with resumable file checks."""
import concurrent.futures
import json
import os
import urllib.error
import urllib.request
import time
from pathlib import Path

REPO = "Chaoyu112358/MINIM-data"
REVISION = "fd1786cb08dc42173905588e7ca015de968c8134"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.jsonl"


def download(path: str) -> tuple[str, str]:
    dest = DATA / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return path, "cached"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}?download=true"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MINIM-reproduction/1.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
                while True:
                    block = r.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            tmp.replace(dest)
            return path, "downloaded"
        except Exception:
            if attempt == 6:
                raise
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 30))
    return path, "failed"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        download("manifest.jsonl")
    with MANIFEST.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    paths = sorted({r["tree_with_scores_path"] for r in rows})
    failures = []
    done = 0
    workers = int(os.environ.get("MINIM_DOWNLOAD_WORKERS", "8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, path): path for path in paths}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], repr(exc)))
            done += 1
            if done % 100 == 0 or done == len(paths):
                print(f"downloaded {done}/{len(paths)}; failures={len(failures)}", flush=True)
    if failures:
        (ROOT / "logs" / "dataset_download_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"{len(failures)} files failed; rerun to retry")
    print(f"Complete: {len(paths)} tree files + manifest at {DATA}")


if __name__ == "__main__":
    main()

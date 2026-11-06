#!/usr/bin/env python3
"""Download an HTTP range-capable file in parallel chunks with retries."""
import concurrent.futures
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: parallel_download.py URL OUTPUT [workers=16] [sha256]")
    url, output = sys.argv[1:3]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    expected_hash = sys.argv[4] if len(sys.argv) > 4 else ""
    head = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(head, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        range_ok = "bytes" in response.headers.get("Accept-Ranges", "").lower()
    if not range_ok:
        raise SystemExit("server does not advertise byte ranges")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out.with_name(out.name + ".parts")
    part_dir.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024
    pieces = [(start, min(size, start + chunk) - 1) for start in range(0, size, chunk)]

    def fetch(item):
        index, (start, end) = item
        part = part_dir / f"{index:05d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return index, expected
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={
                    "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                    "User-Agent": "MINIM-reproduction/1.0",
                })
                with urllib.request.urlopen(req, timeout=180) as response:
                    if response.status != 206:
                        raise RuntimeError(f"range request status {response.status}")
                    data = response.read()
                if len(data) != expected:
                    raise RuntimeError(f"range length {len(data)} != {expected}")
                part.write_bytes(data)
                return index, expected
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 20))

    complete = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, (i, pair)) for i, pair in enumerate(pieces)]
        for future in concurrent.futures.as_completed(futures):
            _, n = future.result()
            complete += n
            print(f"downloaded {complete}/{size} bytes", flush=True)
    temp = out.with_suffix(out.suffix + ".assembling")
    with temp.open("wb") as destination:
        for i in range(len(pieces)):
            with (part_dir / f"{i:05d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
    if temp.stat().st_size != size:
        raise SystemExit(f"assembled length {temp.stat().st_size} != {size}")
    if expected_hash:
        hasher = hashlib.sha256()
        with temp.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != expected_hash:
            raise SystemExit(f"sha256 mismatch: {digest} != {expected_hash}")
    temp.replace(out)
    print(f"saved {out} ({size} bytes)")


if __name__ == "__main__":
    main()

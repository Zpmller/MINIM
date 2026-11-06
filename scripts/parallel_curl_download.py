#!/usr/bin/env python3
"""Range-download an HTTP file through curl, useful for HF's Xet CDN endpoints."""
import concurrent.futures
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: parallel_curl_download.py URL OUTPUT [workers=16] [sha256] [size]")
    url, output = sys.argv[1:3]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    expected_hash = sys.argv[4] if len(sys.argv) > 4 else ""
    known_size = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    if known_size:
        size = known_size
    else:
        headers = subprocess.check_output(["curl", "-sSIL", "-L", "--max-time", "60", url], text=True)
        lengths = re.findall(r"(?im)^content-length:\s*(\d+)", headers)
        if not lengths:
            raise SystemExit("could not determine content length")
        size = int(lengths[-1])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = out.with_name(out.name + ".parts")
    parts.mkdir(exist_ok=True)
    chunk = 4 * 1024 * 1024
    ranges = [(start, min(size, start + chunk) - 1) for start in range(0, size, chunk)]

    def fetch(item):
        i, (start, end) = item
        part = parts / f"{i:05d}.part"
        wanted = end - start + 1
        if part.exists() and part.stat().st_size == wanted:
            return wanted
        for attempt in range(10):
            try:
                result = subprocess.run([
                    "curl", "-fLsS", "-L", "--retry", "3", "--retry-all-errors",
                    "--connect-timeout", "30", "--max-time", "240",
                    "-H", f"Range: bytes={start}-{end}", "-o", str(part), url,
                ], capture_output=True, text=True)
                if result.returncode == 0 and part.exists() and part.stat().st_size == wanted:
                    return wanted
                raise RuntimeError(result.stderr[-400:] or f"bad chunk size {part.stat().st_size if part.exists() else 0}/{wanted}")
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 20))

    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, (i, pair)) for i, pair in enumerate(ranges)]
        for f in concurrent.futures.as_completed(futures):
            total += f.result()
            print(f"downloaded {total}/{size} bytes", flush=True)
    temp = out.with_suffix(out.suffix + ".assembling")
    with temp.open("wb") as dest:
        for i in range(len(ranges)):
            with (parts / f"{i:05d}.part").open("rb") as src:
                while block := src.read(1024 * 1024):
                    dest.write(block)
    if temp.stat().st_size != size:
        raise SystemExit(f"assembled {temp.stat().st_size}, expected {size}")
    if expected_hash:
        h = hashlib.sha256()
        with temp.open("rb") as src:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected_hash:
            raise SystemExit(f"SHA-256 mismatch: {h.hexdigest()}")
    temp.replace(out)
    print(f"saved {out} ({size} bytes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parallel range-download the Linux CUDA wheels required by the pinned torch wheel."""
import concurrent.futures
import html.parser
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/tmp/minim-wheelhouse")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128/"
PACKAGES = {
    "torch": "2.10.0+cu128",
    "cuda-bindings": "12.9.4",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cuda-cupti-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-cusparse-cu12": "12.5.8.93",
    "nvidia-cusparselt-cu12": "0.7.1",
    "nvidia-nccl-cu12": "2.27.5",
    "nvidia-nvshmem-cu12": "3.4.5",
    "nvidia-nvtx-cu12": "12.8.90",
    "nvidia-nvjitlink-cu12": "12.8.93",
    "nvidia-cufile-cu12": "1.13.1.3",
    "triton": "3.6.0",
}


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def resolve(name, version):
    if name == "triton":
        meta = urllib.request.urlopen(f"https://pypi.org/pypi/triton/{version}/json", timeout=60)
        import json
        files = json.load(meta)["urls"]
        candidates = [x for x in files if "cp310" in x["filename"] and "x86_64" in x["filename"]]
        if not candidates:
            raise RuntimeError("no matching triton wheel")
        f = candidates[0]
        return f["url"], f["digests"]["sha256"], f["filename"]
    page = urllib.request.urlopen(TORCH_INDEX + name + "/", timeout=90).read().decode()
    parser = Links()
    parser.feed(page)
    candidates = []
    for href in parser.hrefs:
        filename = urllib.parse.unquote(urllib.parse.urlsplit(href).path.rsplit("/", 1)[-1])
        if version in filename and "x86_64" in filename and filename.endswith(".whl"):
            if name == "cuda-bindings" and "cp310" not in filename:
                continue
            candidates.append((href, filename))
    if not candidates:
        raise RuntimeError(f"no Linux wheel for {name}=={version}")
    href, filename = candidates[0]
    frag = urllib.parse.parse_qs(urllib.parse.urlsplit(href).fragment)
    return href.split("#", 1)[0], frag["sha256"][0], filename


def run(item):
    name, version = item
    url, digest, filename = resolve(name, version)
    out = ROOT / filename
    cmd = [sys.executable, "scripts/parallel_download.py", url, str(out), "8", digest]
    print(f"Starting {name}=={version} -> {out.name}", flush=True)
    subprocess.run(cmd, check=True)
    return name, version, str(out)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, item) for item in PACKAGES.items()]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print("Complete", result[0], flush=True)
    print("Downloaded", len(results), "CUDA dependency wheels")


if __name__ == "__main__":
    main()

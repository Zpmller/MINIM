#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-$HOME/.local/bin/uv}"
"$UV" venv --seed --python 3.10 .venv
.venv/bin/python scripts/download_torch_cuda_deps.py
wheels=()
for wheel in /tmp/minim-wheelhouse/*.whl; do
  [[ "$wheel" == *nvidia_cuda_nvrtc_cu12-12.8.93.whl ]] && continue
  wheels+=("$wheel")
done
.venv/bin/pip install --no-deps "${wheels[@]}"
.venv/bin/pip install --retries 20 --timeout 120 filelock 'typing-extensions>=4.10.0' \
  'sympy>=1.13.3' 'networkx>=2.5.1' jinja2 'fsspec>=0.8.5'
.venv/bin/pip install --retries 20 --timeout 120 \
  torch-geometric sentence-transformers matplotlib numpy
.venv/bin/pip freeze > reports/environment-freeze.txt

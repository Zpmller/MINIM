#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
DEST="${MINIM_EMBEDDER_PATH:-$PWD/data/models/all-MiniLM-L6-v2}"
FILES=(
  config.json config_sentence_transformers.json data_config.json modules.json
  sentence_bert_config.json special_tokens_map.json tokenizer.json tokenizer_config.json
  vocab.txt 1_Pooling/config.json
)
for file in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$file")"
  curl -fL --retry 10 --retry-all-errors --retry-delay 1 --connect-timeout 30 --max-time 900 \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REVISION/$file?download=true" \
    -o "$DEST/$file"
  echo "Downloaded $file"
done
python3 scripts/parallel_curl_download.py \
  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REVISION/model.safetensors?download=true" \
  "$DEST/model.safetensors" 16 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db 90868376
echo "Embedder snapshot saved to $DEST at $REVISION"

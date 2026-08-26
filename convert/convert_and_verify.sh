#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
#
# Convert Qwen3-VL-Embedding to GGUF and prove the vectors are right.
#
# Two flags carry the whole thing. --sentence-transformers-dense-modules keeps
# the dense layers that follow pooling; without it the vectors are wrong and
# nothing reports it. --pooling last matches this model's 1_Pooling
# (pooling_mode_lasttoken); mean pooling also produces a vector, just a
# different one.
set -euo pipefail
MODEL="${1:?path to the HF model directory}"
OUT="${2:?output directory}"
LLAMA="${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a built llama.cpp checkout}"
mkdir -p "$OUT"

python "$LLAMA/convert_hf_to_gguf.py" "$MODEL" \
  --outfile "$OUT/$(basename "$MODEL")-f16.gguf" --outtype f16 \
  --sentence-transformers-dense-modules
python "$LLAMA/convert_hf_to_gguf.py" "$MODEL" --mmproj \
  --outfile "$OUT/mmproj-$(basename "$MODEL")-f16.gguf" --outtype f16

# The reference comes first, and from torch, on the same text. A conversion
# that was never compared to anything is not a conversion, it is a file.
python convert/torch_ref.py "$MODEL" convert/probes.txt "$OUT/torch_ref.json"
"$LLAMA/build/bin/llama-embedding" -m "$OUT/$(basename "$MODEL")-f16.gguf" \
  --pooling last -f convert/probes.txt --embd-output-format json \
  -ngl 99 -c 2048 --batch-size 2048 > "$OUT/ggml_emb.json"
python convert/compare.py "$OUT/torch_ref.json" "$OUT/ggml_emb.json"

# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Torch reference embeddings for the same probe text llama.cpp will see.

Parity has to be measured on identical input. The earlier benchmark used
random token ids, which is fine for latency and useless for correctness.
"""
import json, sys, time, statistics as st, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

MODEL = sys.argv[1]; PROBES = sys.argv[2]; OUT = sys.argv[3]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
texts = [l.rstrip("\n") for l in open(PROBES) if l.strip()]

tok = AutoTokenizer.from_pretrained(MODEL)
full = AutoModel.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).eval().to(dev)
lm = full.language_model

def embed(t):
    ids = tok(t, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        h = lm(input_ids=ids, use_cache=False).last_hidden_state
    return torch.nn.functional.normalize(h[:, -1, :].float(), dim=-1)[0].cpu().numpy()

for t in texts[:3]: embed(t)                      # warm
vecs, ts = [], []
for t in texts:
    torch.mps.synchronize() if dev == "mps" else None
    t0 = time.perf_counter(); v = embed(t)
    torch.mps.synchronize() if dev == "mps" else None
    ts.append(time.perf_counter() - t0); vecs.append(v.tolist())

json.dump({"device": dev, "dim": len(vecs[0]), "texts": texts, "vectors": vecs}, open(OUT, "w"))
ts = sorted(ts)
print(f"  torch {dev} f16: p50 {st.median(ts)*1000:6.1f} ms   dim={len(vecs[0])}   n={len(texts)}")

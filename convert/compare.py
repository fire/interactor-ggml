# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Cosine of the GGUF embeddings against the torch reference, per probe.

Fails the build below 0.99. A conversion that silently changes the vectors
looks exactly like one that did not, right up until every stored embedding is
wrong together.
"""
import json, sys, numpy as np

ref = json.load(open(sys.argv[1]))
got = json.load(open(sys.argv[2]))
gv = [np.array(d["embedding"][0] if isinstance(d["embedding"][0], list) else d["embedding"],
               dtype=np.float64) for d in got["data"]]
rv = [np.array(v, dtype=np.float64) for v in ref["vectors"]]
assert len(gv) == len(rv), f"count mismatch: torch {len(rv)} vs ggml {len(gv)}"

cs = np.array([float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(rv, gv)])
for t, c in zip(ref["texts"], cs):
    print(f"  {t[:44]:46s} {c:9.6f}")
print(f"\n  mean={cs.mean():.6f}  min={cs.min():.6f}  dim={len(rv[0])}")
if cs.min() <= 0.99:
    print("  FAIL: the GGUF does not reproduce the reference"); sys.exit(1)
print("  PASS")

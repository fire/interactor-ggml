# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The comparison that decides it: torch on MPS, which is what pythonx would run."""
import time, statistics as st, torch, torch.nn as nn
from transformers import AutoModel
SEQ, WARM, ITER = 128, 3, 12

class TextEmbedder(nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm = lm
    def forward(self, ids):
        b, s = ids.shape
        pos = torch.arange(s, dtype=torch.long, device=ids.device).unsqueeze(0).expand(b, s)
        h = self.lm(input_ids=ids, position_ids=pos.unsqueeze(0).expand(3, b, s),
                    use_cache=False).last_hidden_state
        return torch.nn.functional.normalize(h[:, -1, :], dim=-1)

g = torch.Generator().manual_seed(11)
probes = [torch.randint(0, 150000, (1, SEQ), generator=g, dtype=torch.int32) for _ in range(WARM+ITER)]

for dev, dt in (("mps", torch.float16), ("mps", torch.float32)):
    full = AutoModel.from_pretrained("qwen3vl2b", dtype=dt, trust_remote_code=True).eval().to(dev)
    m = TextEmbedder(full.language_model).eval()
    with torch.no_grad():
        for p in probes[:WARM]: m(p.to(dev)); torch.mps.synchronize()
        ts = []
        for p in probes[WARM:]:
            pp = p.to(dev); torch.mps.synchronize()
            t0 = time.perf_counter(); m(pp); torch.mps.synchronize(); ts.append(time.perf_counter()-t0)
    ts = sorted(ts)
    print(f"torch {dt.__str__().split('.')[-1]:8s} {dev.upper():4s}  p50 {st.median(ts)*1000:7.1f} ms  p95 {ts[int(len(ts)*.95)-1]*1000:7.1f} ms  {1000/(st.median(ts)*1000):5.1f} emb/s")
    del full, m; torch.mps.empty_cache()

# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Latency per embedding, with a baseline beside it.

A LiteRT number alone says nothing. The question is whether converting bought
anything over the torch path we would otherwise run under pythonx, so torch
float32 on CPU is measured on the same probes, on the same machine, in the same
process.
"""
import time, statistics as st, numpy as np, torch, torch.nn as nn
from transformers import AutoModel
from ai_edge_litert.interpreter import Interpreter

SEQ, WARM, ITER = 128, 3, 12
torch.set_num_threads(8)

class TextEmbedder(nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm = lm
    def forward(self, input_ids):
        b, s = input_ids.shape
        pos = torch.arange(s, dtype=torch.long).unsqueeze(0).expand(b, s)
        pos3 = pos.unsqueeze(0).expand(3, b, s)
        h = self.lm(input_ids=input_ids, position_ids=pos3, use_cache=False).last_hidden_state
        return torch.nn.functional.normalize(h[:, -1, :], dim=-1)

g = torch.Generator().manual_seed(11)
probes = [torch.randint(0, 150000, (1, SEQ), generator=g, dtype=torch.int32) for _ in range(WARM+ITER)]

def stats(ts):
    ts = sorted(ts)
    return st.median(ts)*1000, ts[int(len(ts)*0.95)-1]*1000

print(f"{'runtime':28s} {'p50 ms':>9s} {'p95 ms':>9s} {'emb/s':>8s}")

full = AutoModel.from_pretrained("qwen3vl2b", dtype=torch.float32, trust_remote_code=True).eval()
m = TextEmbedder(full.language_model).eval()
with torch.no_grad():
    for p in probes[:WARM]: m(p)
    ts = []
    for p in probes[WARM:]:
        t0 = time.perf_counter(); m(p); ts.append(time.perf_counter()-t0)
p50, p95 = stats(ts); base = p50
print(f"{'torch f32 CPU (baseline)':28s} {p50:9.1f} {p95:9.1f} {1000/p50:8.1f}")
del full, m

for tag, path in (("litert f32", "qwen3vl2b_text_f32.tflite"),
                  ("litert fp16", "qwen3vl2b_text_fp16.tflite"),
                  ("litert int8-dyn", "qwen3vl2b_text_int8.tflite")):
    it = Interpreter(model_path=path, num_threads=8); it.allocate_tensors()
    i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
    for p in probes[:WARM]:
        it.set_tensor(i0['index'], p.numpy()); it.invoke()
    ts = []
    for p in probes[WARM:]:
        it.set_tensor(i0['index'], p.numpy())
        t0 = time.perf_counter(); it.invoke(); ts.append(time.perf_counter()-t0)
        it.get_tensor(o0['index'])
    p50, p95 = stats(ts)
    print(f"{tag:28s} {p50:9.1f} {p95:9.1f} {1000/p50:8.1f}   {base/p50:4.2f}x vs torch")
    del it

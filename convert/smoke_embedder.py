# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Prove the toolchain on an embedder-SHAPED graph before spending gigabytes.

Same shape as Qwen3-VL-Embedding's text path: int32 token ids in, a stack of
pre-norm attention+MLP blocks, last-token pooling, L2 normalize, fixed 128-token
window. Tiny weights, real ops. If this cannot convert, neither can the 8B, and
we learn it in seconds instead of after a 16 GB download.
"""
import numpy as np, torch, torch.nn as nn, litert_torch

D, L, H, V, SEQ = 128, 2, 4, 512, 128

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.up, self.gate, self.down = nn.Linear(D, 4*D, bias=False), nn.Linear(D, 4*D, bias=False), nn.Linear(4*D, D, bias=False)
    def forward(self, x):
        h = self.n1(x); b, s, _ = h.shape
        q, k, v = (t.view(b, s, H, D//H).transpose(1, 2) for t in (self.q(h), self.k(h), self.v(h)))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, D))
        h = self.n2(x)
        return x + self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))

class Embedder(nn.Module):
    """Token ids -> normalized last-token embedding. No KV cache, no decode loop."""
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.norm = nn.RMSNorm(D)
    def forward(self, input_ids):
        x = self.tok(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return torch.nn.functional.normalize(self.norm(x)[:, -1, :], dim=-1)

torch.manual_seed(0)
m = Embedder().eval()
ids = torch.randint(0, V, (1, SEQ), dtype=torch.int32)
with torch.no_grad():
    ref = m(ids).numpy()
print(f"torch reference: {ref.shape} {ref.dtype}  |‖v‖={np.linalg.norm(ref):.6f}")

print("converting ...")
edge = litert_torch.convert(m.eval(), (ids,))
edge.export("smoke_embedder.tflite")
import os; print(f"wrote smoke_embedder.tflite  {os.path.getsize('smoke_embedder.tflite')/1024:.1f} KB")

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="smoke_embedder.tflite"); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
print(f"tflite input : {inp['name']} {inp['dtype'].__name__} {inp['shape']}")
print(f"tflite output: {out['name']} {out['dtype'].__name__} {out['shape']}")
it.set_tensor(inp['index'], ids.numpy()); it.invoke()
got = it.get_tensor(out['index'])

cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got)))
mx  = float(np.abs(ref - got).max())
print(f"\ncosine(torch, litert) = {cos:.8f}")
print(f"max |abs diff|        = {mx:.3e}")
print("VERDICT:", "PASS" if cos > 0.9999 and mx < 1e-3 else "FAIL")

# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Quantize the Qwen3-VL-Embedding-2B text tower, and measure what it costs.

float32 converted exactly at 6.41 GB, which is correct and unusable. The
question is what fp16 and int8 cost in accuracy, and the only honest way to
answer it is a cosine against the float32 torch reference on the same inputs --
a size number without an accuracy number beside it is not a measurement.
"""
import os, numpy as np, torch, torch.nn as nn, litert_torch
from transformers import AutoModel
from litert_torch.generative.quantize import quant_recipes
from ai_edge_litert.interpreter import Interpreter

SEQ, N = 128, 8

class TextEmbedder(nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm = lm
    def forward(self, input_ids):
        b, s = input_ids.shape
        pos = torch.arange(s, dtype=torch.long).unsqueeze(0).expand(b, s)
        pos3 = pos.unsqueeze(0).expand(3, b, s)
        h = self.lm(input_ids=input_ids, position_ids=pos3, use_cache=False).last_hidden_state
        return torch.nn.functional.normalize(h[:, -1, :], dim=-1)

full = AutoModel.from_pretrained("qwen3vl2b", dtype=torch.float32, trust_remote_code=True).eval()
m = TextEmbedder(full.language_model).eval()

g = torch.Generator().manual_seed(7)
probes = [torch.randint(0, 150000, (1, SEQ), generator=g, dtype=torch.int32) for _ in range(N)]
with torch.no_grad():
    refs = np.concatenate([m(p).numpy() for p in probes], 0)
print(f"torch f32 reference: {refs.shape}")

def bench(tag, path, quant):
    if not os.path.exists(path):
        print(f"\n--- {tag}: converting ---")
        litert_torch.convert(m, (probes[0],), quant_config=quant).export(path)
    sz = os.path.getsize(path)
    it = Interpreter(model_path=path); it.allocate_tensors()
    i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
    outs = []
    for p in probes:
        it.set_tensor(i0['index'], p.numpy()); it.invoke()
        outs.append(it.get_tensor(o0['index']).copy())
    got = np.concatenate(outs, 0)
    cos = np.array([float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))) for a, b in zip(refs, got)])
    print(f"{tag:10s} {sz/1073741824:6.2f} GB   cos mean={cos.mean():.6f} min={cos.min():.6f}   maxdiff={np.abs(refs-got).max():.2e}")
    return sz, cos.mean()

print(f"\n{'variant':10s} {'size':>9s}   accuracy vs torch f32")
bench("f32", "qwen3vl2b_text_f32.tflite", None)
bench("fp16", "qwen3vl2b_text_fp16.tflite", quant_recipes.full_fp16_recipe())
bench("int8-dyn", "qwen3vl2b_text_int8.tflite", quant_recipes.full_dynamic_recipe())

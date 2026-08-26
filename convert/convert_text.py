# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Convert the TEXT path of Qwen3-VL-Embedding-2B to LiteRT.

Text first, vision second. The text tower is what embeds bios, world names and
notes today, and it shares the output space with the multimodal path, so a
text-only artifact is useful on its own rather than a stepping stone that gets
thrown away.

Fixed 128-token window, last-token pooling, L2 normalize. No KV cache and no
decode loop: an embedder is one forward pass, which is the whole reason this
converts when a generative model does not.
"""
import numpy as np, torch, torch.nn as nn, litert_torch
from transformers import AutoModel

SEQ = 128

class TextEmbedder(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
    def forward(self, input_ids):
        b, s = input_ids.shape
        pos = torch.arange(s, dtype=torch.long).unsqueeze(0).expand(b, s)
        # Qwen3-VL uses interleaved mRoPE: position_ids is [3, batch, seq] for
        # the temporal/height/width axes. Text-only means all three agree.
        pos3 = pos.unsqueeze(0).expand(3, b, s)
        h = self.lm(input_ids=input_ids, position_ids=pos3, use_cache=False).last_hidden_state
        return torch.nn.functional.normalize(h[:, -1, :], dim=-1)

print("loading ...")
full = AutoModel.from_pretrained("qwen3vl2b", dtype=torch.float32, trust_remote_code=True).eval()
m = TextEmbedder(full.language_model).eval()
ids = torch.randint(0, 150000, (1, SEQ), dtype=torch.int32)

with torch.no_grad():
    ref = m(ids).numpy()
print(f"torch reference {ref.shape} |v|={np.linalg.norm(ref):.6f}")

print("converting (this is the part that either works or does not) ...")
edge = litert_torch.convert(m, (ids,))
edge.export("qwen3vl2b_text_f32.tflite")
import os
sz = os.path.getsize("qwen3vl2b_text_f32.tflite")
print(f"wrote qwen3vl2b_text_f32.tflite  {sz/1073741824:.2f} GB  ({sz:,} bytes)")
print("FLATBUFFER 2GB CAP:", "EXCEEDED" if sz > 2**31 else "under")

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="qwen3vl2b_text_f32.tflite"); it.allocate_tensors()
i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
print("tflite in :", i0['dtype'].__name__, i0['shape'])
print("tflite out:", o0['dtype'].__name__, o0['shape'])
it.set_tensor(i0['index'], ids.numpy()); it.invoke()
got = it.get_tensor(o0['index'])
cos = float(np.dot(ref.ravel(), got.ravel())/(np.linalg.norm(ref)*np.linalg.norm(got)))
print(f"\ncosine(torch, litert) = {cos:.8f}")
print(f"max |abs diff|        = {float(np.abs(ref-got).max()):.3e}")
print("VERDICT:", "PASS" if cos > 0.999 else "FAIL")

<!-- SPDX-License-Identifier: Apache-2.0 OR MIT -->

# 2026-08-26 — Qwen3-VL-Embedding-2B text tower converts to LiteRT

No LiteRT build of any Qwen3-VL-Embedding existed. This one does now.

## What was measured

Apple M2 Pro, 32 GB, macOS. `litert-torch` 0.9.4, `ai-edge-litert` 2.2.0,
torch 2.13.0, transformers 5.16.1, Python 3.12. Input `int32[1,128]`, output
`float32[1,2048]`, last-token pooling and L2 normalize. Eight fixed random
token probes, seed 7. Accuracy is cosine against the **torch float32**
reference on the same probes.

| variant | size | cos mean | cos min | max abs diff |
| --- | --- | --- | --- | --- |
| float32 | 6.41 GB | 1.000000 | 1.000000 | 6.26e-07 |
| **float16** | **3.21 GB** | **1.000000** | **1.000000** | 7.00e-07 |
| int8 dynamic, channelwise | 1.61 GB | 0.997661 | 0.996853 | 1.90e-02 |

Conversion wall time 64 s at float32. Graph is 362.854 G ops / 181.427 G MACs.

## What it means

**float16 is free.** Half the size for no measurable accuracy change, so it is
the default. int8 buys another 2x for 0.0023 mean cosine; that trade is not
worth taking at 2B, and becomes interesting at 8B where float16 would be about
12 GB.

**The 2 GB FlatBuffer cap is not a wall.** The converter prints "Module size is
greater than 2GB" and then writes a file the interpreter loads and runs
correctly. Both float32 and float16 exceed 2 GB and both work.

**The toolchain was never the risk.** A negative control ran first
(`convert/smoke_embedder.py`): a tiny embedder-shaped graph — RMSNorm, causal
SDPA, SwiGLU, last-token pooling, L2 normalize — converted at cosine 1.00000012
and max deviation 4.47e-08. Had the real conversion failed after that, the
failure would have been the model's and not the tools'.

`litert-torch` needs no TensorFlow, which was the expected macOS blocker. Note
`ai-edge-torch` is deprecated and renamed to `litert-torch`.

## Not done yet

- **The vision tower.** Only `language_model` is converted. llama.cpp lists
  "Qwen3-VL-Embedding (vision encoder + pooling)" as unavailable, so the fused
  multimodal path is the part with no precedent anywhere. Text shares the output
  space, so this artifact is useful on its own rather than a stepping stone.
- **8B.** Same architecture, 36 layers against 28, hidden 4096 against 2048.
- **Matryoshka.** The card claims 64–4096 for the 8B; untested here.
- **Latency.** Not measured. A size without a speed is half a number.

## Reproducing

```sh
uv venv --python 3.12 .venv && uv pip install -r convert/pyproject.toml
python convert/smoke_embedder.py    # the control, first
python convert/convert_text.py      # float32 + parity
python convert/quantize_text.py     # float16 and int8 + parity
```

---

# Addendum, same day — latency, and it reverses the recommendation

The conversion is correct. On this hardware it is also the wrong runtime, and
the numbers are not close.

Same machine, same probes (seed 11), 3 warmup and 12 timed iterations, 8
threads, batch of one, 128 tokens.

| runtime | p50 ms | p95 ms | emb/s | vs best |
| --- | --- | --- | --- | --- |
| **torch float16 MPS** | **102.6** | 103.4 | **9.7** | **1.00x** |
| torch float32 MPS | 104.3 | 105.1 | 9.6 | 0.98x |
| torch float32 CPU | 376.8 | 409.8 | 2.7 | 0.27x |
| LiteRT int8 dynamic (CPU) | 415.7 | 420.6 | 2.4 | 0.25x |
| LiteRT float16 (CPU) | 1004.6 | 1013.9 | 1.0 | 0.10x |
| LiteRT float32 (CPU) | 1060.3 | 2312.1 | 0.9 | 0.10x |

**torch on MPS is 9.8x faster than the LiteRT float16 build**, and LiteRT loses
to torch even on CPU, where int8 — the only competitive LiteRT variant — is
still 10% behind and pays 0.0023 mean cosine for the privilege.

## Why, and what is not yet ruled out

`ai_edge_litert.gpu` raises `ImportError` on the macOS arm64 wheel. The
`Delegate` and `load_delegate` symbols exist, but no Metal delegate ships in
the wheel, so **LiteRT here is CPU-and-XNNPACK only in practice**, while torch
reaches the GPU through MPS. That is the whole gap.

This is a caveat on the result, not a rescue of it. `litert-community`'s own
Qwen3-Embedding-0.6B artifact is named `qwen3emb_gpu_fp16.tflite` and was
benchmarked on a Pixel's Tensor G3 GPU, so the fast path plainly exists on
platforms whose delegate ships. It does not ship for macOS in the current
wheel. Until it does, the measured comparison stands.

## What this changes

**The default embedding runtime becomes torch on MPS under `pythonx`, not
LiteRT.** That satisfies the same rule LM Studio failed: it is declared in
`convert/pyproject.toml`, pinned to exact versions, and reproducible.

**The LiteRT artifact is kept and published anyway**, for three honest reasons
and no dishonest ones: nobody had converted any Qwen3-VL-Embedding, so the
recipe is worth existing; it is a single hashable file with no Python runtime
at all; and it is the fallback if the MPS path breaks or the Metal delegate
lands. It is not kept because it is fast, because it is not.

**Nothing architectural has to change**, which was the point of putting a port
in front of the model. The adapter set is unchanged; only which one is default
moves.

## The general lesson, recorded

LiteRT entered this project on a speed argument for TimesFM, was moved to a
privacy argument for embeddings, and the privacy argument is the one that
survived contact with a measurement. Privacy is satisfied by any local runtime,
including torch. So LiteRT's remaining seat here is portability and packaging —
narrower than either argument that put it on the list.

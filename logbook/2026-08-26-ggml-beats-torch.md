<!-- SPDX-License-Identifier: Apache-2.0 OR MIT -->

# 2026-08-26 — ggml clears the bar, and beats torch

The bar was "on par with torch or better", set at the measured torch latency
rather than a guess. ggml cleared both halves of it.

## Measured

Apple M2 Pro, 32 GB, macOS. llama.cpp `925e117`, Metal, f16.
Qwen3-VL-Embedding-2B, ten short real-world strings, identical text through
both paths. Latency through `llama-server` on loopback, model already loaded,
4 warmup and 30 timed requests.

| | cosine vs torch f32 | p50 | p95 | emb/s |
| --- | --- | --- | --- | --- |
| **ggml, llama.cpp Metal f16** | **0.999997** mean, 0.999995 min | **29.9 ms** | 36.3 ms | **33.5** |
| torch float16 MPS | 1.0 by definition | 51.2 ms | — | 19.5 |

**1.71x faster than torch on the same machine and the same inputs.**

Note this probe set is short real text, so 51.2 ms is not the 102.6 ms from
RFD 0007 — that was 128 padded tokens. Both numbers are torch on MPS on this
machine; they differ because the input differs. Compare like with like.

## Why this beat LiteRT so decisively

LiteRT delegated 77 of 2377 ops to the GPU and lost to torch by 9.8x (RFD
0008). ggml has no delegate: the Metal backend is the runtime, so there is no
coverage fraction and no CPU round-trip per unsupported op.

## What was almost done instead, and did not need doing

llama.cpp PR #18665 was to be forked and completed. Reading the thread to its
end instead: the author closed it himself because the model was private at the
time, its `1_Pooling` blocker has since been fixed upstream, and a commenter
demonstrated in July that these models **already run on master**. The PR only
adds an OpenAI-style JSON schema to the server's `/v1/embeddings` — a
convenience on a surface this project does not use.

Cost of not reading to the end: nearly a fork of llama.cpp.

## Two flags that carry everything

- `--sentence-transformers-dense-modules` at convert time keeps the dense
  layers after pooling. Without it the vectors are wrong and nothing says so.
- `--pooling last` at run time. This model pools the last token; mean pooling
  returns a vector too, just a different one.

Both are silent when wrong, which is why `convert/compare.py` runs against a
torch reference and fails below cosine 0.99.

## Published

`ifire/Qwen3-VL-Embedding-2B-GGUF` — text tower plus the `mmproj` vision
tower, so multimodal input works. Placed on `3-interactor` by the manifest,
per RFD 1141: a model comes from the interactor side.

## Not done

- The 8B. Same architecture; GGUFs with mmproj already exist publicly.
- Image and mixed image+text embeddings are converted but **not yet verified**
  against a torch reference. Text is. Do not claim the vision path works until
  it has its own cosine number.
- `embed/2` on the Elixir NIF. The server proves the path; the NIF is the
  in-process form.

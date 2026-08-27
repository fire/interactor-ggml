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

---

# Addendum — the vision path is broken, and "it works" was not evidence

The text result above stands. The multimodal claim beside it did not survive
its first measurement.

## Measured

Ten licence-clean COCO train2017 images (CC-BY and no-known-restrictions,
never the val2017 holdout), through torch and through `llama-server`.

Cosine against the torch reference: **0.11**. Not a numerical difference — the
vectors are unrelated. The diagnostic says why:

```
image A  vs  image B         : 1.00000000
image A  vs  NO IMAGE at all : 1.00000000
image A  vs  different text  : 0.68915199
```

Every image returns the same vector, identical to passing no image at all.
`image_data` is accepted without error on `/v1/embeddings` and **silently
dropped**; the endpoint embeds the prompt text. The OpenAI-style
`input: [{type:"image_url"}]` form returns HTTP 500 rather than working.

The pairwise-similarity check is what turned "wrong" into "why": torch's ten
images sit at mean similarity 0.0996 spread over [-0.061, 0.326], while ggml's
sit at exactly 1.0000. A structure that collapses to a point is not a
transform to be recovered, it is an input being ignored.

## The weights are not at fault

The mmproj carries all 18 deepstack tensors (`v.deepstack.{5,11,17}.*`,
matching `deepstack_visual_indexes: [5, 11, 17]`), and `clip.cpp` implements
deepstack. The server's embeddings path simply never routes media to them.

## Retraction

The entry above says PR #18665 "only adds an OpenAI-style JSON schema to the
server's `/v1/embeddings` — a convenience on a surface this project does not
use", and concludes that reading four more comments saved a fork.

**That was wrong.** #18665 is the wiring, not a convenience. The conclusion was
drawn from two forum comments — "you can run these models already on current
master" and "it works unbelievably well" — neither of which carried a number.
The first measurement of the thing they described disagreed with both.

The rule that would have caught it was already written down in this repository,
one section above, about two different flags: **it fails silently.** A
capability nobody measured is not a capability, and someone else's impression
is not a measurement.

## Consequence

- The published card at `ifire/Qwen3-VL-Embedding-2B-GGUF` claimed multimodal
  input works. Corrected on the hub, with the numbers, the same day.
- Fork #18665 onto current master and finish it. That is the original
  instruction, and it was right.
- Text embeddings are unaffected: cosine 0.999997, 29.9 ms, 1.71x torch.

---

# Second addendum — the vision path works; the previous addendum was my bug

The addendum above says images are silently dropped and that PR #18665 is "the
wiring". **Both wrong, and the error was mine.**

`image_data` is a `/completion` field. The embeddings endpoint takes
`multimodal_data`:

```json
{"input": {"prompt_string": "…<__media__>…", "multimodal_data": ["<base64>"]}}
```

`tokenize_input_subprompt` in `server-common.cpp` has handled this the whole
time, and `handle_embeddings_impl` already passes `mctx` through. Reading the
source settled in minutes what two rounds of reading forum comments got wrong
in both directions.

## What is actually true

| input | cosine vs torch |
| --- | --- |
| text | 0.999997 mean, 0.999995 min |
| image | **0.967420** mean, 0.951388 min |

Images produce distinct, meaningful vectors — two different images score 0.146
against each other. The vision tower is genuinely used. But 0.967 is not
0.9999, and for retrieval that gap is real.

**`corr(pixels, cosine) = +0.730`** across the sample: larger images agree
better. That points at **image preprocessing** — resize and interpolation
between `Qwen3VLImageProcessor` and llama.cpp's clip path — not at the weights
or the fusion. Worth a narrower experiment: feed pre-resized images at the
model's native grid and see whether the gap closes.

## Where that leaves PR #18665

Back to the original reading: it adds the OpenAI-style
`input: [{"type":"image_url"}]` schema, which currently returns HTTP 500. A
convenience, not the wiring. Forked to `fire/interactor-llama-cpp` anyway; the
branch is 2,983 commits behind master and the draft carries two real defects —
a dangling `if (!ctx_server.mctx)` with no body that swallows the following
statement, and an `auto inputs` in the else branch that shadows the outer
vector. Neither was ever finished.

## The actual lesson, which is not the one I drew last time

Last time I concluded "someone else's impression is not a measurement" and then
made a *worse* error: I measured the wrong thing and trusted my own bad
measurement over two people's correct reports. A measurement of the wrong call
is not evidence either.

What separated signal from noise both times was **reading the source**. The
forum told me it worked; my first test said it did not; the source said which
of them was right and why. Sixty lines of `server-common.cpp` outranked
everything else in this entry.

## Standing footgun, worth reporting upstream

`image_data` on `/v1/embeddings` is accepted without error and ignored. Every
image then returns the identical text-only vector. Invisible without a
reference to compare against — the same silent-failure class as omitting
`--pooling last`.

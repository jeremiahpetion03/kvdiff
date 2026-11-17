# kvdiff

A differential tester for KV-cache correctness in [Hugging Face Transformers](https://github.com/huggingface/transformers). It checks that batching, padding, caching, and `generate()` don't change what a model computes, and it separates real failures from ordinary floating-point rounding before calling anything a bug.

## What it checks

For every model × attention implementation × precision, each path is compared against a reference that does **not** share the code under test:

| Check | Path under test | Reference |
|---|---|---|
| **padding** | padded batch, full forward | each row alone, unpadded |
| **cache** | padded batch, token-by-token decoding through the KV cache | padded batch, full forward |
| **generate** | `model.generate(use_cache=True)`, batched with left padding | each row generated alone, no cache |

Prompts are deliberately uneven (one is 3 tokens long next to 37-token rows), so the batch needs heavy left padding. Position ids come from the attention mask, so padding never shifts positions. fp32 runs with TF32 disabled, because Ampere GPUs otherwise run fp32 matmuls at about 10 mantissa bits, which would hide real discrepancies.

## Results

Run on an NVIDIA RTX 3060, PyTorch 2.15 (CUDA 13.4), Transformers 5.23. Models: GPT-2, SmolLM2-135M (Llama architecture), Qwen2.5-0.5B (RoPE, grouped-query attention). Attention: eager and sdpa. Precision: fp32, bf16, fp16.

**54 checks: 46 passed, 8 flagged.** All 18 fp32 checks passed, with relative errors around 2×10⁻⁶. Triage of the 8 flags:

| Flags | Configuration | Verdict | Evidence |
|---|---|---|---|
| 6 | GPT-2, bf16 (eager and sdpa) | **False positive** (my threshold) | Every flagged token had a reference margin of exactly 1.000. GPT-2's logits sit near ±100 to ±200, where adjacent bf16 values are 1.0 apart, so the "preferred" token was a tie. Against an fp32 ground truth, cached and full-forward errors match (ratio 0.60–1.25). |
| 1 | Qwen2.5, eager, fp16 (`cache`) | **Real, deterministic NaN** | See below |
| 1 | Qwen2.5, eager, fp16 (`generate`) | Same root cause | First-step divergence on the same heavily padded row |

### The Qwen2.5 eager fp16 NaN

Re-running the exact inputs 30 times ([`repro2.py`](repro2.py)):

- **NaN logits in 30/30 runs**, bit-identical every time, always on the 3-token prompt with 34 padding tokens.
- A forward hook on every module locates the **first non-finite value in `model.layers.1.self_attn.o_proj`, during prefill.**
- Controls: the same row **without padding** is clean. **sdpa** is clean. **bf16** is clean. **fp32** is clean.

**Likely mechanism.** This is consistent with the data, but the propagation step is not yet directly instrumented:

1. A left-padding position has no key it may attend to. Eager attention gives every key the same masked score, so the softmax averages over all positions and produces a meaningless value.
2. For these inputs that value overflows fp16 (max 65,504) in layer 1's output projection.
3. The infinity enters the residual stream and the KV cache at the padding position.
4. Real tokens give that position an attention weight of exactly 0, but 0 × ∞ = NaN, so masked padding contaminates real outputs.

sdpa handles fully masked rows differently (max padding-position activation 57.9 vs 1,281 for eager on other inputs), which is consistent with it staying clean.

**Prior report.** The symptom (Qwen2, eager attention, fp16 → NaN) was reported in Transformers PR [#33317](https://github.com/huggingface/transformers/pull/33317) (September 2024), which was closed. This project rediscovered it independently through differential testing and narrows it to a specific layer, input condition, and decoding path. It still reproduces in Transformers 5.23.

### A hypothesis this project ruled out

The first reproduction ([`repro.py`](repro.py)) tested overflow with different prompts. All 100 trials were clean, and padding-position activations peaked at 1,281, about 50x below fp16's limit. Overflow depends on the specific input, so a generic "padding overflows fp16" claim would have been wrong. The deterministic reproduction above uses the exact inputs that failed.

### Open question

The first `kvdiff.py` run reported token flips for the Qwen2.5 eager fp16 cache check rather than NaN, although `repro2.py` produces NaN on every run with the same inputs. That discrepancy is unexplained.

## Methodology notes

- **Every threshold is per precision.** Relative-error tolerance is 1e-4 for fp32, 1e-2 for fp16, and 5e-2 for bf16.
- **Token flips count only with a clear margin.** A flip counts only when the reference preferred its token by more than a per-precision margin. The GPT-2 false positives showed that the margin also has to account for the precision's step size at the logits' magnitude.
- **Triage against fp32 ground truth.** [`triage.py`](triage.py) measures every low-precision path against the same fp32 reference. If the cached path's error is comparable to the full forward's, the disagreement is rounding; if it is several times larger, the cache path is adding error. NaN is reported as `NON-FINITE` and never as rounding (an earlier version mislabeled it, because comparisons with NaN are always false).

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu134
pip install transformers accelerate

python kvdiff.py      # 54 checks; writes results.json
python triage.py      # per-row error vs fp32 ground truth; writes triage.json
python repro.py       # overflow hypothesis, 5 controlled configurations
python repro2.py      # exact failing inputs x 30, first non-finite module
```

All scripts also run on CPU (`--device cpu` for kvdiff and triage), much more slowly. Python 3.13 is recommended: some dependencies don't yet publish Windows wheels for 3.15.

## Files

```
kvdiff.py     differential tester: padding, cache, and generate checks
triage.py     error vs fp32 ground truth, per row, with NaN handling
repro.py      controlled test of the overflow hypothesis
repro2.py     deterministic reproduction and first-bad-module localization
results.json  kvdiff output from the run above
triage.json   triage output from the run above
```

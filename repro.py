"""Minimal reproduction for the kvdiff candidate: non-finite outputs from cached
decoding with a heavily left-padded row, eager attention, fp16.

Hypothesis: hidden states at LEFT-PADDING positions overflow fp16 (max 65504)
to inf; those infs are written into the KV cache; masking adds a large
negative number to the attention score, but inf/NaN keys or values cannot be
masked away by addition (inf - inf = NaN, 0 * inf = NaN), so the "ignored"
padding poisons real tokens.

For each configuration, runs prefill + a few decode steps N times and reports:
  * how often the decoded logits are non-finite,
  * whether any cached key/value at PAD vs REAL positions is non-finite,
  * the largest |hidden state| at pad vs real positions during prefill.

    python repro.py
    python repro.py --trials 50
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import configure_determinism

MODEL = "Qwen/Qwen2.5-0.5B"
LONG = ("The harbor was quiet before dawn, and the fishing boats rocked against their moorings "
        "while the gulls waited on the pilings and the tide came in slowly over the stones.")
SHORT = "The harbor"


def cache_tensors(cache):
    """Yield (keys, values) per layer across Transformers cache API versions."""
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, "key_cache"):
        yield from zip(cache.key_cache, cache.value_cache, strict=True)
    else:  # legacy tuple-of-tuples
        for k, v in cache:
            yield k, v


@torch.no_grad()
def trial(model, ids, mask, steps: int) -> dict:
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                output_hidden_states=True)
    pad = mask == 0  # (B, T)
    h_pad = max((h[pad].abs().max().item() if pad.any() else 0.0) for h in out.hidden_states)
    h_real = max(h[~pad].abs().max().item() for h in out.hidden_states)
    kv_pad_bad = kv_real_bad = False
    for k, v in cache_tensors(out.past_key_values):  # (B, heads, T, d)
        for t in (k, v):
            bad = ~torch.isfinite(t).all(dim=(1, 3))  # (B, T)
            kv_pad_bad |= bool((bad & pad).any())
            kv_real_bad |= bool((bad & ~pad).any())

    prefill_bad = not torch.isfinite(out.logits[:, -1]).all().item()
    past, last, cur_mask = out.past_key_values, pos[:, -1:], mask
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    decode_bad_rows = set()
    for _ in range(steps):
        cur_mask = torch.cat([cur_mask, torch.ones_like(nxt)], dim=1)
        last = last + 1
        o = model(input_ids=nxt, attention_mask=cur_mask, position_ids=last, past_key_values=past,
                  use_cache=True)
        past = o.past_key_values
        bad_rows = (~torch.isfinite(o.logits[:, -1]).all(-1)).nonzero().flatten().tolist()
        decode_bad_rows.update(bad_rows)
        nxt = torch.nan_to_num(o.logits[:, -1]).argmax(-1, keepdim=True)
    return {"h_pad": h_pad, "h_real": h_real, "kv_pad_bad": kv_pad_bad, "kv_real_bad": kv_real_bad,
            "prefill_bad": prefill_bad, "decode_bad_rows": sorted(decode_bad_rows)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--steps", type=int, default=8)
    args = p.parse_args(argv)
    configure_determinism()
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    padded = tok([LONG, SHORT], return_tensors="pt", padding=True).to(device)
    alone = tok([SHORT], return_tensors="pt").to(device)
    n_pad = int((padded.attention_mask[1] == 0).sum())
    print(f"batch of 2: long prompt {padded.input_ids.shape[1]} tokens, short prompt padded with {n_pad} pad tokens")

    configs = [("eager", "fp16", True), ("eager", "fp16", False), ("sdpa", "fp16", True),
               ("eager", "bf16", True), ("eager", "fp32", True)]
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    print(f"\n{'attn':6s} {'dtype':5s} {'padded':6s}  bad-decode-trials  inf/NaN in KV (pad / real)  "
          f"max|hidden| (pad / real)")
    for attn, d, use_pad in configs:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtypes[d],
                                                     attn_implementation=attn).to(device).eval()
        batch = padded if use_pad else alone
        results = [trial(model, batch.input_ids, batch.attention_mask, args.steps) for _ in range(args.trials)]
        bad = sum(bool(r["decode_bad_rows"]) or r["prefill_bad"] for r in results)
        kvp = sum(r["kv_pad_bad"] for r in results)
        kvr = sum(r["kv_real_bad"] for r in results)
        hp = max(r["h_pad"] for r in results)
        hr = max(r["h_real"] for r in results)
        rows = sorted({row for r in results for row in r["decode_bad_rows"]})
        print(f"{attn:6s} {d:5s} {str(use_pad):6s}  {bad:3d}/{args.trials:<3d} rows {rows!s:8s}"
              f"  {kvp:3d} / {kvr:<3d} of {args.trials:<8d}  {hp:12,.1f} / {hr:,.1f}")
        del model
        torch.cuda.empty_cache()
    print("\nfp16 max finite value is 65,504. A pad-position |hidden| at or above that means overflow.")


if __name__ == "__main__":
    main()

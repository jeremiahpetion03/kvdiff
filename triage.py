"""Triage kvdiff flags: is a low-precision discrepancy ordinary rounding, or is
one code path doing extra damage?

Each low-precision path (unpadded alone / padded full forward / padded cached
decode) is measured against the SAME fp32 ground truth. If the cached path's
error is comparable to the full-forward path's error, the disagreement between
them is just rounding. If it is several times larger, the cached path is
adding error of its own, which is a real candidate bug.

Also prints the two numbers that explain rounding-only flags:
  * the largest internal activation in fp32 (fp16 loses absolute precision
    as values grow: its step size at magnitude m is about m / 1024), and
  * the low-precision step size at the model's logit magnitude (a "margin"
    equal to one step is a tie, not a preference).

    python triage.py
    python triage.py --models Qwen/Qwen2.5-0.5B --dtypes fp16 --attn eager sdpa
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import (DTYPES, configure_determinism, logits_padded_cached, logits_padded_full,
                    logits_unpadded, make_rows)

MANTISSA_BITS = {"fp32": 23, "fp16": 10, "bf16": 7}
RATIO_FLAG = 3.0  # cached error this many times the full-forward error => candidate


def step_at(magnitude: float, dtype_name: str) -> float:
    """Spacing between adjacent representable numbers near `magnitude`."""
    if magnitude <= 0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(magnitude)) - MANTISSA_BITS[dtype_name])


def per_row_error(x: torch.Tensor, truth: torch.Tensor) -> list[float]:
    return (x - truth).abs().amax(dim=(1, 2)).tolist()


def load(name: str, dtype, attn: str, device):
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn).to(device).eval()


@torch.no_grad()
def max_hidden(model, rows, device) -> float:
    worst = 0.0
    for prompt, cont in rows:
        out = model(input_ids=torch.tensor([prompt + cont], device=device), output_hidden_states=True)
        worst = max(worst, max(h.abs().max().item() for h in out.hidden_states))
    return worst


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["openai-community/gpt2", "Qwen/Qwen2.5-0.5B"])
    p.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    p.add_argument("--attn", nargs="+", default=["eager", "sdpa"])
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("triage.json"))
    args = p.parse_args(argv)

    configure_determinism()
    device = torch.device(args.device)
    report = []
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        rows = make_rows(tok, args.rows, args.k, args.seed)

        gt_model = load(name, torch.float32, "eager", device)
        truth = torch.stack(logits_unpadded(gt_model, rows, args.k, device))  # (B, k, V)
        hidden = max_hidden(gt_model, rows, device)
        logit_mag = truth.abs().max().item()
        del gt_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        print(f"\n=== {name}")
        print(f"fp32 largest activation: {hidden:,.1f}   largest |logit|: {logit_mag:.1f}")
        for d in args.dtypes:
            print(f"  {d}: step size at largest activation = {step_at(hidden, d):.4g}, "
                  f"at largest logit = {step_at(logit_mag, d):.4g}")

        for attn in args.attn:
            for d in args.dtypes:
                model = load(name, DTYPES[d], attn, device)
                alone = torch.stack(logits_unpadded(model, rows, args.k, device))
                full = logits_padded_full(model, rows, args.k, pad_id, device)
                cached = logits_padded_cached(model, rows, args.k, pad_id, device)
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None

                e_alone, e_full, e_cached = (per_row_error(x, truth) for x in (alone, full, cached))
                print(f"\n  {attn:5s} {d}   per-row max |error| vs fp32 ground truth")
                print("  row  prompt  alone     padded-full  padded-cached  cached/full  verdict")
                for i, (prompt, _) in enumerate(rows):
                    ratio = e_cached[i] / max(e_full[i], 1e-9)
                    if not all(math.isfinite(e) for e in (e_alone[i], e_full[i], e_cached[i])):
                        verdict = "NON-FINITE"  # NaN/inf is never rounding (NaN comparisons are always False)
                    elif ratio > RATIO_FLAG and e_cached[i] > 3 * e_alone[i]:
                        verdict = "CANDIDATE"
                    else:
                        verdict = "rounding"
                    print(f"  {i:3d}  {len(prompt):6d}  {e_alone[i]:8.4f}  {e_full[i]:11.4f}  "
                          f"{e_cached[i]:13.4f}  {ratio:11.2f}  {verdict}")
                    report.append({"model": name, "attn": attn, "dtype": d, "row": i,
                                   "prompt_len": len(prompt), "err_alone": e_alone[i],
                                   "err_padded_full": e_full[i], "err_padded_cached": e_cached[i],
                                   "cached_over_full": ratio, "verdict": verdict})

    args.out.write_text(json.dumps(report, indent=2))
    n = sum(r["verdict"] != "rounding" for r in report)
    print(f"\n{n} candidate rows. Wrote {args.out}")


if __name__ == "__main__":
    main()

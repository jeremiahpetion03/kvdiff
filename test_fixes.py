"""Test two fixes for the Qwen2.5 eager fp16 NaN on kvdiff's exact failing inputs.

The bug (see README): in fp16, score + (-65504) rounds to -inf once the score
is below about -16; a padding query row whose keys ALL overflow becomes all
-inf, and softmax over it is NaN.

Variants of the eager attention function, each run on the same inputs:
  original          Transformers' own function (the bug)
  replica           my copy of it with NO fix: must be bit-identical to
                    `original`, or the comparison below is invalid
  fp32_mask         add the mask in fp32 before the softmax
  zero_masked_rows  as original, then set the weights of query rows that
                    have no allowed key to 0 (what sdpa effectively outputs)

For each variant: non-finite rows at prefill and over 15 teacher-forced
decode steps; whether the rows the bug never touched stay bit-identical to
`original`; and, for real (non-padding) logits, max |difference| and argmax
agreement against an fp32 eager reference, next to sdpa fp16 as a baseline.

    python test_fixes.py        # writes test_fixes.json
"""

import json

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import configure_determinism, left_pad, make_rows, positions_from_mask

MODEL = "Qwen/Qwen2.5-0.5B"
ORIGINAL = qwen2.eager_attention_forward


def make_eager(fix):
    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        k = qwen2.repeat_kv(key, module.num_key_value_groups)
        v = qwen2.repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, k.transpose(2, 3)) * scaling
        mask = None if attention_mask is None else attention_mask[:, :, :, : k.shape[-2]]
        if mask is not None:
            scores = scores.float() + mask.float() if fix == "fp32_mask" else scores + mask
        w = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if fix == "zero_masked_rows" and mask is not None:
            no_allowed_key = (mask < 0).all(-1, keepdim=True)            # (B, 1, Tq, 1)
            w = w.masked_fill(no_allowed_key, 0.0)
        w = nn.functional.dropout(w, p=dropout, training=module.training)
        out = torch.matmul(w, v).transpose(1, 2).contiguous()
        return out, w
    return eager


@torch.no_grad()
def run(model, ids, mask, pos, rows, k, device):
    """Prefill + teacher-forced decode, as kvdiff does. Returns (B, k, V) logits."""
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    steps, past, last, cur = [out.logits[:, -1]], out.past_key_values, pos[:, -1:], mask
    for j in range(k - 1):
        tok_j = torch.tensor([[c[j]] for _, c in rows], device=device)
        cur = torch.cat([cur, torch.ones_like(tok_j)], dim=1)
        last = last + 1
        o = model(input_ids=tok_j, attention_mask=cur, position_ids=last, past_key_values=past, use_cache=True)
        past = o.past_key_values
        steps.append(o.logits[:, -1])
    return torch.stack(steps, dim=1)


def bits(t):
    return t.view(torch.int16) if t.dtype in (torch.float16, torch.bfloat16) else t.view(torch.int32)


def main():
    configure_determinism()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = make_rows(tok, 6, 16, 0)
    ids, mask = left_pad([p for p, _ in rows], pad_id, device)
    pos = positions_from_mask(mask)
    k = 16

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    ref = run(ref_model, ids, mask, pos, rows, k, device).float()
    del ref_model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager").to(device).eval()
    logits = {}
    for name, fn in [("original", ORIGINAL), ("replica", make_eager(None)),
                     ("fp32_mask", make_eager("fp32_mask")), ("zero_masked_rows", make_eager("zero_masked_rows"))]:
        qwen2.eager_attention_forward = fn
        logits[name] = run(model, ids, mask, pos, rows, k, device)
    qwen2.eager_attention_forward = ORIGINAL
    del model
    torch.cuda.empty_cache()
    sdpa_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    logits["sdpa_baseline"] = run(sdpa_model, ids, mask, pos, rows, k, device)

    bad_rows_original = (~torch.isfinite(logits["original"].float())).flatten(1).any(1)   # rows the bug hit
    report = {"model": MODEL, "torch": torch.__version__, "prompt_lengths": [len(p) for p, _ in rows],
              "rows_hit_by_bug_in_original": bad_rows_original.nonzero().flatten().tolist(), "variants": {}}
    for name, lg in logits.items():
        f = lg.float()
        nonfinite_rows = (~torch.isfinite(f)).flatten(1).any(1)
        finite_rows = ~nonfinite_rows
        diff = (f - ref).abs().flatten(1).amax(1)                     # per row, over steps and vocab
        agree = (f.argmax(-1) == ref.argmax(-1)).float().mean(1)      # per row, over steps
        entry = {
            "nonfinite_rows": nonfinite_rows.nonzero().flatten().tolist(),
            "untouched_rows_bit_identical_to_original": bool(torch.equal(
                bits(lg[~bad_rows_original]), bits(logits["original"][~bad_rows_original]))) if lg.dtype == logits["original"].dtype else None,
            "max_abs_diff_vs_fp32_per_row": [round(float(d), 4) if fin else None for d, fin in zip(diff, finite_rows)],
            "argmax_agreement_vs_fp32_per_row": [round(float(a), 3) if fin else None for a, fin in zip(agree, finite_rows)],
        }
        report["variants"][name] = entry
    report["replica_bit_identical_to_original"] = bool(torch.equal(
        torch.nan_to_num(logits["replica"]).view(torch.int16), torch.nan_to_num(logits["original"]).view(torch.int16))
        and torch.equal(torch.isnan(logits["replica"]), torch.isnan(logits["original"])))

    print(json.dumps(report, indent=2))
    with open("test_fixes.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote test_fixes.json")


if __name__ == "__main__":
    main()

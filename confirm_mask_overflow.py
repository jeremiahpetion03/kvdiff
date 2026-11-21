"""Test the mask-overflow hypothesis for the Qwen2.5 eager fp16 NaN directly.

snapshot.py showed that the NaN appears in layer 1's softmax weights (padding
query rows only), that the mask uses fp16's minimum (-65504) rather than -inf,
and that o_proj does not overflow. Hypothesis: in fp16, score + (-65504)
rounds to -inf once the score is below about -16 (values near 65504 are 32
apart). If EVERY key in a row overflows, softmax over an all -inf row is NaN.

This wraps Transformers' eager attention function and, for each layer,
measures on kvdiff's exact failing inputs:
  * rows (batch, head, query) whose masked scores are ALL -inf in fp16,
  * the same sum done in fp32 (no -inf expected there),
  * the largest raw score in those rows (the hypothesis needs it below -16),
  * whether the NaN softmax rows are exactly the all -inf rows.
It does not change the computation: the original function runs unchanged.

    python confirm_mask_overflow.py
"""

import json

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import configure_determinism, left_pad, make_rows, positions_from_mask

MODEL = "Qwen/Qwen2.5-0.5B"
original = qwen2.eager_attention_forward
report = {}


def wrapped(module, query, key, value, attention_mask, scaling, *args, **kwargs):
    out = original(module, query, key, value, attention_mask, scaling, *args, **kwargs)
    with torch.no_grad():
        k = qwen2.repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, k.transpose(2, 3)) * scaling         # same op and dtype as eager
        m = attention_mask[:, :, :, : k.shape[-2]]
        masked16 = scores + m                                              # (B, H, T, T)
        masked32 = scores.float() + m.float()
        all_inf = torch.isneginf(masked16).all(-1)                         # (B, H, T)
        weights = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        nan_rows = torch.isnan(weights).any(-1) if isinstance(weights, torch.Tensor) else None
        entry = {"dtype": str(scores.dtype), "mask_min": float(m.min()),
                 "rows_all_neg_inf_fp16": int(all_inf.sum()),
                 "rows_with_any_neg_inf_fp32": int(torch.isneginf(masked32).any(-1).sum())}
        if all_inf.any():
            b, h, q = all_inf.nonzero().unbind(-1)
            entry["batches"] = sorted(set(b.tolist()))
            entry["heads"] = sorted(set(h.tolist()))
            entry["query_positions"] = sorted(set(q.tolist()))
            entry["max_raw_score_in_those_rows"] = float(scores[all_inf].float().max())
        if nan_rows is not None:
            entry["softmax_nan_rows"] = int(nan_rows.sum())
            entry["nan_rows_equal_all_neg_inf_rows"] = bool(torch.equal(nan_rows, all_inf))
        report.setdefault(f"layer_{module.layer_idx}", entry)              # first call = prefill
    return out


def main():
    configure_determinism()
    qwen2.eager_attention_forward = wrapped
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = make_rows(tok, 6, 16, 0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 attn_implementation="eager").to(device).eval()
    ids, mask = left_pad([p for p, _ in rows], pad_id, device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask), use_cache=True)
    qwen2.eager_attention_forward = original
    report["called"] = bool(report)
    report["prefill_logits_nonfinite_rows"] = (~torch.isfinite(out.logits[:, -1])).any(-1).nonzero().flatten().tolist()
    for name in ("layer_0", "layer_1", "layer_2"):
        if name in report:
            print(name, json.dumps(report[name]))
    with open("confirm_mask_overflow.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nlayers with all -inf rows:", [k for k, v in report.items()
                                          if isinstance(v, dict) and v.get("rows_all_neg_inf_fp16")])
    print("wrote confirm_mask_overflow.json")


if __name__ == "__main__":
    main()

"""Instrument the Qwen2.5 eager fp16 NaN directly, on kvdiff's exact failing
inputs, instead of inferring the mechanism.

Three measurements:

1. FIRST NON-FINITE SNAPSHOT. A forward pre-hook on every module copies its
   tensor inputs BEFORE it runs (so an in-place op can't hide anything); a
   forward hook checks its output. At the first module whose output contains
   NaN, +inf or -inf, the script keeps that module's inputs and output and
   records shapes, dtypes, the position of the first bad element, and whether
   it is a padding position.

2. ATTENTION PROBE (layers 0 and 1). The additive attention mask (dtype, value
   used for masked keys), the softmax weights of padding vs real query rows,
   and the max |value| entering and leaving o_proj at padding vs real
   positions. o_proj is also recomputed in fp32: if the fp32 result exceeds
   fp16's max (65,504), the overflow is in the math, not in accumulation.

3. PROPAGATION. After prefill: which cached keys/values are non-finite at
   padding vs real positions. Then decode step by step and record which rows'
   logits are non-finite at each step.

Control: logits with hooks must be bit-identical to logits without hooks.

    python snapshot.py                 # writes snapshot.json and snapshot.pt
    python snapshot.py --attn sdpa     # control run
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import DTYPES, configure_determinism, left_pad, make_rows, positions_from_mask
from repro import cache_tensors

MODEL = "Qwen/Qwen2.5-0.5B"
FP16_MAX = 65504.0


def tensors_in(obj):
    """Tensors in an argument, one level into tuples/lists/dicts."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for t in obj if isinstance(t, torch.Tensor)]
    if isinstance(obj, dict):
        return [t for t in obj.values() if isinstance(t, torch.Tensor)]
    return []


def float_tensors(obj):
    return [t for t in tensors_in(obj) if t.is_floating_point()]


def describe(t: torch.Tensor) -> dict:
    f = t.detach().float()
    finite = torch.isfinite(f)
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
            "n_nan": int(torch.isnan(f).sum()), "n_posinf": int(torch.isposinf(f).sum()),
            "n_neginf": int(torch.isneginf(f).sum()),
            "max_abs_finite": float(f[finite].abs().max()) if finite.any() else None}


def split_max(t: torch.Tensor, pad: torch.Tensor) -> dict:
    """Max |value| at padding vs real positions for a (B, T, ...) tensor."""
    if t.dim() < 2 or tuple(t.shape[:2]) != tuple(pad.shape):
        return {}
    a = t.detach().float().abs().flatten(2).nan_to_num(nan=float("inf")).amax(-1)  # (B, T)
    out = {}
    for label, sel in (("pad", pad), ("real", ~pad)):
        out[label] = float(a[sel].max()) if sel.any() else None
    return out


class Snapshot:
    def __init__(self, model, pad: torch.Tensor):
        self.pad = pad          # (B, T) prefill padding mask, True = padding
        self.active = True
        self.first = None       # report for the first bad module
        self.saved = {}         # tensors for snapshot.pt
        self.pending = {}       # name -> input copies while the module runs
        self.handles = []
        for name, m in model.named_modules():
            if not name:
                continue
            self.handles.append(m.register_forward_pre_hook(self._pre(name), with_kwargs=True))
            self.handles.append(m.register_forward_hook(self._post(name), with_kwargs=True))

    def _pre(self, name):
        def fn(_m, args, kwargs):
            if self.active and self.first is None:
                ins = [t.detach().clone() for a in args for t in float_tensors(a)]
                hs = kwargs.get("hidden_states")
                if isinstance(hs, torch.Tensor):
                    ins.append(hs.detach().clone())
                self.pending[name] = ins
        return fn

    def _post(self, name):
        def fn(_m, _args, _kwargs, output):
            ins = self.pending.pop(name, None)
            if not self.active or self.first is not None:
                return
            outs = float_tensors(output)
            bad = [o for o in outs if not torch.isfinite(o).all()]
            if not bad:
                return
            o = bad[0]
            idx = tuple(int(v) for v in (~torch.isfinite(o)).nonzero()[0])
            is_pad = bool(self.pad[idx[0], idx[1]]) if o.dim() >= 2 and tuple(o.shape[:2]) == tuple(self.pad.shape) else None
            self.first = {
                "module": name,
                "first_bad_index": list(idx),
                "first_bad_is_padding_position": is_pad,
                "inputs": [describe(t) | {"max_abs_by_position": split_max(t, self.pad)} for t in (ins or [])],
                "output": describe(o) | {"max_abs_by_position": split_max(o, self.pad)},
            }
            self.saved = {"inputs": [t.cpu() for t in (ins or [])], "output": o.detach().cpu(),
                          "pad_mask": self.pad.cpu()}
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


class AttentionProbe:
    """Records the mask, attention weights and o_proj in/out for chosen layers."""

    def __init__(self, model, layers, pad):
        self.pad, self.active, self.report, self.handles = pad, True, {}, []
        base = model.model.layers
        for i in layers:
            attn = base[i].self_attn
            self.handles.append(attn.register_forward_hook(self._attn(i), with_kwargs=True))
            self.handles.append(attn.o_proj.register_forward_hook(self._oproj(i, attn.o_proj)))

    def _attn(self, i):
        def fn(_m, _args, kwargs, output):
            if not self.active:
                return
            r = self.report.setdefault(f"layer_{i}", {})
            mask = kwargs.get("attention_mask")
            if isinstance(mask, torch.Tensor):
                m = mask.detach().float()
                r["mask"] = {"shape": list(mask.shape), "dtype": str(mask.dtype),
                             "min": float(m.min()), "has_neg_inf": bool(torch.isneginf(m).any()),
                             "dtype_min": float(torch.finfo(mask.dtype).min) if mask.is_floating_point() else None}
            w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if isinstance(w, torch.Tensor):
                wf = w.detach().float()                       # (B, H, T, T)
                rows = {"pad": self.pad, "real": ~self.pad}
                stats = {}
                for label, sel in rows.items():
                    q = wf.permute(0, 2, 1, 3)[sel]           # (n_rows, H, T)
                    if q.numel() == 0:
                        continue
                    spread = (q.amax(-1) - q.amin(-1))
                    stats[label] = {"n_query_rows": int(q.shape[0]), "finite": bool(torch.isfinite(q).all()),
                                    "row_sum_min": float(q.sum(-1).min()), "row_sum_max": float(q.sum(-1).max()),
                                    "max_weight": float(q.max()),
                                    "fraction_uniform_rows": float((spread < 1e-3).float().mean())}
                r["attn_weights"] = {"shape": list(w.shape), "dtype": str(w.dtype)} | stats
            else:
                r["attn_weights"] = "not returned by this attention implementation"
        return fn

    def _oproj(self, i, lin):
        def fn(_m, args, output):
            if not self.active:
                return
            x = args[0].detach()
            r = self.report.setdefault(f"layer_{i}", {})
            r["o_proj_input"] = describe(x) | {"max_abs_by_position": split_max(x, self.pad)}
            r["o_proj_output"] = describe(output) | {"max_abs_by_position": split_max(output, self.pad)}
            ref = torch.nn.functional.linear(x.float(), lin.weight.float(),
                                             None if lin.bias is None else lin.bias.float())
            r["o_proj_output_recomputed_fp32"] = {
                "max_abs_by_position": split_max(ref, self.pad),
                "exceeds_fp16_max": bool(ref.abs().max() > FP16_MAX)}
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bits, so NaN == NaN when the payloads match (torch.equal says NaN != NaN)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    as_int = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
              torch.float64: torch.int64}.get(a.dtype)
    return bool(torch.equal(a.view(as_int), b.view(as_int))) if as_int else bool(torch.equal(a, b))


@torch.no_grad()
def prefill(model, ids, mask, pos):
    return model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--attn", default="eager")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-layers", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out", default="snapshot")
    args = p.parse_args(argv)
    configure_determinism()
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = make_rows(tok, args.rows, args.k, args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype],
                                                 attn_implementation=args.attn).to(device).eval()
    ids, mask = left_pad([pr for pr, _ in rows], pad_id, device)
    pos = positions_from_mask(mask)
    pad = mask == 0
    report = {"model": args.model, "attn": args.attn, "dtype": args.dtype,
              "prompt_lengths": [len(pr) for pr, _ in rows], "batch_shape": list(ids.shape),
              "pad_tokens_per_row": pad.sum(1).tolist(), "torch": torch.__version__}

    # Control: logits without hooks, then with hooks, must be bit-identical.
    plain = prefill(model, ids, mask, pos).logits
    snap = Snapshot(model, pad)
    probe = AttentionProbe(model, args.probe_layers, pad)
    out = prefill(model, ids, mask, pos)
    snap.active = probe.active = False
    hooked = out.logits
    report["hooks_bit_identical"] = bit_equal(plain, hooked)
    report["first_nonfinite"] = snap.first
    report["attention_probe"] = probe.report

    # Propagation 1: non-finite cache entries at padding vs real positions, per layer.
    cache_rows = []
    for li, (k, v) in enumerate(cache_tensors(out.past_key_values)):   # (B, kv_heads, T, d)
        bad = (~torch.isfinite(k)).any(-1).any(1) | (~torch.isfinite(v)).any(-1).any(1)  # (B, T)
        if bad.any():
            cache_rows.append({"layer": li, "nonfinite_pad_positions": int((bad & pad).sum()),
                               "nonfinite_real_positions": int((bad & ~pad).sum())})
    report["kv_cache_after_prefill"] = cache_rows or "all finite"
    report["prefill_last_token_logits_nonfinite_rows"] = (
        (~torch.isfinite(out.logits[:, -1])).any(-1).nonzero().flatten().tolist())

    # Propagation 2: decode token by token (teacher forcing, as kvdiff does).
    steps, past, last, cur = [], out.past_key_values, pos[:, -1:], mask
    for j in range(args.k - 1):
        tok_j = torch.tensor([[c[j]] for _, c in rows], device=device)
        cur = torch.cat([cur, torch.ones_like(tok_j)], dim=1)
        last = last + 1
        with torch.no_grad():
            o = model(input_ids=tok_j, attention_mask=cur, position_ids=last, past_key_values=past, use_cache=True)
        past = o.past_key_values
        steps.append((~torch.isfinite(o.logits[:, -1])).any(-1).nonzero().flatten().tolist())
    report["decode_nonfinite_rows_by_step"] = steps
    snap.remove(); probe.remove()

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if snap.saved:
        torch.save(snap.saved, args.out + ".pt")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}.json" + (f" and {args.out}.pt" if snap.saved else ""))


if __name__ == "__main__":
    main()

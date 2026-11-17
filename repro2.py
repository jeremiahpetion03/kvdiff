"""Re-run the EXACT inputs where triage saw NaN (kvdiff's rows, seed 0) many
times, and locate where non-finite values first appear.

Distinguishes two cases:
  * input-dependent: NaN on every (or almost every) trial with these inputs
  * flaky: NaN only sometimes on identical inputs (nondeterministic kernels,
    hardware, or memory state), which is much harder to report upstream

On each trial a forward hook records the FIRST module, in execution order,
whose output contains inf or NaN, along with the decode step.

    python repro2.py
    python repro2.py --trials 50 --attn eager --dtype fp16
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvdiff import DTYPES, configure_determinism, left_pad, make_rows, positions_from_mask

MODEL = "Qwen/Qwen2.5-0.5B"


def nonfinite(x) -> bool:
    if isinstance(x, torch.Tensor):
        return x.is_floating_point() and not torch.isfinite(x).all().item()
    if isinstance(x, (tuple, list)):
        return any(nonfinite(t) for t in x)
    return False


class FirstBad:
    """Forward hooks that remember the first module to emit inf/NaN."""

    def __init__(self, model):
        self.first = None
        self.step = None
        self.current_step = -1
        self.handles = [m.register_forward_hook(self._hook(name)) for name, m in model.named_modules() if name]

    def _hook(self, name):
        def fn(_module, _inputs, output):
            if self.first is None and nonfinite(output):
                self.first, self.step = name, self.current_step
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def cached_run(model, rows, k, pad_id, device, tracker: FirstBad) -> torch.Tensor:
    """Same computation as kvdiff.logits_padded_cached, with step tracking."""
    ids, mask = left_pad([p for p, _ in rows], pad_id, device)
    pos = positions_from_mask(mask)
    tracker.current_step = 0  # prefill
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    steps = [out.logits[:, -1]]
    past, last = out.past_key_values, pos[:, -1:]
    for j in range(k - 1):
        tracker.current_step = j + 1
        tok_j = torch.tensor([[c[j]] for _, c in rows], device=device)
        mask = torch.cat([mask, torch.ones_like(tok_j)], dim=1)
        last = last + 1
        out = model(input_ids=tok_j, attention_mask=mask, position_ids=last, past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits[:, -1])
    return torch.stack(steps, dim=1).float()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--attn", default="eager")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    configure_determinism()
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = make_rows(tok, args.rows, args.k, args.seed)
    print("prompt lengths:", [len(pr) for pr, _ in rows])

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype],
                                                 attn_implementation=args.attn).to(device).eval()
    tracker = FirstBad(model)
    bad_trials = 0
    where = Counter()
    bad_rows = Counter()
    first_logits = None
    identical = True
    for _ in range(args.trials):
        tracker.first = tracker.step = None
        logits = cached_run(model, rows, args.k, pad_id, device, tracker)
        if first_logits is None:
            first_logits = logits
        elif not torch.equal(torch.nan_to_num(logits), torch.nan_to_num(first_logits)):
            identical = False
        row_bad = (~torch.isfinite(logits)).any(dim=(1, 2))
        if row_bad.any():
            bad_trials += 1
            where[(tracker.first, tracker.step)] += 1
            for r in row_bad.nonzero().flatten().tolist():
                bad_rows[r] += 1
    tracker.remove()

    print(f"\n{args.attn} {args.dtype}: non-finite logits in {bad_trials}/{args.trials} trials")
    print(f"outputs bit-identical across all trials: {identical}")
    if bad_trials:
        print("rows affected (row: trials):", dict(bad_rows))
        print("first module to emit inf/NaN (module, step): trials")
        for (name, step), n in where.most_common(10):
            print(f"  {name}  step {step}{' (prefill)' if step == 0 else ''}: {n}")


if __name__ == "__main__":
    main()

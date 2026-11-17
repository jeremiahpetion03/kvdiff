"""kvdiff: a differential tester for KV-cache correctness in Hugging Face Transformers.

For every model x attention implementation x dtype it checks three properties
users rely on, each against a reference that does not share the code path
under test:

  padding   padded batch, full forward          vs  each row alone, unpadded
  cache     padded batch, token-by-token         vs  padded batch, full forward
            decoding through the KV cache
  generate  model.generate(use_cache=True),     vs  each row generated alone,
            batched with left padding                with no cache

Logit checks report the largest difference relative to the logit scale, and
count argmax flips. Floating-point noise can flip a near-tie, so a flip only
counts as a candidate bug when the reference preferred its token by a clear
margin. Anything flagged is a CANDIDATE: reduce it to a minimal reproduction
and rule out expected behavior before reporting it upstream.

    python kvdiff.py
    python kvdiff.py --models openai-community/gpt2 --dtypes fp32 --attn eager sdpa
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "openai-community/gpt2",        # learned absolute positions
    "HuggingFaceTB/SmolLM2-135M",   # Llama architecture, RoPE
    "Qwen/Qwen2.5-0.5B",            # RoPE + grouped-query attention
]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# A logit check is flagged when max|test - ref| / max|ref| exceeds this.
REL_TOL = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 5e-2}
# A token flip is a candidate bug only if the reference's top-1 beat its
# top-2 by more than this many logits; smaller margins are near-ties.
MARGIN = {"fp32": 1e-2, "fp16": 0.25, "bf16": 0.75}

# Original text, used only as a source of realistic token sequences.
PASSAGE = (
    "The harbor was quiet before dawn, and the fishing boats rocked against their moorings while the "
    "gulls waited on the pilings. Mara walked the length of the pier with a thermos of coffee and a "
    "notebook full of measurements she had taken every morning for three years: water temperature, "
    "wind direction, the height of the tide at six o'clock, and the number of boats that left before "
    "sunrise. Her neighbors thought the habit was strange, but the notebook had already told her things "
    "nobody else in town had noticed. The tides were arriving a few minutes later each spring, the "
    "water was warmer in October than it used to be in August, and the boats that went out earliest "
    "came back with the smallest catches. She had started to sketch a simple model on the back pages, "
    "a line for each variable and arrows showing which ones seemed to move together. Some of the arrows "
    "were obvious, like wind and wave height, but others surprised her. When the water warmed, the "
    "older captains stayed home, and the younger ones went farther out. That one change explained half "
    "of the pattern she had been chasing. She wrote it down carefully, underlined it twice, and then "
    "added a question underneath, because a good explanation always seemed to raise a better question. "
    "By the time the sun cleared the breakwater, the first boats were already turning for home."
)


@dataclass
class Result:
    model: str
    attn: str
    dtype: str
    check: str
    status: str            # OK | FLAG | SKIP | ERROR
    max_rel_diff: float | None = None
    flips: int = 0         # argmax / token disagreements of any size
    real_flips: int = 0    # disagreements where the reference margin was clear
    note: str = ""


def configure_determinism() -> None:
    # Ampere GPUs (e.g. RTX 30-series) can run fp32 matmuls as TF32, which
    # keeps only ~10 mantissa bits and would hide real fp32 discrepancies.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_rows(tok, n_rows: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """(prompt, continuation) token pairs with deliberately varied prompt lengths,
    including a very short one, so the batch needs heavy left padding."""
    ids = tok(PASSAGE, add_special_tokens=False)["input_ids"]
    max_prompt = 48
    if len(ids) < max_prompt + k + 1:
        raise ValueError("passage too short for the requested prompt and continuation lengths")
    rng = random.Random(seed)
    lengths = [3] + [rng.randint(5, max_prompt) for _ in range(n_rows - 1)]
    rows = []
    for p in lengths:
        start = rng.randint(0, len(ids) - p - k)
        rows.append((ids[start:start + p], ids[start + p:start + p + k]))
    return rows


def left_pad(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    return ids.to(device), mask.to(device)


def positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Position 0 at each row's first real token, so padding does not shift positions."""
    return (mask.cumsum(-1) - 1).clamp(min=0)


# --------------------------------------------------------------------------- #
# The three forward paths
# --------------------------------------------------------------------------- #


@torch.no_grad()
def logits_unpadded(model, rows, k: int, device) -> list[torch.Tensor]:
    """Reference: each row alone, no padding, no cache. Returns per row a (k, V)
    tensor: the logits predicting each continuation token."""
    out = []
    for prompt, cont in rows:
        seq = torch.tensor([prompt + cont[:k - 1]], device=device)
        out.append(model(input_ids=seq, use_cache=False).logits[0, -k:].float())
    return out


@torch.no_grad()
def logits_padded_full(model, rows, k: int, pad_id: int, device) -> torch.Tensor:
    ids, mask = left_pad([p + c[:k - 1] for p, c in rows], pad_id, device)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask),
                   use_cache=False).logits
    return logits[:, -k:].float()


@torch.no_grad()
def logits_padded_cached(model, rows, k: int, pad_id: int, device) -> torch.Tensor:
    """Prefill the padded prompts, then feed the continuation one token at a time
    through the KV cache (teacher forcing, so both paths see identical tokens)."""
    ids, mask = left_pad([p for p, _ in rows], pad_id, device)
    pos = positions_from_mask(mask)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    steps = [out.logits[:, -1]]
    past, last_pos = out.past_key_values, pos[:, -1:]
    for j in range(k - 1):
        tok_j = torch.tensor([[c[j]] for _, c in rows], device=device)
        mask = torch.cat([mask, torch.ones_like(tok_j)], dim=1)
        last_pos = last_pos + 1
        out = model(input_ids=tok_j, attention_mask=mask, position_ids=last_pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits[:, -1])
    return torch.stack(steps, dim=1).float()


@torch.no_grad()
def greedy(model, ids, mask, k: int, pad_id: int, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=k, min_new_tokens=k,
                         do_sample=False, use_cache=use_cache, pad_token_id=pad_id,
                         output_scores=True, return_dict_in_generate=True)
    return out.sequences[:, ids.shape[1]:], torch.stack(out.scores, dim=1).float()


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def compare_logits(test: torch.Tensor, ref: torch.Tensor, dtype_name: str) -> tuple[float, int, int]:
    """(relative max diff, all argmax flips, flips where the reference margin was clear)."""
    rel = ((test - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    flips = test.argmax(-1) != ref.argmax(-1)
    return rel, int(flips.sum()), int((flips & (margin > MARGIN[dtype_name])).sum())


def logit_result(name, attn, dtype_name, check, pairs) -> Result:
    worst, flips, real = 0.0, 0, 0
    for test, ref in pairs:
        rel, f, r = compare_logits(test, ref, dtype_name)
        worst = rel if (math.isnan(rel) or math.isnan(worst)) else max(worst, rel)
        flips, real = flips + f, real + r
    ok = (not math.isnan(worst)) and worst <= REL_TOL[dtype_name] and real == 0
    return Result(name, attn, dtype_name, check, "OK" if ok else "FLAG", worst, flips, real)


def first_divergence(a: torch.Tensor, b: torch.Tensor) -> int | None:
    diff = (a != b).nonzero()
    return int(diff[0]) if len(diff) else None


def run_config(name: str, attn: str, dtype_name: str, rows, k: int, device) -> list[Result]:
    tok = AutoTokenizer.from_pretrained(name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPES[dtype_name],
                                                     attn_implementation=attn).to(device).eval()
    except (ValueError, ImportError) as e:
        return [Result(name, attn, dtype_name, "load", "SKIP", note=str(e).splitlines()[0][:160])]

    results = []
    ref = logits_unpadded(model, rows, k, device)
    full = logits_padded_full(model, rows, k, pad_id, device)
    cached = logits_padded_cached(model, rows, k, pad_id, device)
    results.append(logit_result(name, attn, dtype_name, "padding (batched vs alone)",
                                [(full[i], ref[i]) for i in range(len(rows))]))
    results.append(logit_result(name, attn, dtype_name, "cache (cached vs full forward)",
                                [(cached[i], full[i]) for i in range(len(rows))]))

    # generate(): the path users actually call. Reference: each row alone, no cache.
    ids, mask = left_pad([p for p, _ in rows], pad_id, device)
    batched, _ = greedy(model, ids, mask, k, pad_id, use_cache=True)
    flips = real = 0
    notes = []
    for i, (prompt, _) in enumerate(rows):
        one = torch.tensor([prompt], device=device)
        alone, alone_scores = greedy(model, one, torch.ones_like(one), k, pad_id, use_cache=False)
        j = first_divergence(batched[i], alone[0])
        if j is None:
            continue
        flips += 1
        top2 = alone_scores[0, j].topk(2).values
        margin = float(top2[0] - top2[1])
        if margin > MARGIN[dtype_name]:
            real += 1
            notes.append(f"row {i} (prompt len {len(prompt)}) diverges at step {j}, ref margin {margin:.3f}")
    results.append(Result(name, attn, dtype_name, "generate (batched+cache vs alone, no cache)",
                          "OK" if real == 0 else "FLAG", None, flips, real, "; ".join(notes)))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--dtypes", nargs="+", default=["fp32", "bf16", "fp16"], choices=list(DTYPES))
    p.add_argument("--attn", nargs="+", default=["eager", "sdpa"])
    p.add_argument("--rows", type=int, default=6, help="batch size (prompts of different lengths)")
    p.add_argument("--k", type=int, default=16, help="continuation length checked")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("results.json"))
    args = p.parse_args(argv)

    configure_determinism()
    device = torch.device(args.device)
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version(), "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    print(json.dumps(env))

    results: list[Result] = []
    t0 = time.perf_counter()
    for name in args.models:
        rows = make_rows(AutoTokenizer.from_pretrained(name), args.rows, args.k, args.seed)
        for attn in args.attn:
            for dtype_name in args.dtypes:
                try:
                    batch = run_config(name, attn, dtype_name, rows, args.k, device)
                except Exception as e:  # a crash on valid input is itself a finding
                    batch = [Result(name, attn, dtype_name, "run", "ERROR",
                                    note=f"{type(e).__name__}: {e}"[:300])]
                    traceback.print_exc()
                for r in batch:
                    rel = "" if r.max_rel_diff is None else f"{r.max_rel_diff:.2e}"
                    print(f"{r.status:5s}  {r.model:28s} {r.attn:6s} {r.dtype:5s} {r.check:46s} "
                          f"rel={rel:9s} flips={r.flips} real={r.real_flips} {r.note}")
                results.extend(batch)

    flagged = [r for r in results if r.status in ("FLAG", "ERROR")]
    args.out.write_text(json.dumps({"env": env, "args": {k: str(v) for k, v in vars(args).items()},
                                    "results": [asdict(r) for r in results]}, indent=2))
    print(f"\n{len(results)} checks in {time.perf_counter() - t0:.0f}s, "
          f"{len(flagged)} flagged. Wrote {args.out}")


if __name__ == "__main__":
    main()

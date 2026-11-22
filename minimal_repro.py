"""Standalone reproduction: Qwen2.5-0.5B, eager attention, fp16, left padding -> NaN logits.

Needs only torch and transformers (CUDA GPU). One 3-token prompt (" used to be")
left-padded with 34 pad tokens, as in a batch whose longest prompt is 37 tokens.

Cases:
  A  eager fp16, explicit position_ids from the mask       (the failing case)
  B  eager fp16, no position_ids (model default positions)
  C  sdpa  fp16, explicit position_ids                      (control)
  D  eager bf16, explicit position_ids                      (control)
  E  eager fp16, the same prompt with NO padding            (control)
  F  generate() on the padded input, eager fp16             (the path most users call)
  G  sweep: eager fp16 NaN or not, for 0..40 pad tokens

    python minimal_repro.py
"""

import platform

import torch
import transformers
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = [1483, 311, 387]   # " used to be"
PAD = 151643                # <|endoftext|>, Qwen2.5's pad and eos token
N_PAD = 34
DEVICE = "cuda"


def padded(n_pad):
    ids = torch.tensor([[PAD] * n_pad + PROMPT], device=DEVICE)
    mask = torch.tensor([[0] * n_pad + [1] * len(PROMPT)], device=DEVICE)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)     # position 0 at the first real token
    return ids, mask, pos


def load(dtype, attn):
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(DEVICE).eval()


@torch.no_grad()
def nonfinite(model, ids, mask, pos=None):
    kwargs = {"input_ids": ids, "attention_mask": mask}
    if pos is not None:
        kwargs["position_ids"] = pos
    logits = model(**kwargs).logits
    return bool((~torch.isfinite(logits[:, -1])).any())


def main():
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | python {platform.python_version()} | "
          f"{torch.cuda.get_device_name(0)}")
    results = {}
    eager16 = load(torch.float16, "eager")
    ids, mask, pos = padded(N_PAD)
    results["A eager fp16, explicit position_ids"] = nonfinite(eager16, ids, mask, pos)
    results["B eager fp16, default positions"] = nonfinite(eager16, ids, mask)
    e_ids, e_mask, e_pos = padded(0)
    results["E eager fp16, no padding"] = nonfinite(eager16, e_ids, e_mask, e_pos)

    with torch.no_grad():
        out = eager16.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                               output_scores=True, return_dict_in_generate=True, pad_token_id=PAD)
    scores_bad = any(bool((~torch.isfinite(s)).any()) for s in out.scores)
    results["F generate(), eager fp16"] = scores_bad
    gen_tokens = out.sequences[0, ids.shape[1]:].tolist()

    sweep = [n for n in range(0, 41) if nonfinite(eager16, *padded(n))]
    del eager16
    torch.cuda.empty_cache()

    sdpa16 = load(torch.float16, "sdpa")
    results["C sdpa fp16, explicit position_ids"] = nonfinite(sdpa16, ids, mask, pos)
    del sdpa16
    torch.cuda.empty_cache()
    eagerbf = load(torch.bfloat16, "eager")
    results["D eager bf16, explicit position_ids"] = nonfinite(eagerbf, ids, mask, pos)

    print(f"\nprompt {PROMPT} (' used to be'), {N_PAD} left pad tokens (id {PAD})\n")
    for name in sorted(results):
        print(f"  {name:40s} {'NaN/inf logits' if results[name] else 'finite'}")
    print(f"\n  generate() tokens: {gen_tokens}")
    print(f"  G sweep, pad counts 0..40 with non-finite logits (eager fp16, explicit positions): {sweep or 'none'}")


if __name__ == "__main__":
    main()

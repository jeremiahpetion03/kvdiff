"""Does the Qwen2.5 eager fp16 NaN poison gradients, and do the two fixes repair them?

Same failing input as minimal_repro.py (a 3-token prompt, left-padded with 34 pad tokens),
but run with labels and loss.backward(), as in training or fine-tuning. Pad positions get
label -100, so the loss covers only the real tokens.

Variants, all in fp16 on the same weights and input:
  original          Transformers' own eager attention (the bug)
  fp32_mask         add the mask in fp32 before the softmax
  zero_masked_rows  zero the weights of query rows with no allowed key
  zero_rows_safe    the same, with the softmax input for those rows set to 0 first
  sdpa              the sdpa backend (control)
and an fp32 eager model as the gradient reference.

For each: is the loss finite, how many gradient elements are non-finite, the total gradient
norm, and the cosine similarity of the gradient with the fp32 reference gradient.

    python backward_check.py        # writes backward_check.json
"""

import json
import math

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM

from kvdiff import configure_determinism
from test_fixes import ORIGINAL, make_eager

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = [1483, 311, 387]   # " used to be"
PAD = 151643
N_PAD = 34
DEVICE = "cuda"


def make_batch():
    ids = torch.tensor([[PAD] * N_PAD + PROMPT], device=DEVICE)
    mask = torch.tensor([[0] * N_PAD + [1] * len(PROMPT)], device=DEVICE)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    labels = ids.clone()
    labels[mask == 0] = -100
    return ids, mask, pos, labels


def load(dtype, attn):
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(DEVICE).eval()


def forward_backward(model, batch):
    ids, mask, pos, labels = batch
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, attention_mask=mask, position_ids=pos, labels=labels).loss
    loss.backward()
    return loss.detach()


def grad_summary(model):
    bad = total = 0
    sq = 0.0
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        gf = g.float()
        bad += int((~torch.isfinite(gf)).sum())
        total += gf.numel()
        sq += float(gf.double().pow(2).sum())
    return {"nonfinite_grad_elements": bad, "total_grad_elements": total,
            "grad_norm": math.sqrt(sq) if math.isfinite(sq) else None}


def grads_to_cpu(model):
    return {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}


def cosine_vs(model, ref):
    dot = n1 = n2 = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().cpu()
        r = ref[n]
        dot += float((g * r).sum(dtype=torch.float64))
        n1 += float(g.pow(2).sum(dtype=torch.float64))
        n2 += float(r.pow(2).sum(dtype=torch.float64))
    if not all(map(math.isfinite, (dot, n1, n2))) or n1 == 0 or n2 == 0:
        return None
    return dot / math.sqrt(n1 * n2)


def main():
    configure_determinism()
    batch = make_batch()
    report = {"model": MODEL, "torch": torch.__version__, "prompt": PROMPT, "n_pad": N_PAD, "variants": {}}

    ref_model = load(torch.float32, "eager")
    forward_backward(ref_model, batch)
    ref = grads_to_cpu(ref_model)
    report["fp32_reference"] = grad_summary(ref_model)
    del ref_model
    torch.cuda.empty_cache()

    model = load(torch.float16, "eager")
    for name, fn in [("original", ORIGINAL), ("fp32_mask", make_eager("fp32_mask")),
                     ("zero_masked_rows", make_eager("zero_masked_rows")),
                     ("zero_rows_safe", make_eager("zero_rows_safe"))]:
        qwen2.eager_attention_forward = fn
        loss = forward_backward(model, batch)
        entry = {"loss": float(loss) if math.isfinite(float(loss)) else None, "loss_finite": bool(torch.isfinite(loss))}
        entry.update(grad_summary(model))
        entry["cosine_vs_fp32_grad"] = cosine_vs(model, ref)
        report["variants"][name] = entry
    qwen2.eager_attention_forward = ORIGINAL
    del model
    torch.cuda.empty_cache()

    sdpa = load(torch.float16, "sdpa")
    loss = forward_backward(sdpa, batch)
    entry = {"loss": float(loss) if math.isfinite(float(loss)) else None, "loss_finite": bool(torch.isfinite(loss))}
    entry.update(grad_summary(sdpa))
    entry["cosine_vs_fp32_grad"] = cosine_vs(sdpa, ref)
    report["variants"]["sdpa_baseline"] = entry

    print(json.dumps(report, indent=2))
    with open("backward_check.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote backward_check.json")


if __name__ == "__main__":
    main()

"""Function-level regression test for the fp16 mask overflow in eager attention.

No model download. It builds a tiny attention problem in fp16 where two padding query
rows have every key masked and every raw score below about -16, so `score + fp16 min`
rounds to -inf, then calls Transformers' own Qwen2 eager attention function and the two
tested fixes.

    pytest -v test_mask_overflow_regression.py
"""

import pytest
import torch
from torch import nn

import transformers.models.qwen2.modeling_qwen2 as qwen2
from test_fixes import ORIGINAL, make_eager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ROWS = [0, 1]
REAL_ROWS = [2, 3]


FIXES = ["fp32_mask", "zero_masked_rows", "zero_rows_safe"]


class Stub(nn.Module):
    num_key_value_groups = 1


def inputs(dtype=torch.float16):
    """Batch 1, 1 head, 4 positions: rows 0-1 are left padding, rows 2-3 are real tokens."""
    g = torch.Generator().manual_seed(0)
    q = torch.zeros(1, 1, 4, 4)
    q[0, 0, :2, 0] = 1.0                      # padding rows read the first channel
    q[0, 0, 2:, 1] = 1.0                      # real rows read the second channel
    k = torch.randn(1, 1, 4, 4, generator=g) * 0.1
    k[0, 0, :, 0] = -30.0                     # every score for the padding rows is -30, below -16
    v = torch.randn(1, 1, 4, 4, generator=g)
    allowed = torch.zeros(4, 4, dtype=torch.bool)
    for i in REAL_ROWS:
        for j in REAL_ROWS:
            allowed[i, j] = j <= i            # causal over real tokens; padding keys never allowed
    mask = torch.zeros(4, 4).masked_fill(~allowed, torch.finfo(dtype).min).view(1, 1, 4, 4)
    return [t.to(device=DEVICE, dtype=dtype) for t in (q, k, v, mask)]


def call(fn, q, k, v, mask):
    return fn(Stub().eval(), q, k, v, mask, 1.0)   # (attn_output, attn_weights)


def test_original_function_is_nonfinite_on_fully_masked_rows():
    q, k, v, mask = inputs()
    _, w = call(ORIGINAL, q, k, v, mask)
    if torch.isfinite(w).all():
        pytest.skip("Transformers' eager attention is finite here: the bug appears to be fixed upstream.")
    assert torch.isnan(w[0, 0, PAD_ROWS]).all()
    assert torch.isfinite(w[0, 0, REAL_ROWS]).all()


@pytest.mark.parametrize("fix", FIXES)
def test_fix_gives_finite_outputs(fix):
    q, k, v, mask = inputs()
    out, w = call(make_eager(fix), q, k, v, mask)
    assert torch.isfinite(w).all()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("fix", FIXES)
def test_fix_leaves_unaffected_rows_bit_identical(fix):
    q, k, v, mask = inputs()
    _, w_orig = call(ORIGINAL, q, k, v, mask)
    _, w_fix = call(make_eager(fix), q, k, v, mask)
    assert torch.equal(w_orig[0, 0, REAL_ROWS].view(torch.int16), w_fix[0, 0, REAL_ROWS].view(torch.int16))


@pytest.mark.parametrize("fix", ["zero_masked_rows", "zero_rows_safe"])
def test_zeroing_fixes_match_sdpa_semantics(fix):
    q, k, v, mask = inputs()
    out, w = call(make_eager(fix), q, k, v, mask)
    assert (w[0, 0, PAD_ROWS] == 0).all()


def grads(fix):
    q, k, v, mask = inputs()
    for t in (q, k, v):
        t.requires_grad_()
    out, _ = call(make_eager(fix), q, k, v, mask)
    out.float().sum().backward()
    return [q.grad, k.grad, v.grad]


@pytest.mark.parametrize("fix", ["fp32_mask", "zero_rows_safe"])
def test_backward_is_finite(fix):
    assert all(torch.isfinite(g).all() for g in grads(fix))


@pytest.mark.xfail(strict=False, reason="measured on Qwen2.5-0.5B (backward_check.py): zeroing the weights AFTER "
                   "the softmax leaves the softmax output NaN, so its backward pass is NaN too")
def test_zeroing_after_softmax_has_finite_backward():
    assert all(torch.isfinite(g).all() for g in grads("zero_masked_rows"))


def test_unaffected_rows_match_fp32_reference():
    q, k, v, mask = inputs()
    _, w_fix = call(make_eager("zero_masked_rows"), q, k, v, mask)
    q32, k32, v32, m32 = (t.float() for t in (q, k, v, mask))
    _, w_ref = call(make_eager(None), q32, k32, v32, m32)
    torch.testing.assert_close(w_fix[0, 0, REAL_ROWS].float(), w_ref[0, 0, REAL_ROWS], atol=1e-3, rtol=1e-3)


def test_zero_rows_safe_gradients_are_exactly_zero_where_expected():
    """Fully masked query rows get zero weights, and real rows give padding keys zero weight,
    so no gradient should reach the padding rows' queries or the padding positions' keys and values."""
    gq, gk, gv = grads("zero_rows_safe")
    assert (gq[0, 0, PAD_ROWS] == 0).all()
    assert (gk[0, 0, PAD_ROWS] == 0).all()
    assert (gv[0, 0, PAD_ROWS] == 0).all()
    assert (gq[0, 0, REAL_ROWS] != 0).any()      # control: real rows still get gradient

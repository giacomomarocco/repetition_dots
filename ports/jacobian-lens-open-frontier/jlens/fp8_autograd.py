"""Autograd formulas for the finegrained-fp8 kernel ops (activation grads only).

The Triton kernels (`w8a8_block_dynamic_fp8_matmul{,_grouped,_batched}`) register
no autograd formula, so any backward through a natively-loaded FP8 model dies
with "no autograd formula was registered". DeepSeek sidestepped this via
dequantize=True; GLM-5.2 cannot (1.5 TB bf16 > VRAM), so the formulas are
registered here.

Semantics: weights are frozen, so only grad w.r.t. A is produced:

    grad_A = grad_out @ dequant(B)

where dequant expands the [128,128] block scales. This is a STRAIGHT-THROUGH
approximation: the op quantizes activations per-token internally, and that
quantization step is treated as identity in the backward. The Jacobian measured
is therefore of the fp8 model as served, with quantization noise passed through
-- the same class of approximation as fitting on a dequantized model, just on
the other side of the quantizer. Documented, not hidden.

Import AFTER transformers loads the kernel (or call ensure_registered()).
"""

import torch
from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

_REGISTERED = False


def _dequant(B: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    """Blockwise dequant of a [N, K] fp8 weight with [nb, kb] scales -> bf16."""
    N, K = B.shape[-2], B.shape[-1]
    nb, kb = Bs.shape[-2], Bs.shape[-1]
    bm, bn = (N + nb - 1) // nb, (K + kb - 1) // kb
    scale = Bs.to(torch.float32).repeat_interleave(bm, dim=-2)[..., :N, :]
    scale = scale.repeat_interleave(bn, dim=-1)[..., :, :K]
    return (B.to(torch.float32) * scale).to(torch.bfloat16)


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    load_finegrained_fp8_kernel()          # makes the ops exist
    all_ops = torch._C._dispatch_get_all_op_names()

    def ns_of(suffix):
        hits = [n for n in all_ops if n.endswith("::" + suffix)]
        assert len(hits) == 1, (suffix, hits)
        return hits[0]

    # ---- dense: (A[M,K], B[N,K], Bs, block_size, output_dtype) -> [M,N]
    dense = ns_of("w8a8_block_dynamic_fp8_matmul")

    def dense_setup(ctx, inputs, output):
        A, B, Bs, *_ = inputs
        ctx.save_for_backward(B, Bs)
        ctx.a_dtype = A.dtype

    def dense_backward(ctx, grad):
        B, Bs = ctx.saved_tensors
        W = _dequant(B, Bs)                            # [N, K], transient
        gA = (grad.to(torch.bfloat16) @ W).to(ctx.a_dtype)
        return gA, None, None, None, None

    torch.library.register_autograd(dense, dense_backward, setup_context=dense_setup)

    # ---- grouped: (A[T,K] sorted by expert, B[E,N,K], Bs[E,nb,kb],
    #                offsets, tokens_per_expert, block_size, output_dtype)
    grouped = ns_of("w8a8_block_dynamic_fp8_matmul_grouped")

    def grouped_setup(ctx, inputs, output):
        A, B, Bs, offsets, tokens_per_expert, *_ = inputs
        ctx.save_for_backward(B, Bs, tokens_per_expert)
        ctx.a_dtype = A.dtype

    def grouped_backward(ctx, grad):
        B, Bs, tpe = ctx.saved_tensors
        gA = torch.empty(grad.shape[0], B.shape[-1],
                         dtype=torch.bfloat16, device=grad.device)
        start = 0
        for e, cnt in enumerate(tpe.tolist()):
            if cnt == 0:
                continue
            W = _dequant(B[e], Bs[e])
            gA[start:start + cnt] = grad[start:start + cnt].to(torch.bfloat16) @ W
            start += cnt
        return gA.to(ctx.a_dtype), None, None, None, None, None, None

    torch.library.register_autograd(grouped, grouped_backward,
                                    setup_context=grouped_setup)

    # ---- batched: (A[T,K], B[E,N,K], Bs, expert_ids[T], block_size, output_dtype)
    batched = ns_of("w8a8_block_dynamic_fp8_matmul_batched")

    def batched_setup(ctx, inputs, output):
        A, B, Bs, expert_ids, *_ = inputs
        ctx.save_for_backward(B, Bs, expert_ids)
        ctx.a_dtype = A.dtype

    def batched_backward(ctx, grad):
        B, Bs, eids = ctx.saved_tensors
        gA = torch.empty(grad.shape[0], B.shape[-1],
                         dtype=torch.bfloat16, device=grad.device)
        for e in torch.unique(eids).tolist():
            m = eids == e
            W = _dequant(B[e], Bs[e])
            gA[m] = grad[m].to(torch.bfloat16) @ W
        return gA.to(ctx.a_dtype), None, None, None, None, None

    torch.library.register_autograd(batched, batched_backward,
                                    setup_context=batched_setup)

    _REGISTERED = True
    print(f"[jlens.fp8_autograd] registered backward for: {dense}, {grouped}, "
          f"{batched}", flush=True)


ensure_registered()

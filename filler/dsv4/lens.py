"""Logit-lens projection for DeepSeek V4's multi-stream residual states.

DeepSeek V4 does not expose a conventional ``[hidden_size]`` residual at the
end of each block.  Its blocks carry ``hc_mult`` streams.  The native output
path first collapses those streams with the learned mHC head, then applies the
final RMSNorm and vocabulary head.  This module mirrors that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass(frozen=True)
class DeepseekV4LensWeights:
    hc_head_fn: torch.Tensor
    hc_head_base: torch.Tensor
    hc_head_scale: torch.Tensor
    norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    norm_eps: float = 1e-6
    hc_eps: float = 1e-6

    @property
    def hc_mult(self) -> int:
        return self.hc_head_base.numel()

    @property
    def hidden_size(self) -> int:
        return self.norm_weight.numel()


def load_checkpoint_readout(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DeepseekV4LensWeights:
    """Load only the five tensors needed by the lens from a DSV4 checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    shard = checkpoint_dir / "model-00045-of-00048.safetensors"
    if not shard.is_file():
        raise FileNotFoundError(shard)
    with safe_open(shard, framework="pt", device=str(device)) as handle:
        return DeepseekV4LensWeights(
            hc_head_fn=handle.get_tensor("hc_head_fn"),
            hc_head_base=handle.get_tensor("hc_head_base"),
            hc_head_scale=handle.get_tensor("hc_head_scale"),
            norm_weight=handle.get_tensor("norm.weight"),
            lm_head_weight=handle.get_tensor("head.weight"),
        )


def collapse_mhc(hidden_states: torch.Tensor, weights: DeepseekV4LensWeights) -> torch.Tensor:
    """Apply the model's learned mHC output collapse to a layer residual."""
    expected = weights.hc_mult * weights.hidden_size
    if hidden_states.shape[-2:] == (weights.hc_mult, weights.hidden_size):
        streams = hidden_states
    elif hidden_states.shape[-1] == expected:
        streams = hidden_states.reshape(
            *hidden_states.shape[:-1], weights.hc_mult, weights.hidden_size
        )
    else:
        raise ValueError(
            f"Expected trailing shape ({weights.hc_mult}, {weights.hidden_size}) "
            f"or ({expected},), got {tuple(hidden_states.shape)}"
        )

    original_dtype = streams.dtype
    flat = streams.flatten(-2).float()
    rms_inv = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + weights.norm_eps)
    mixes = F.linear(flat, weights.hc_head_fn.float()) * rms_inv
    gates = torch.sigmoid(
        mixes * weights.hc_head_scale.float() + weights.hc_head_base.float()
    ) + weights.hc_eps
    collapsed = (gates.unsqueeze(-1) * streams.float()).sum(dim=-2)
    return collapsed.to(original_dtype)


def project_logits(
    hidden_states: torch.Tensor, weights: DeepseekV4LensWeights
) -> torch.Tensor:
    """Project one or more layer residuals through the native final readout."""
    collapsed = collapse_mhc(hidden_states, weights)
    normalized = collapsed.float() * torch.rsqrt(
        collapsed.float().square().mean(dim=-1, keepdim=True) + weights.norm_eps
    )
    normalized = (normalized * weights.norm_weight.float()).to(collapsed.dtype)
    return F.linear(normalized, weights.lm_head_weight).float()


def project_selected_logits(
    hidden_states: torch.Tensor,
    weights: DeepseekV4LensWeights,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Project residuals only onto selected vocabulary rows.

    This is mathematically identical to selecting columns from
    :func:`project_logits`, but avoids materializing the complete vocabulary
    when only a restricted class such as canonical integer tokens is needed.
    """
    collapsed = collapse_mhc(hidden_states, weights)
    normalized = collapsed.float() * torch.rsqrt(
        collapsed.float().square().mean(dim=-1, keepdim=True) + weights.norm_eps
    )
    normalized = (normalized * weights.norm_weight.float()).to(collapsed.dtype)
    selected = weights.lm_head_weight.index_select(0, token_ids.to(weights.lm_head_weight.device))
    return F.linear(normalized, selected).float()


def project_sglang_logits(
    hidden_states: torch.Tensor, weights: DeepseekV4LensWeights, *, tp_size: int = 4
) -> torch.Tensor:
    """Use SGLang's actual CUDA mHC/RMSNorm kernels and vocabulary partitions.

    The Torch reference can cross a BF16 rounding boundary differently from
    the fused serving kernels. Use this path for comparisons to native output.
    It needs only readout weights and saved residuals, not a loaded model.
    """
    from sglang.srt.layers.mhc_head import fused_hc_head
    from sgl_kernel import rmsnorm

    if hidden_states.shape[-2:] == (weights.hc_mult, weights.hidden_size):
        leading = hidden_states.shape[:-2]
    elif hidden_states.shape[-1] == weights.hc_mult * weights.hidden_size:
        leading = hidden_states.shape[:-1]
    else:
        raise ValueError("unexpected mHC residual shape")
    streams = hidden_states.reshape(-1, weights.hc_mult, weights.hidden_size).contiguous()
    collapsed = fused_hc_head(
        streams, weights.hc_head_fn, weights.hc_head_scale, weights.hc_head_base,
        norm_eps=weights.norm_eps, hc_eps=weights.hc_eps,
    )
    normalized = rmsnorm(collapsed, weights.norm_weight, weights.norm_eps)
    # The serving run uses TP=4 vocabulary sharding. Match the per-rank GEMM
    # shapes as well as the fused normalization/mHC arithmetic.
    # Native generation prunes to one answer row before the head. Keep M=1
    # for each lens row too, avoiding batch-dependent BF16 GEMM rounding.
    shards = weights.lm_head_weight.tensor_split(tp_size, dim=0)
    logits = torch.cat([torch.cat([F.linear(row[None], shard) for shard in shards], dim=-1)
                        for row in normalized], dim=0)
    return logits.reshape(*leading, weights.lm_head_weight.shape[0]).float()


def topk_tokens(
    hidden_states: torch.Tensor, weights: DeepseekV4LensWeights, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(project_logits(hidden_states, weights), k=k, dim=-1)

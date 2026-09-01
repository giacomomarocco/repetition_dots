import torch

from deepseek_v4_logit_lens import (
    DeepseekV4LensWeights,
    collapse_mhc,
    project_logits,
)


def _weights() -> DeepseekV4LensWeights:
    torch.manual_seed(7)
    return DeepseekV4LensWeights(
        hc_head_fn=torch.randn(3, 12),
        hc_head_base=torch.randn(3),
        hc_head_scale=torch.randn(1),
        norm_weight=torch.randn(4),
        lm_head_weight=torch.randn(11, 4),
    )


def test_flat_and_stream_inputs_are_equivalent():
    weights = _weights()
    states = torch.randn(2, 3, 4)
    torch.testing.assert_close(
        collapse_mhc(states, weights), collapse_mhc(states.flatten(-2), weights)
    )


def test_projection_matches_explicit_native_readout():
    weights = _weights()
    states = torch.randn(2, 3, 4)
    collapsed = collapse_mhc(states, weights)
    normalized = collapsed.float() * torch.rsqrt(
        collapsed.float().square().mean(-1, keepdim=True) + weights.norm_eps
    )
    normalized = normalized * weights.norm_weight.float()
    expected = normalized @ weights.lm_head_weight.float().T
    torch.testing.assert_close(project_logits(states, weights), expected)

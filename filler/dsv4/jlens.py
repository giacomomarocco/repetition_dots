"""Forward-only readout of rectangular and stream-mean DeepSeek V4 Jacobians."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from filler.dsv4.lens import DeepseekV4LensWeights

SOURCE_LAYERS = tuple(range(19, 40))
WORKSPACE_SOURCE_LAYERS = tuple(range(42))


@dataclass(frozen=True)
class JLens:
    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int
    d_source: int
    stream_reduction: str = "flatten"
    hc_mult: int = 4
    target_layer: int | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.jacobians))


def load_jlens(
    path: str | Path, *, expected_shape: tuple[int, int] = (4096, 16384),
    expected_layers: tuple[int, ...] = SOURCE_LAYERS, expected_n_prompts: int = 1000,
    validate_values: bool = True,
) -> JLens:
    """Load tensor-only data, retaining fp16 mapped storage until each layer is used.

    Full matrix validation belongs on a compute node for the released checkpoint.
    Shape overrides support small fixtures; production defaults identify the 0731 fit.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"J", "n_prompts", "source_layers", "d_model", "d_source"}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError("not a rectangular JacobianLens checkpoint")
    if (saved["d_model"], saved["d_source"]) != expected_shape:
        raise ValueError("Jacobian dimensions do not match the requested model")
    matrices = saved["J"]
    if (not isinstance(matrices, dict) or any(type(key) is not int for key in matrices)
            or tuple(sorted(matrices)) != expected_layers
            or tuple(saved["source_layers"]) != expected_layers):
        raise ValueError("Jacobian layer keys and source_layers must match the fitted band")
    if type(saved["n_prompts"]) is not int or saved["n_prompts"] != expected_n_prompts:
        raise ValueError("unexpected fitting prompt count")
    for layer, matrix in matrices.items():
        if (not isinstance(matrix, torch.Tensor) or tuple(matrix.shape) != expected_shape
                or matrix.dtype != torch.float16):
            raise ValueError(f"L{layer}: expected an fp16 matrix of shape {expected_shape}")
        if validate_values and (not bool(torch.isfinite(matrix).all()) or not bool(torch.any(matrix != 0))):
            raise ValueError(f"L{layer}: matrix is nonfinite or identically zero")
    return JLens(matrices, saved["n_prompts"], *expected_shape)


def load_workspace_jlens(
    path: str | Path, *, expected_d_model: int = 4096,
    expected_layers: tuple[int, ...] = WORKSPACE_SOURCE_LAYERS,
    validate_values: bool = True,
) -> JLens:
    """Load camilablank/workspace-lenses' square DeepSeek V4 J-Lens.

    The release stores ``J`` as a dictionary keyed by post-block layer, including
    an identity at the penultimate block (41). ``skip_first=4`` applies to fitting
    token positions; layers 0--3 are present and usable.
    ``validate_values=False`` is for metadata preflight only: mmap keeps tensor
    pages unread. Full validation and scoring belong on compute nodes.
    """
    if (type(expected_d_model) is not int or expected_d_model < 1 or not expected_layers
            or any(type(layer) is not int or layer < 0 for layer in expected_layers)
            or tuple(sorted(set(expected_layers))) != expected_layers):
        raise ValueError("expected dimensions/layers must be positive and ordered")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"J", "n_prompts", "source_layers", "d_model", "provenance"}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError("not a workspace square J-Lens checkpoint")
    if (type(saved["d_model"]) is not int or saved["d_model"] != expected_d_model
            or saved.get("d_source", expected_d_model) != expected_d_model):
        raise ValueError("square Jacobian dimensions do not match the requested model")
    matrices, layers = saved["J"], saved["source_layers"]
    if (not isinstance(matrices, dict) or any(type(key) is not int for key in matrices)
            or tuple(sorted(matrices)) != expected_layers
            or not isinstance(layers, (list, tuple))
            or any(type(layer) is not int for layer in layers)
            or tuple(layers) != expected_layers):
        raise ValueError("Jacobian layer keys and source_layers must match the fitted band")
    if type(saved["n_prompts"]) is not int or saved["n_prompts"] != 25:
        raise ValueError("unexpected fitting prompt count; expected 25")
    provenance = saved["provenance"]
    expected_metadata = {"model_id": "deepseek-ai/DeepSeek-V4-Flash",
                         "target_layer": expected_layers[-1], "n_prompts": 25,
                         "skip_first": 4, "dataset_id": "NeelNanda/pile-10k"}
    if not isinstance(provenance, dict) or any(
        type(provenance.get(key)) is not type(value) or provenance[key] != value
        for key, value in expected_metadata.items()
    ):
        raise ValueError("workspace lens provenance does not match the requested fit")
    # The repository also contains an identically shaped R-Lens.
    try:
        config = json.loads(provenance["config_json"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing or invalid workspace estimator config") from error
    if not isinstance(config, dict) or config.get("estimator") != "standard" or config.get("arm") != "std":
        raise ValueError("expected the standard J-Lens estimator, not an R-Lens")
    shape = (expected_d_model, expected_d_model)
    for layer, matrix in matrices.items():
        if (not isinstance(matrix, torch.Tensor) or tuple(matrix.shape) != shape
                or matrix.dtype != torch.float16):
            raise ValueError(f"L{layer}: expected an fp16 matrix of shape {shape}")
        if validate_values and (not bool(torch.isfinite(matrix).all())
                                or not bool(torch.any(matrix != 0))):
            raise ValueError(f"L{layer}: matrix is nonfinite or identically zero")
    if validate_values and not torch.equal(
        matrices[expected_layers[-1]], torch.eye(expected_d_model, dtype=torch.float16)
    ):
        raise ValueError("workspace target-layer anchor must be exactly identity")
    return JLens(matrices, saved["n_prompts"], expected_d_model, expected_d_model,
                 stream_reduction="mean", target_layer=expected_layers[-1],
                 provenance=dict(provenance))


def mean_mhc_streams(states: torch.Tensor, *, hidden_size: int, hc_mult: int = 4) -> torch.Tensor:
    """Average the stream axis, retaining every batch/token dimension.

    Accept complete post-block ``[..., 4, D]`` or flattened ``[..., 4*D]``
    captures. Already averaged inputs are deliberately not inferred by shape.
    Convert before reduction so BF16 captures do not round the mean to BF16.
    """
    if states.ndim == 0 or not states.is_floating_point():
        raise ValueError("residual must be a floating point tensor")
    if states.ndim >= 2 and tuple(states.shape[-2:]) == (hc_mult, hidden_size):
        streams = states
    elif states.shape[-1] == hc_mult * hidden_size:
        streams = states.reshape(*states.shape[:-1], hc_mult, hidden_size)
    else:
        raise ValueError(f"residual must end in ({hc_mult}, {hidden_size}) or ({hc_mult * hidden_size},)")
    if not bool(torch.isfinite(streams).all()):
        raise ValueError("nonfinite residual")
    result = streams.float().mean(dim=-2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite stream mean")
    return result


def transport(states: torch.Tensor, lens: JLens, layer: int) -> torch.Tensor:
    """Apply ``J @ h`` using the reduction selected by the checkpoint loader.

    Rectangular 0731 lenses flatten complete streams; workspace square lenses
    first average four streams. Neither path applies the learned mHC head.
    """
    if layer not in lens.jacobians:
        raise ValueError(f"L{layer} has no fitted Jacobian; available layers: {lens.source_layers}")
    if states.ndim == 0 or not states.is_floating_point():
        raise ValueError("residual must be a floating point tensor")
    if lens.stream_reduction == "mean":
        if lens.d_source != lens.d_model:
            raise ValueError("mean-stream transport requires a square Jacobian")
        states = mean_mhc_streams(states, hidden_size=lens.d_model, hc_mult=lens.hc_mult)
    elif lens.stream_reduction == "flatten":
        streams = lens.d_source // lens.d_model
        if states.ndim >= 2 and tuple(states.shape[-2:]) == (streams, lens.d_model):
            states = states.flatten(-2)
        elif states.shape[-1] != lens.d_source:
            raise ValueError(f"residual must end in ({streams}, {lens.d_model}) or ({lens.d_source},)")
    else:
        raise ValueError(f"unknown stream reduction: {lens.stream_reduction}")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("nonfinite residual")
    matrix = lens.jacobians[layer].to(device=states.device, dtype=torch.float32)
    result = F.linear(states.float(), matrix)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite transported residual")
    return result


def unembed_transported(residual: torch.Tensor, weights: DeepseekV4LensWeights) -> torch.Tensor:
    """Published HF readout: cast to head dtype, RMSNorm, then vocabulary head.

    The input is already collapsed, including when a batch dimension equals four.
    HF DeepseekV4RMSNorm rounds the normalized vector *before* multiplying the
    norm weight; the existing SGLang lens follows its fused native rounding instead.
    """
    if residual.ndim == 0 or residual.shape[-1] != weights.hidden_size:
        raise ValueError("transported residual has the wrong target width")
    head = weights.lm_head_weight
    value = residual.to(device=head.device, dtype=head.dtype)
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(-1, keepdim=True) + weights.norm_eps
    )
    normalized = normalized.to(head.dtype) * weights.norm_weight.to(device=head.device, dtype=head.dtype)
    return F.linear(normalized, head).float()


def project_jlens_logits(
    states: torch.Tensor, lens: JLens, layer: int, weights: DeepseekV4LensWeights,
) -> torch.Tensor:
    source_width = weights.hidden_size * (1 if lens.stream_reduction == "mean" else weights.hc_mult)
    if (lens.d_model != weights.hidden_size or lens.d_source != source_width
            or (lens.stream_reduction == "mean" and lens.hc_mult != weights.hc_mult)):
        raise ValueError("lens and checkpoint readout dimensions disagree")
    return unembed_transported(transport(states, lens, layer), weights)


def published_readout_fixture(weights: DeepseekV4LensWeights):
    """Use the author's adapter and HF's actual norm without constructing a model.

    This is an independent readout oracle, never an inference server. Its collapse
    module deliberately fails if the adapter attempts to collapse J-transported data.
    """
    from jlens.hf import DeepseekV4LensModel
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm

    class NoCollapse(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("a transported residual must not be collapsed again")

    adapter = object.__new__(DeepseekV4LensModel)
    adapter.hc_mult = weights.hc_mult
    adapter.d_model = weights.hidden_size
    adapter._logit_softcap = None
    adapter.target_module = NoCollapse()
    adapter._final_norm = DeepseekV4RMSNorm(weights.hidden_size, eps=weights.norm_eps)
    adapter._final_norm.weight = torch.nn.Parameter(weights.norm_weight, requires_grad=False)
    # Linear on meta avoids allocating/initializing a second full vocabulary head.
    adapter._lm_head = torch.nn.Linear(weights.hidden_size, weights.lm_head_weight.shape[0],
                                       bias=False, device="meta")
    adapter._lm_head.weight = torch.nn.Parameter(weights.lm_head_weight, requires_grad=False)
    return adapter

# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace adapter.

Wraps an already-loaded HF model as a :class:`~jlens.protocol.LensModel` so
the rest of the package stays model-library-agnostic. Model loading
(``from_pretrained``, device placement, dtype) stays the caller's job;
:func:`from_hf` only locates the residual stack inside whatever it's handed.

Any model library can be plugged in the same way: implement the
:class:`~jlens.protocol.LensModel` members directly (``tests/tiny.py`` is a
minimal example) and the rest of the package works unchanged.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve_attr_path(obj: Any, dotted_path: str) -> Any:
    return functools.reduce(getattr, dotted_path.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Where the lens-relevant submodules live inside a HuggingFace model.

    Attributes:
        path: Dotted attribute path from the ``*ForCausalLM`` to the bare text
            decoder (the module to call for a hooks-visible forward pass).
        layers: Attribute name on the text decoder for the residual blocks.
        norm: Attribute name for the final pre-unembed norm.
        embed: Attribute name for the input token embedding.
        lm_head: Attribute name on the ``*ForCausalLM`` for the unembedding.
    """

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


#: Known layouts, tried in order. The first whose ``path`` resolves and whose
#: text decoder has all three of ``layers``/``norm``/``embed`` wins. Covers
#: Llama / Qwen / Mistral / Gemma / OLMo / StableLM (the modern HF default),
#: their multimodal-wrapper variants, plus Phi, GPT-2, and GPT-NeoX.
_LAYOUTS: tuple[Layout, ...] = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),  # Phi
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
    Layout(
        "gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"
    ),  # Pythia
)


def _find_layout(hf_model: nn.Module) -> Layout:
    """Locate the text decoder inside an HF ``*ForCausalLM`` /
    ``*ForConditionalGeneration`` by trying :data:`_LAYOUTS` in order."""
    for layout in _LAYOUTS:
        try:
            candidate = _resolve_attr_path(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(candidate, a) for a in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(
        f"could not locate the text decoder inside {type(hf_model).__name__} "
        f"(tried {len(_LAYOUTS)} known layouts); pass layout= explicitly"
    )


class HFLensModel:
    """:class:`~jlens.protocol.LensModel` over a loaded HuggingFace model.

    Holds references into the caller's model; nothing is copied. The
    constructor mutates that model in place: every parameter gets
    ``requires_grad_(False)`` (the Jacobian fit needs grads only with respect
    to activations), ``compile=True`` replaces each block with a
    :func:`torch.compile` wrapper, and ``force_bos`` may set
    ``tokenizer.add_bos_token``. Pass a model you don't otherwise need.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        compile: bool = False,
        force_bos: bool = True,
    ) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        if (
            force_bos
            and getattr(tokenizer, "bos_token_id", None) is not None
            and hasattr(tokenizer, "add_bos_token")
        ):
            tokenizer.add_bos_token = True

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        if layout is None:
            layout = _find_layout(hf_model)
        self.layout = layout
        self._text_module = _resolve_attr_path(hf_model, layout.path)
        self.layers: nn.ModuleList = getattr(self._text_module, layout.layers)
        self._final_norm: nn.Module = getattr(self._text_module, layout.norm)
        self._embed_tokens: nn.Module = getattr(self._text_module, layout.embed)
        self._lm_head: nn.Module = getattr(hf_model, layout.lm_head)

        text_config = hf_model.config.get_text_config()
        self.n_layers: int = text_config.num_hidden_layers
        self.d_model: int = text_config.hidden_size
        self._logit_softcap: float | None = getattr(
            text_config, "final_logit_softcapping", None
        )
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self.layers)} blocks at {layout.path}.{layout.layers}"
            )

        # Per-layer compile: each block stays a hook boundary, so
        # ActivationRecorder still fires and the retained graph is bounded per
        # block. Whole-module compile would inline the blocks and bypass the
        # hooks.
        if compile:
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(
                    self.layers[i], mode="default", dynamic=False
                )

    def __repr__(self) -> str:
        return (
            f"HFLensModel({type(self._hf_model).__name__}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._text_module(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        logits = self._lm_head(
            self._final_norm(residual.to(target_dtype).to(target_device))
        )
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits


class DeepseekV4LensModel(HFLensModel):
    """:class:`HFLensModel` for DeepSeek-V4's hyper-connection residual.

    V4 carries ``config.hc_mult`` parallel residual streams: every block takes
    and returns ``[batch, seq, hc_mult, hidden_size]``, and only the final
    ``model.hc_head`` collapses them back to one vector per position. A square
    ``J_l`` is therefore not well-defined, so the lens is fitted rectangular:

    * source = a block's output, flattened to ``d_source = hc_mult *
      hidden_size``, so all four streams are read losslessly;
    * target = ``hc_head``'s output at ``d_target = hidden_size``, the last
      point where the residual is a single vector and the input the final norm
      + LM head already expects.

    Targeting ``hc_head`` rather than the flattened final block keeps the
    backward count at ``d_target / dim_batch`` — a quarter of the fully
    expanded alternative, at no cost in source-side information.

    Attributes:
        hc_mult: Number of parallel residual streams.
        target_module: ``model.hc_head``, addressed by the virtual layer index
            ``n_layers``; see :class:`~jlens.protocol.LensModel`.
    """

    def __init__(self, hf_model: nn.Module, tokenizer: Any, **kwargs: Any) -> None:
        super().__init__(hf_model, tokenizer, **kwargs)
        text_config = hf_model.config.get_text_config()
        self.hc_mult: int = text_config.hc_mult
        self.d_source: int = self.hc_mult * self.d_model
        self.d_target: int = self.d_model
        self.target_module: nn.Module = self._text_module.hc_head

    def __repr__(self) -> str:
        return (
            f"DeepseekV4LensModel({type(self._hf_model).__name__}, "
            f"n_layers={self.n_layers}, d_model={self.d_model}, "
            f"hc_mult={self.hc_mult})"
        )

    def unembed(
        self, residual: torch.Tensor, *, collapse: bool | None = None
    ) -> torch.Tensor:
        """Final norm + LM head, collapsing the hyper-connection streams first
        if the residual still carries them.

        Two kinds of tensor reach here and both must work, because
        :meth:`jlens.lens.JacobianLens.apply` and :mod:`jlens.vis` call this on
        each without knowing the difference:

        * ``[..., hc_mult, hidden_size]`` — a raw block output, either the final
          layer (whose readout must reproduce the model's own logits) or an
          intermediate one with ``use_jacobian=False``. Running it through
          ``hc_head`` is what makes the latter the natural mHC generalisation of
          the vanilla logit lens: read an early residual with the *final*
          collapse operator.
        * ``[..., hidden_size]`` — already transported by ``J_l``, which
          absorbed the collapse into its target basis during the fit.

        Args:
            residual: The tensor to decode.
            collapse: Force the ``hc_head`` path on or off. ``None``
                (the default) sniffs it from the trailing two axes. The sniff is
                unambiguous for the shapes the package itself produces, but a
                caller passing a batched ``[batch, seq, hidden_size]`` tensor
                whose ``seq`` happens to equal ``hc_mult`` must say so here.
        """
        if collapse is None:
            collapse = residual.dim() >= 3 and tuple(residual.shape[-2:]) == (
                self.hc_mult,
                self.d_model,
            )
        if collapse:
            # hc_head indexes the stream axis positionally (`flatten(2)` /
            # `sum(dim=2)`), so it only accepts a 4-D [batch, seq, hc_mult, d] —
            # while apply() and vis hand us [n_positions, hc_mult, d]. It is
            # position-wise (an RMS norm over d, a linear, a weighted sum over
            # the streams), so folding every leading axis into one batch of
            # length 1 is exact, not an approximation. hc_head is held in fp32
            # (`_keep_in_fp32_modules_strict`) and casts internally, so only the
            # device has to match.
            leading = residual.shape[:-2]
            streams = residual.reshape(1, -1, self.hc_mult, self.d_model)
            collapsed_streams = self.target_module(
                streams.to(self.target_module.hc_fn.device)
            )
            residual = collapsed_streams.reshape(*leading, self.d_model)
        return super().unembed(residual)


def from_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    layout: Layout | None = None,
    text_module: str | None = None,
    compile: bool = False,
    force_bos: bool = True,
) -> HFLensModel:
    """Wrap a loaded HuggingFace model as a :class:`~jlens.protocol.LensModel`.

    Args:
        hf_model: A loaded ``*ForCausalLM`` (or ``*ForConditionalGeneration``),
            already on the target device and dtype.
        tokenizer: The matching HF tokenizer.
        layout: Where the residual blocks / final norm / embedding / LM head
            live inside ``hf_model``. Auto-detected for the common HF families;
            pass explicitly only for unusual layouts.
        text_module: Deprecated alias for ``layout=Layout(path=text_module)``.
        compile: Wrap each residual block in :func:`torch.compile`. Faster
            backward in :func:`jlens.fitting.fit` after a one-time compilation
            cost. Do not combine with ``device_map="auto"``.
        force_bos: Some instruction-tuned checkpoints ship with
            ``add_bos_token=False``; raw-text prompts are degraded without an
            attention-sink BOS, so this sets it ``True`` by default. The
            attribute may have no effect for some fast-tokenizer
            configurations.
    """
    if text_module is not None:
        if layout is not None:
            raise TypeError("pass at most one of layout= / text_module=")
        layout = Layout(path=text_module)
    if layout is None:
        layout = _find_layout(hf_model)
    return _lens_model_class(hf_model, layout)(
        hf_model, tokenizer, layout=layout, compile=compile, force_bos=force_bos
    )


def _lens_model_class(hf_model: nn.Module, layout: Layout) -> type[HFLensModel]:
    """Pick the adapter for ``hf_model``: the multi-stream subclass when the
    text decoder carries several residual streams per position, otherwise the
    plain single-stream one."""
    text_config = hf_model.config.get_text_config()
    text_module = _resolve_attr_path(hf_model, layout.path)
    if getattr(text_config, "hc_mult", 1) > 1 and hasattr(text_module, "hc_head"):
        return DeepseekV4LensModel
    return HFLensModel

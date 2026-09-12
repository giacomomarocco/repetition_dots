# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""The model interface the lens is typed against.

Any model can be plugged in by implementing these members.
:func:`jlens.hf.from_hf` is the HuggingFace adapter; ``tests/tiny.py`` is a
minimal from-scratch example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """What the lens needs from a model.

    Attributes:
        n_layers: Number of residual blocks.
        d_model: Residual-stream width.
        layers: The residual blocks, indexable by integer; what
            :class:`~jlens.hooks.ActivationRecorder` hooks.
        tokenizer: Tokenizer used by the visualisation helpers; must provide
            ``decode(token_ids) -> str``. Fitting and :meth:`apply` never
            touch it.

    Optional attributes (read through :func:`d_source_of` /
    :func:`d_target_of` / :func:`target_module_of`, so implementations that
    do not define them keep the square single-stream behaviour):

        d_source: Flattened width of a *source* residual, when a block's
            output is not a plain ``[batch, seq, d_model]`` tensor. DeepSeek-V4
            carries ``hc_mult`` parallel streams, so its blocks emit
            ``[batch, seq, hc_mult, d_model]`` and ``d_source`` is
            ``hc_mult * d_model``. Defaults to ``d_model``.
        d_target: Width of the *target* activation the Jacobian maps into.
            Defaults to ``d_model``. Together these make ``J_l`` a
            ``[d_target, d_source]`` matrix; when both equal ``d_model`` the
            estimator, the checkpoint format and
            :meth:`~jlens.lens.JacobianLens.transport` behave exactly as they
            did before.
        target_module: A module *outside* the block stack to take the Jacobian
            with respect to, addressed by the virtual layer index
            ``n_layers``. DeepSeek-V4 sets this to ``model.hc_head``, the
            operator that collapses the four streams back to one
            ``[batch, seq, d_model]`` tensor — the last point in the network
            where the residual is still a single vector per position, and the
            input :meth:`unembed` expects. ``None`` (the default) means the
            target must be one of :attr:`layers`.
    """

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]`` on the
        model's input device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack on ``input_ids`` (no LM head). Must build an
        autograd graph through :attr:`layers` when grad is enabled, and must be
        deterministic across batch elements (eval mode, dropout off) — the
        fitting estimator replicates the prompt along the batch axis."""
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a residual-stream tensor ``[..., d_model]`` to logits
        ``[..., vocab_size]`` (final norm + LM head).

        Multi-stream models are also handed *uncollapsed* residuals
        ``[..., d_source // d_model, d_model]`` (straight off a block, with no
        ``J_l`` transport) and are expected to collapse them first; see
        :class:`jlens.hf.DeepseekV4LensModel`."""
        ...


def d_source_of(model: LensModel) -> int:
    """Flattened source width of ``model``; ``d_model`` unless overridden."""
    return getattr(model, "d_source", model.d_model)


def d_target_of(model: LensModel) -> int:
    """Target width of ``model``; ``d_model`` unless overridden."""
    return getattr(model, "d_target", model.d_model)


def target_module_of(model: LensModel) -> nn.Module | None:
    """The out-of-stack Jacobian target of ``model``, or ``None``."""
    return getattr(model, "target_module", None)

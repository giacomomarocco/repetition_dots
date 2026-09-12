# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:func:`jacobian_for_prompt`): for each output dimension, inject a
one-hot cotangent at *every valid target position at once* and backprop. The
gradient at source position ``p`` is then ``sum_{p' >= p} dh_final[p'] / dh_l[p]``,
the sum over later target positions; we take the mean over source positions
``p``. This is the reduction used in the paper. A per-position estimator
(``dh_final[p] / dh_l[p]`` averaged over ``p``) gives a slightly different
``J_l``; both work as a lens.

Cost: one forward pass and ``ceil(d_target / dim_batch)`` backward passes per
prompt. Shard across machines by running :func:`fit` on disjoint prompt
slices and merging with :meth:`jlens.lens.JacobianLens.merge`.

``J_l`` is ``[d_target, d_source]``. Both default to ``d_model`` — the square
single-residual-stream case — but a model may widen the source (a block that
emits several parallel residual streams, flattened here) and/or move the target
out of the block stack; see :class:`jlens.protocol.LensModel`. Only the *target*
width sets the backward count, so a multi-stream model costs no more to fit
than a single-stream one of the same width.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel, d_source_of, d_target_of, target_module_of

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short to
            leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def _check_layer_indices(
    source_layers: Sequence[int] | None,
    target_layer: int | None,
    n_layers: int,
    *,
    allow_virtual_target: bool = False,
) -> tuple[list[int], int]:
    """Resolve None/negative layer indices, bounds-check, enforce source < target.

    ``allow_virtual_target`` admits the one extra index ``n_layers``, which
    addresses a model's out-of-stack ``target_module`` rather than a block, and
    makes it the default target — every block is then a valid source. Negative
    indices still count from the end of the *block stack*, so ``-1`` remains the
    last block whether or not a virtual target exists.
    """
    max_target = n_layers if allow_virtual_target else n_layers - 1
    if target_layer is None:
        target = max_target
    else:
        target = target_layer
        if target < 0:
            target += n_layers
    if not 0 <= target <= max_target:
        raise ValueError(
            f"target_layer={target_layer} out of range for {n_layers} layers"
        )
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    position_grads: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_target / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position. See the
    module docstring for the resulting estimator and how it relates to
    a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer, or to the virtual index ``n_layers`` (the model's
            :attr:`~jlens.protocol.LensModel.target_module`) when it has one;
            negative indices count from the end of the block stack. In some
            cases, targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.
        position_grads: Debug hook for validating the estimator against finite
            differences. If a dict is passed it is filled with the raw
            per-source-position gradients, *before* the mean over positions, as
            ``{layer: Tensor[d_target, seq_len, *source_shape]}`` on the CPU.
            This is ``d_target`` times the size of one backward's gradients —
            fine on a toy model, ruinous on a real one — so leave it ``None``
            outside tests.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_target, d_source]`` fp32 CPU tensor (both are
        ``d_model`` unless the model overrides them).

    Raises:
        ValueError: If the target activation is not ``[batch, seq, d_target]``.
            The one-hot cotangent is built with three indices, so a target with
            extra axes would be indexed wrong — silently, hence the check.
    """
    n_layers = model.n_layers
    d_source, d_target = d_source_of(model), d_target_of(model)
    target_module = target_module_of(model)
    source_layers, target_layer = _check_layer_indices(
        source_layers,
        target_layer,
        n_layers,
        allow_virtual_target=target_module is not None,
    )
    # The virtual index resolves to target_module; every other index is a block.
    extra_modules = (
        {n_layers: target_module} if target_layer == n_layers else None
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_target, d_source, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_target / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
            extra_modules=extra_modules,
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[
            target_layer
        ]  # [dim_batch, seq_len, d_target]
        if target_activation.dim() != 3 or target_activation.shape[-1] != d_target:
            raise ValueError(
                f"target activation at layer {target_layer} has shape "
                f"{tuple(target_activation.shape)}, expected "
                f"[{dim_batch}, {seq_len}, {d_target}]; the one-hot cotangent "
                "below indexes it with exactly three indices"
            )
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)
        if position_grads is not None:
            for layer, activation in zip(source_layers, source_activations, strict=True):
                position_grads[layer] = torch.zeros(
                    d_target, *activation.shape[1:], dtype=torch.float32
                )

        for pass_idx, dim_start in enumerate(range(0, d_target, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_target - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every valid target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, *source_shape] on whatever device
                # this layer lives on. flatten(start_dim=2) folds any residual-
                # stream axes into the source width (a no-op for the usual 3-D
                # [dim_batch, seq_len, d_model] grad, and the same row-major
                # (stream, dim) order JacobianLens.transport flattens with);
                # the mean over valid positions then gives dim_batch rows.
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                selected = grad[:n_dims_this_pass, positions_on_device].float()
                rows = selected.flatten(start_dim=2).mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
                if position_grads is not None:
                    position_grads[layer][dim_start : dim_start + n_dims_this_pass] = (
                        grad[:n_dims_this_pass].float().cpu()
                    )
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    normalize_per_prompt: bool = False,
) -> JacobianLens:
    """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_target * d_source
            * 4`` bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.
        normalize_per_prompt: Scale each prompt's ``J_l`` to unit Frobenius norm
            before accumulating, so every prompt contributes equally to the
            averaged *direction*.

            Off by default — the plain mean is the paper's estimator and what
            the released lenses use. Turn it on when the per-prompt Jacobian
            norm is heavy-tailed, which happens when transport spans many
            blocks: a per-block gain of 1.19 versus 1.06 compounds over 40
            layers into a 150× difference, so a plain mean ends up dominated by
            a handful of high-gain prompts. Measured on DeepSeek-V4-Flash-0731 at
            S=128, ``max||J||/sqrt(d)`` ranges over 8–1346 across WikiText
            prompts.

            This discards nothing observable: :meth:`~jlens.lens.JacobianLens.transport`
            feeds ``unembed``, which applies the final RMS norm before the LM
            head, so *uniform rescaling of* ``J_l`` *leaves the logits
            unchanged*. The prompt's Jacobian norm is therefore a weighting the
            readout cannot see, and weighting by it is arbitrary rather than
            principled.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    n_layers = model.n_layers
    d_source, d_target = d_source_of(model), d_target_of(model)
    source_layers, target_layer = _check_layer_indices(
        source_layers,
        target_layer,
        n_layers,
        allow_virtual_target=target_module_of(model) is not None,
    )

    logger.info(
        "fit: n_layers=%d d_target=%d d_source=%d, fitting %d source layers "
        "(target=L%d) on %d prompts",
        n_layers,
        d_target,
        d_source,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
            ("d_source", d_source),
            ("d_target", d_target),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_target, d_source, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                    "d_source": d_source,
                    "d_target": d_target,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_target)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        # Measured before any normalisation, so the diagnostic still reports the
        # raw gain even when normalize_per_prompt makes it 1 by construction.
        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d

        if normalize_per_prompt:
            for layer in source_layers:
                norm = per_prompt_J[layer].norm()
                if norm > 0:
                    per_prompt_J[layer] /= norm
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(
        jacobians=jacobian_mean,
        n_prompts=n_done,
        d_model=d_target,
        d_source=d_source,
    )

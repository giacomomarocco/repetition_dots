# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Causal interventions on the lens subspace.

Reading a lens says what a residual is *disposed* to make the model say. It
does not show that the disposition is load-bearing. That needs an edit: change
the lens coordinate, and see whether the model's output follows. Everything
here writes to the residual stream through forward hooks; nothing here fits or
reads a lens (see :mod:`jlens.fitting` and :mod:`jlens.lens` for those).

Three primitives, matching the protocols in ``data/experiments/README.md``:

* :func:`swap` — replace one token's direction with another's. The workhorse:
  if swapping an unspoken intermediate flips the answer, that intermediate was
  being used.
* :func:`steer` — add a token's direction outright, for concept injection.
* :func:`ablate` — project out the top-k directions, to ask what breaks.

**The lens direction.** Token ``t``'s readout vector at layer ``l`` is row
``t`` of ``W_U @ J_l``, i.e. ``J_l.T @ W_U[t]``, a vector in the *source* space
the lens reads from. Its coordinate in a residual ``h`` is ``<h, d_t>``. This
is the "lens coordinate" the experiment protocols refer to.

**Multi-stream models.** On DeepSeek-V4 a block emits
``[batch, seq, hc_mult, d_model]`` while a direction lives in the flattened
``d_source = hc_mult * d_model`` space. The hooks flatten the trailing stream
axis, edit, and restore the original shape, so the same code path serves both
single- and multi-stream models. This reshape is the one genuinely
mHC-specific step, and it is why :func:`swap` has its own tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "Intervention",
    "ablate",
    "contrastive_direction",
    "lens_directions",
    "steer",
    "swap",
]


def _readout_weight(model: LensModel) -> torch.Tensor:
    """The unembedding matrix ``[vocab, d_target]``.

    Not part of :class:`~jlens.protocol.LensModel` — that Protocol only
    promises :meth:`~jlens.protocol.LensModel.unembed`, which is a function,
    not a matrix. Interventions need the matrix itself to build a token's
    direction, so this looks for it under the names the two in-tree
    implementations use, and asks for an explicit ``unembed_weight`` otherwise.
    """
    for attr in ("unembed_weight", "lm_head", "_lm_head"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        weight = getattr(obj, "weight", obj)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return weight
    raise AttributeError(
        f"{type(model).__name__} exposes no unembedding matrix; set "
        "`model.unembed_weight = <Tensor[vocab, d_target]>` to use jlens.intervene."
    )


def lens_directions(
    lens: JacobianLens,
    model: LensModel,
    layer: int,
    token_ids: Sequence[int],
    *,
    center: bool = True,
) -> torch.Tensor:
    """Unit-normalised lens directions for ``token_ids`` at ``layer``.

    Args:
        lens: The fitted lens.
        model: Supplies the unembedding matrix.
        layer: Source layer; must be one of ``lens.source_layers``.
        token_ids: Vocabulary indices.
        center: Subtract the vocabulary-mean unembedding row before
            transporting. Logits are only meaningful up to a constant shift, so
            the component of ``W_U[t]`` shared by the whole vocabulary raises
            every token equally and cannot discriminate. Leaving it in makes
            every token's direction partly the same generic "some token here"
            direction, which is what makes an uncentred swap read as a generic
            perturbation at high strength instead of a substitution.

    Returns:
        ``[len(token_ids), lens.d_source]``, each row unit-norm. Rows for
        tokens whose direction is degenerate (zero norm) are left at zero, so
        they contribute nothing rather than producing NaNs.
    """
    if layer not in lens.jacobians:
        raise KeyError(
            f"layer {layer} not in lens (has {lens.source_layers[0]}.."
            f"{lens.source_layers[-1]})"
        )
    J = lens.jacobians[layer]                                   # [d_target, d_source]
    W = _readout_weight(model)                                  # [vocab, d_target]
    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=W.device)
    rows = W[ids].to(J.device, J.dtype)                         # [n, d_target]
    if center:
        rows = rows - W.mean(0).to(J.device, J.dtype)
    D = rows @ J                                                # [n, d_source]
    norm = D.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, D / norm.clamp_min(1e-12), torch.zeros_like(D))


def contrastive_direction(
    lens: JacobianLens,
    model: LensModel,
    layer: int,
    from_token: int,
    to_token: int,
) -> torch.Tensor:
    """Unit direction that trades ``from_token``'s logit for ``to_token``'s.

    Formed by differencing in the *target* basis and transporting once —
    ``normalize(J_l.T @ (W_U[to] - W_U[from]))`` — rather than transporting each
    token separately and subtracting two already-normalised vectors. The two are
    not the same: independent normalisation discards the tokens' relative
    magnitudes, so the difference no longer points along the logit contrast that
    actually decides between them.

    No ``center`` argument: the vocabulary mean cancels in the difference, so
    centring is a no-op here by construction.
    """
    J = lens.jacobians[layer]
    W = _readout_weight(model)
    w = (W[to_token] - W[from_token]).to(J.device, J.dtype)
    d = w @ J
    return d / d.norm().clamp_min(1e-12)


def _coordinate_scale(flat: torch.Tensor) -> torch.Tensor:
    """Typical magnitude of a *single* coordinate of ``flat``.

    ``flat.norm()`` is the length of the whole ``d_source``-dimensional vector,
    which is about ``sqrt(d_source)`` times larger than any one coordinate of
    it. Scaling a unit direction by the full norm therefore injects a vector
    comparable to the entire residual — on DeepSeek-V4 that is a factor of
    ``sqrt(16384) = 128`` too much, compounded over every band layer, which
    destroys the forward pass rather than editing it. (Measured: it took the
    multi-hop swap from 0.434 to 0.000, with every single item landing on
    neither the original nor the swapped answer.)

    Dividing by ``sqrt(d_source)`` puts the injected amount on the same footing
    as the coordinate it replaces, which is what makes ``alpha=1`` mean "as
    strongly present as a typical concept" rather than "as large as everything".
    """
    return flat.norm(dim=-1).mean() / (flat.shape[-1] ** 0.5)


class Intervention:
    """Context manager that edits block outputs on the forward pass.

    Args:
        model: The model whose ``layers`` are hooked.
        layers: Block indices to edit at — the workspace band, typically.
        edit: ``(flat, layer) -> flat`` where ``flat`` is
            ``[batch, seq, d_source]``. Returning ``flat`` unchanged is a
            no-op, which is what makes ``alpha=0`` exactly identity.
        positions: Sequence positions to edit, or ``None`` for every position.
            Negative indices count from the end.
    """

    def __init__(
        self,
        model: LensModel,
        layers: Iterable[int],
        edit,
        *,
        positions: Sequence[int] | None = None,
    ) -> None:
        self._model = model
        self._layers = sorted(set(layers))
        self._edit = edit
        self._positions = None if positions is None else list(positions)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        def hook(module: nn.Module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            shape = tensor.shape
            flat = tensor.reshape(shape[0], shape[1], -1)
            edited = self._edit(flat, layer)
            if self._positions is not None:
                # Restore every position we were not asked to touch. Editing a
                # subset in-place would alias the graph, so rebuild instead.
                keep = torch.ones(shape[1], dtype=torch.bool, device=flat.device)
                idx = torch.as_tensor(
                    [p % shape[1] for p in self._positions], device=flat.device
                )
                keep[idx] = False
                edited = torch.where(keep[None, :, None], flat, edited)
            new = edited.reshape(shape).to(tensor.dtype)
            if torch.is_tensor(output):
                return new
            return (new, *output[1:])

        return hook

    def __enter__(self) -> Intervention:
        try:
            for layer in self._layers:
                self._handles.append(
                    self._model.layers[layer].register_forward_hook(
                        self._make_hook(layer)
                    )
                )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def swap(
    lens: JacobianLens,
    model: LensModel,
    *,
    layers: Iterable[int],
    from_token: int,
    to_token: int,
    positions: Sequence[int] | None = None,
    alpha: float = 1.0,
    mode: str = "transfer",
) -> Intervention:
    """Replace ``from_token``'s lens coordinate with ``to_token``'s.

    ``"transfer"`` (default, and the one the protocol actually describes)
        ``c = <h, d_from>``; ``h' = h + alpha * c * (d_to - d_from)``, i.e.
        ``h - alpha*c*d_from + alpha*c*d_to``. At ``alpha=1`` this *is*
        "clamping a lens coordinate replaces one token's direction with
        another's" written out: the source component is removed and the same
        amount is re-added along the target. Because the edit is proportional
        to ``c``, it is naturally **sparse** — it bites only at the positions
        where the concept is actually present, which measurement shows is a
        handful (``max|c|`` runs 10–20× ``mean|c|`` across positions).

    ``"clamp"``, ``"contrastive"``
        Dense variants that set the target coordinate to a fixed per-layer
        level instead of a position-dependent one. **Both fail badly on
        DeepSeek-V4** — 0/53 on the multi-hop swap, with every item landing on
        neither answer — because injecting a constant at all ~30 positions ×
        21 band layers is a uniform bias, not a substitution. Kept because the
        failure is informative and the modes are the right shape for a model
        whose concepts are spread rather than localised; do not reach for them
        without checking the unchanged/other split first.

    Reading the failure mode: **"unchanged" means the edit is too weak,
    "other" means it is too strong.** ``transfer`` at low alpha shows 21
    unchanged / 12 other; the dense modes show 0 unchanged / 53 other. That
    split diagnoses scale errors faster than any amount of staring at the code.

    Args:
        lens: The fitted lens.
        model: Model to hook.
        layers: Band layers to edit at.
        from_token: Token whose direction is removed.
        to_token: Token whose direction replaces it.
        positions: Positions to edit, or ``None`` for all.
        alpha: Swap strength. ``0.0`` is exactly the identity in every mode.
        mode: One of ``"transfer"``, ``"clamp"``, ``"contrastive"``.

    Returns:
        An :class:`Intervention` to use as a context manager.
    """
    if mode not in ("transfer", "clamp", "contrastive"):
        raise ValueError(f"unknown swap mode {mode!r}")
    cache: dict[int, torch.Tensor] = {}

    def dirs(layer: int) -> torch.Tensor:
        if layer not in cache:
            if mode == "contrastive":
                cache[layer] = torch.stack([
                    lens_directions(lens, model, layer, [from_token])[0],
                    contrastive_direction(lens, model, layer, from_token, to_token),
                ])
            else:
                cache[layer] = lens_directions(
                    lens, model, layer, [from_token, to_token]
                )
        return cache[layer]

    def edit(flat: torch.Tensor, layer: int) -> torch.Tensor:
        D = dirs(layer).to(flat.device, flat.dtype)
        d_from, d_to = D[0], D[1]
        coeff = flat @ d_from                                   # [batch, seq]
        if mode == "transfer":
            return flat + alpha * coeff[..., None] * (d_to - d_from)
        kappa = _coordinate_scale(flat)
        if mode == "clamp":
            # alpha scales the whole substitution, not just the added term, so
            # that alpha=0 is the identity and stays usable as the control.
            return flat + alpha * (kappa * d_to - coeff[..., None] * d_from)
        return flat + alpha * kappa * d_to

    return Intervention(model, layers, edit, positions=positions)


def steer(
    lens: JacobianLens,
    model: LensModel,
    *,
    layers: Iterable[int],
    token: int,
    strength: float,
    positions: Sequence[int] | None = None,
) -> Intervention:
    """Inject ``token``'s direction additively.

    Follows the protocol in ``data/experiments/README.md`` (§verbal-
    introspection) — the unit-normalised direction is scaled by the layer's own
    residual scale times ``strength``, which is what makes one ``strength``
    comparable across layers whose raw norms differ by an order of magnitude
    across depth — **with one deliberate deviation**: the scale used is the
    per-*coordinate* magnitude (:func:`_coordinate_scale`), not the full
    residual norm the README's wording implies. On a 16384-wide source space
    those differ by 128×, and the literal reading injects so much that the
    forward pass collapses. Strength is therefore expressed in units of "a
    typical concept's own coordinate", so ``strength=1`` is a substantial but
    survivable injection rather than an obliterating one.

    Args:
        lens: The fitted lens.
        model: Model to hook.
        layers: Band layers to edit at.
        token: Token to inject.
        strength: Multiple of the layer's mean residual norm. ``0.0`` is
            exactly the identity, and is the protocol's control condition.
        positions: Positions to edit, or ``None`` for all.

    Returns:
        An :class:`Intervention` to use as a context manager.
    """
    cache: dict[int, torch.Tensor] = {}

    def edit(flat: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in cache:
            cache[layer] = lens_directions(lens, model, layer, [token])[0]
        d = cache[layer].to(flat.device, flat.dtype)
        return flat + strength * _coordinate_scale(flat) * d

    return Intervention(model, layers, edit, positions=positions)


def ablate(
    lens: JacobianLens,
    model: LensModel,
    *,
    layers: Iterable[int],
    token_ids: Sequence[int],
    positions: Sequence[int] | None = None,
    random_control: bool = False,
    generator: torch.Generator | None = None,
) -> Intervention:
    """Project the span of ``token_ids``' lens directions out of the residual.

    The directions are orthonormalised before projection, so overlapping tokens
    remove their shared component once rather than several times over — without
    that, a set like {thinking, thoughts, thought} would scrub far more than k
    directions' worth of signal.

    Args:
        lens: The fitted lens.
        model: Model to hook.
        layers: Band layers to edit at.
        token_ids: Tokens whose directions span the removed subspace.
        positions: Positions to edit, or ``None`` for all.
        random_control: Remove a random subspace of the same dimension
            instead. This is the matched-norm control the ablation experiments
            need: same rank removed, no lens semantics.
        generator: RNG for ``random_control``, for reproducible controls.

    Returns:
        An :class:`Intervention` to use as a context manager.
    """
    cache: dict[int, torch.Tensor] = {}

    def basis(layer: int, device, dtype) -> torch.Tensor:
        if layer not in cache:
            D = lens_directions(lens, model, layer, token_ids)
            if random_control:
                D = torch.randn(
                    D.shape, generator=generator, dtype=torch.float32
                ).to(D.device)
            # Q columns span the same subspace; rank may be < len(token_ids)
            # if directions are collinear, which is the correct behaviour.
            Q, _ = torch.linalg.qr(D.float().T)
            cache[layer] = Q
        return cache[layer].to(device, dtype)

    def edit(flat: torch.Tensor, layer: int) -> torch.Tensor:
        Q = basis(layer, flat.device, flat.dtype)               # [d_source, r]
        return flat - (flat @ Q) @ Q.T

    return Intervention(model, layers, edit, positions=positions)

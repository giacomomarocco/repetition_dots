"""Factorial residual-lens and causal-transplant utilities for DeepSeek V4.

This module is deliberately independent of SGLang's HTTP server.  It is meant
to be imported in the model-worker process (or a small direct-model harness),
where the native model object and its paged cache are available.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

from deepseek_v4_logit_lens import DeepseekV4LensWeights, project_logits


@dataclass(frozen=True)
class PositionSet:
    """Absolute prompt positions requested by the experiment."""

    last_question: int
    fillers: tuple[int, ...]
    answer_prompt: int

    def all(self) -> tuple[int, ...]:
        return (self.last_question, *self.fillers, self.answer_prompt)

    def labels(self) -> dict[int, str]:
        out = {self.last_question: "last_question", self.answer_prompt: "answer_prompt"}
        out.update({p: f"filler_{i}" for i, p in enumerate(self.fillers)})
        if len(out) != len(self.all()):
            raise ValueError("last-question, filler, and answer-prompt positions overlap")
        return out


def locate_positions(
    tokenizer: Any,
    prompt: str,
    *,
    question_prefix: str,
    filler_text: str,
    answer_prefix: str = "Answer: ",
    generation_prefix: str = "",
) -> tuple[list[int], PositionSet]:
    """Tokenize exact prompt substrings and return absolute aligned positions.

    Prefix tokenization is used instead of offset mappings because the official
    DeepSeek tokenizer need not be a fast tokenizer.  The function verifies
    that concatenating the four supplied substrings exactly reconstructs the
    rendered prompt.
    """
    if prompt != question_prefix + filler_text + answer_prefix + generation_prefix:
        raise ValueError("position substrings do not concatenate to the rendered prompt")
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    q = encode(question_prefix)
    qf = encode(question_prefix + filler_text)
    qfa = encode(question_prefix + filler_text + answer_prefix)
    full = encode(prompt)
    if (
        not q or not qfa or not full or full[: len(q)] != q
        or full[: len(qf)] != qf or full[: len(qfa)] != qfa
    ):
        raise ValueError("token-boundary merge prevents unambiguous prefix positions")
    fillers = tuple(range(len(q), len(qf)))
    return full, PositionSet(len(q) - 1, fillers, len(qfa) - 1)


@dataclass(frozen=True)
class FactorialCell:
    panel_id: str
    split: str
    row: int
    col: int
    left_id: str
    right_id: str
    left_value: int
    right_value: int
    target: int
    kind: str

    @property
    def cell_id(self) -> str:
        return f"{self.panel_id}:{self.row}{self.col}"


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    payload = "\0".join(parts).encode()
    return prefix + hashlib.sha256(payload).hexdigest()[:20]


def crossed_panels(
    left: Sequence[tuple[str, int]],
    right: Sequence[tuple[str, int]],
    *,
    split: str,
    kind: str,
) -> list[FactorialCell]:
    """Construct all adjacent 2x2 panels and rotate all four target roles.

    Values must already be split by operand/fact identity.  Each adjacent pair
    on each axis yields a basic panel; downstream code uses every cell as the
    target and classifies the other cells by which factor differs.
    """
    if len(left) % 2 or len(right) % 2:
        raise ValueError("each factorial axis must have even length")
    cells: list[FactorialCell] = []
    for li in range(0, len(left), 2):
        for ri in range(0, len(right), 2):
            ls, rs = left[li : li + 2], right[ri : ri + 2]
            sums = {lv + rv for _, lv in ls for _, rv in rs}
            if len(sums) != 4:
                raise ValueError(f"panel ({li // 2}, {ri // 2}) has repeated sums")
            pid = _stable_id(f"{split}-", [kind, *(x[0] for x in ls), *(x[0] for x in rs)])
            for i, (lid, lv) in enumerate(ls):
                for j, (rid, rv) in enumerate(rs):
                    cells.append(FactorialCell(pid, split, i, j, lid, rid, lv, rv, lv + rv, kind))
    return cells


def donor_roles(cells: Sequence[FactorialCell]) -> list[dict[str, str]]:
    """Rotate every cell through target and label its four donor directions."""
    by_panel: dict[str, list[FactorialCell]] = defaultdict(list)
    for cell in cells:
        by_panel[cell.panel_id].append(cell)
    rows = []
    for panel in by_panel.values():
        if {(x.row, x.col) for x in panel} != {(0, 0), (1, 0), (0, 1), (1, 1)}:
            raise ValueError("each panel must contain exactly one complete 2x2 crossing")
        lookup = {(x.row, x.col): x for x in panel}
        for target in panel:
            for di, dj, role in ((0, 0, "identity"), (1, 0, "left"), (0, 1, "right"), (1, 1, "both")):
                donor = lookup[(target.row ^ di, target.col ^ dj)]
                rows.append({"panel_id": target.panel_id, "target_id": target.cell_id,
                             "donor_id": donor.cell_id, "role": role})
    return rows


def assert_aligned_and_single_token(
    records: Sequence[Mapping[str, Any]], tokenizer: Any
) -> None:
    """Enforce absolute alignment and one-token answers without dropping cells."""
    lengths = {len(r["input_ids"]) for r in records}
    positions = {tuple(r["positions"].all()) for r in records}
    if len(lengths) != 1 or len(positions) != 1:
        raise ValueError("panel prompts are not absolutely token aligned")
    bad = [r["cell"].cell_id for r in records if len(tokenizer.encode(str(r["cell"].target), add_special_tokens=False)) != 1]
    if bad:
        raise ValueError(f"non-single-token sums: {bad}")


@dataclass
class ResidualCapture:
    """CPU captures indexed as capture[cell_id][layer][absolute_position]."""

    states: dict[str, dict[int, dict[int, torch.Tensor]]]
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": self.states, "metadata": self.metadata}, path)


class NativeResidualHooks:
    """Capture or replace complete post-block mHC residuals.

    SGLang's fused cross-layer mHC optimization returns an unfinished FFN state;
    it must be disabled before model construction for these hooks to be valid.
    """

    def __init__(self, model: Any):
        core = getattr(model, "model", model)
        self.layers = core.layers
        bad = [i for i, layer in enumerate(self.layers) if getattr(layer, "use_fused_mhc_post_pre", False)]
        if bad:
            raise RuntimeError("native residual hooks require use_fused_mhc_post_pre=False; "
                               "restart/configure the worker with fused mHC post/pre disabled")
        self.handles: list[Any] = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @contextmanager
    def capture(self, positions: Sequence[int], sink: dict[int, dict[int, torch.Tensor]]) -> Iterator[None]:
        wanted = torch.tensor(sorted(set(positions)), dtype=torch.long)
        for layer_id, layer in enumerate(self.layers):
            def hook(_module: Any, _args: Any, output: Any, lid: int = layer_id):
                hidden = output[0]
                idx = wanted.to(hidden.device)
                selected = hidden.index_select(0, idx).detach().to("cpu")
                sink[lid] = {int(p): selected[k].clone() for k, p in enumerate(wanted)}
                return output
            self.handles.append(layer.register_forward_hook(hook))
        try:
            yield
        finally:
            self.close()

    @contextmanager
    def transplant(self, layer_id: int, positions: Sequence[int], donor: Mapping[int, torch.Tensor]) -> Iterator[None]:
        wanted = tuple(positions)
        def hook(_module: Any, _args: Any, output: Any):
            values = list(output)
            hidden = values[0].clone()
            for p in wanted:
                hidden[p].copy_(donor[p].to(device=hidden.device, dtype=hidden.dtype))
            values[0] = hidden
            return tuple(values)
        self.handles.append(self.layers[layer_id].register_forward_hook(hook))
        try:
            yield
        finally:
            self.close()


def runtime_metadata(model: Any, *, backend: str, kv_pool: Any | None = None) -> dict[str, Any]:
    """Record actual runtime precision/layout facts; never infer FP8 from config."""
    core = getattr(model, "model", model)
    first = core.layers[0]
    param = next(model.parameters())
    meta: dict[str, Any] = {
        "torch_version": torch.__version__, "python": platform.python_version(),
        "backend": backend, "parameter_dtype": str(param.dtype),
        "hook_dtype": None, "hc_mult": int(first.hc_mult),
        "hidden_size": int(first.hidden_size), "layer_count": len(core.layers),
    }
    if kv_pool is not None:
        meta["kv_pool_class"] = type(kv_pool).__name__
        for name in ("dtype", "store_dtype", "page_size", "quant_block_size"):
            if hasattr(kv_pool, name):
                meta[f"kv_{name}"] = str(getattr(kv_pool, name))
    return meta


def score_numeric_targets(
    states: torch.Tensor,
    weights: DeepseekV4LensWeights,
    token_ids: Mapping[str, int],
) -> dict[str, dict[str, float | int]]:
    """Exact logit, rank, and log-odds versus all other vocabulary tokens."""
    logits = project_logits(states, weights)
    if logits.ndim != 1:
        raise ValueError("score_numeric_targets expects one residual state")
    out = {}
    for label, token_id in token_ids.items():
        value = logits[token_id]
        rank = int((logits > value).sum().item()) + 1
        other = torch.cat((logits[:token_id], logits[token_id + 1 :]))
        out[label] = {"token_id": token_id, "logit": float(value), "rank": rank,
                      "log_odds_vs_rest": float(value - torch.logsumexp(other, 0))}
    return out


def candidate_sites(
    rows: Sequence[Mapping[str, Any]], *, top_k: int = 24, min_panels: int = 2
) -> list[dict[str, Any]]:
    """Select discovery-only sites by panel-level mean directional log-odds shift."""
    grouped: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] != "discovery":
            raise ValueError("candidate selection may only consume discovery rows")
        key = (int(r["layer"]), str(r["position_label"]), str(r["role"]))
        grouped[key][str(r["panel_id"])].append(float(r["target_log_odds_shift"]))
    ranked = []
    for (layer, position, role), panels in grouped.items():
        panel_means = [sum(x) / len(x) for x in panels.values()]
        if len(panel_means) >= min_panels:
            ranked.append({"layer": layer, "position_label": position, "role": role,
                           "panel_count": len(panel_means),
                           "mean_panel_shift": sum(panel_means) / len(panel_means)})
    return sorted(ranked, key=lambda x: abs(x["mean_panel_shift"]), reverse=True)[:top_k]


def should_run_jlens(ordinary_lens_shift: float, causal_shift: float, *, causal_min: float = 0.0) -> bool:
    """Protocol gate: J-Lens is allowed only for lens-negative/causal-positive sites."""
    return ordinary_lens_shift <= 0.0 and causal_shift > causal_min


def factorial_contrasts(values: Mapping[tuple[int, int], float]) -> dict[str, float]:
    """Compute main and interaction contrasts for one complete 2x2 panel."""
    if set(values) != {(0, 0), (1, 0), (0, 1), (1, 1)}:
        raise ValueError("factorial contrasts require one complete 2x2 panel")
    y00, y10, y01, y11 = (float(values[k]) for k in ((0, 0), (1, 0), (0, 1), (1, 1)))
    return {"left_main": ((y10 + y11) - (y00 + y01)) / 2,
            "right_main": ((y01 + y11) - (y00 + y10)) / 2,
            "interaction": y11 - y10 - y01 + y00}


def copy_cache_rows(pool: Any, layer_id: int, source_locs: torch.Tensor, target_locs: torch.Tensor) -> None:
    """Copy opaque physical KV records, including any colocated FP8 scales.

    This intentionally uses the pool's raw backing tensors rather than decoded
    K/V views.  Source and target must be in the same live pool/layout/backend.
    DSV4 pools may have a single packed KV buffer plus a separate indexer buffer.
    """
    if source_locs.shape != target_locs.shape:
        raise ValueError("source and target cache locations must have equal shape")
    value = getattr(pool, "kv_buffer", None)
    if value is None:
        raise TypeError("unsupported cache pool: no raw packed buffers found")
    buf = value[layer_id - getattr(pool, "start_layer", 0)]
    page_size = int(getattr(pool, "page_size", 1))
    if buf.ndim == 2 and buf.dtype == torch.uint8 and hasattr(pool, "get_bytes_per_token"):
        # DeepSeekV4SingleKVPool: each token's 584-byte opaque record contains
        # FP8 no-PE values, all per-block scales, padding, and BF16 RoPE values.
        width = int(pool.get_bytes_per_token())
        for src, dst in zip(source_locs.tolist(), target_locs.tolist()):
            sp, so = divmod(src, page_size)
            dp, do = divmod(dst, page_size)
            chunk = buf[sp, so * width : (so + 1) * width].clone()
            buf[dp, do * width : (do + 1) * width].copy_(chunk)
        return
    if page_size != 1:
        raise NotImplementedError("unknown paged cache layout; provide a backend-specific copier")
    snapshot = buf.index_select(0, source_locs.to(buf.device)).clone()
    buf.index_copy_(0, target_locs.to(buf.device), snapshot)


def logits_metrics(logits: torch.Tensor, token_ids: Mapping[str, int]) -> dict[str, dict[str, float | int]]:
    """Score causal-run logits with the same exact numeric metrics as the lens."""
    if logits.ndim != 1:
        raise ValueError("expected a one-dimensional vocabulary logit vector")
    out = {}
    for label, token_id in token_ids.items():
        value = logits[token_id].float()
        rest = torch.cat((logits[:token_id], logits[token_id + 1 :])).float()
        out[label] = {"token_id": token_id, "logit": float(value),
                      "rank": int((logits > value).sum()) + 1,
                      "log_odds_vs_rest": float(value - torch.logsumexp(rest, 0))}
    return out


def causal_effect(clean: Mapping[str, Mapping[str, float | int]],
                  patched: Mapping[str, Mapping[str, float | int]]) -> dict[str, float]:
    """Signed patched-minus-clean changes, suitable for panel-level analysis."""
    return {label: float(patched[label]["log_odds_vs_rest"]) - float(clean[label]["log_odds_vs_rest"])
            for label in clean}


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

"""Aggregation and plotting for top-object Logit Lens heatmaps."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

OBJECTS = ("A", "X", "A+X")
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")
ANSWER_POSITIONS = ("answer_word", "answer_colon", "answer_prompt")


def position_order(position: str, filler_length: int) -> int:
    if position == "last_question":
        return -1
    if position in ANSWER_POSITIONS:
        return filler_length + ANSWER_POSITIONS.index(position)
    if position.startswith("filler_"):
        return int(position.removeprefix("filler_"))
    raise ValueError(f"unknown prompt position: {position}")


def position_tick_label(position: str) -> str:
    return {"answer_word": "Answer", "answer_colon": ":",
            "answer_prompt": "space (answer_prompt)"}.get(position, position)


def canonical_numeric_token_ids(tokenizer: Any) -> list[int]:
    """Return tokens whose decoded text is exactly a canonical integer.

    Leading whitespace is allowed because sentencepiece/BPE vocabularies often
    encode it in the token. Signs, punctuation, separators, decimals, leading
    zeroes, and special tokens are excluded.
    """
    ids = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if _INTEGER.fullmatch(text):
            ids.append(token_id)
    return ids


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]], *, filler_length: int, correct: bool | None = None
) -> dict[str, Any]:
    """Aggregate example-level indicators into position-by-layer rates."""
    counts: dict[tuple[str, int, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        if int(row["filler_length"]) != filler_length:
            continue
        if correct is not None and bool(row["clean_correct"]) != correct:
            continue
        position, layer = str(row["position_label"]), int(row["layer"])
        totals[position, layer] += 1
        for obj in OBJECTS:
            target_id = int(row["targets"][obj]["token_id"])
            counts[position, layer, "all", obj] += int(row["top_token_id"] == target_id)
            counts[position, layer, "numeric", obj] += int(
                row["top_numeric_token_id"] == target_id
            )
    if not totals:
        cohort = "all" if correct is None else ("correct" if correct else "wrong")
        raise ValueError(f"no {cohort} rows found for filler_length={filler_length}")
    positions = sorted(
        {p for p, _ in totals},
        key=lambda p: position_order(p, filler_length),
    )
    layers = sorted({layer for _, layer in totals})
    rates = {}
    for scope in ("all", "numeric"):
        rates[scope] = {}
        for obj in OBJECTS:
            rates[scope][obj] = [
                [counts[p, layer, scope, obj] / totals[p, layer] for layer in layers]
                for p in positions
            ]
    cohort = "all" if correct is None else ("correct" if correct else "wrong")
    return {"filler_length": filler_length, "cohort": cohort,
            "positions": positions, "layers": layers,
            "rates": rates, "example_counts": {f"{p}:{l}": n for (p, l), n in totals.items()}}


def plot_stats(stats: Mapping[str, Any], output: Path) -> None:
    """Render the requested 2x3 shared-scale heatmap."""
    positions, layers = stats["positions"], stats["layers"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    image = None
    for row_index, (scope, row_title) in enumerate(
        (("all", "Top token"), ("numeric", "Top numerical token"))
    ):
        for col_index, (obj, title) in enumerate(zip(OBJECTS, ("Fact (A)", "Addend (X)", "Sum (A+X)"))):
            axis = axes[row_index, col_index]
            image = axis.imshow(np.asarray(stats["rates"][scope][obj]), origin="lower",
                                aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
            axis.set_title(title)
            if col_index == 0:
                axis.set_ylabel(f"{row_title}\nprompt position")
                axis.set_yticks(range(len(positions)), [position_tick_label(p) for p in positions])
            if row_index == 1:
                axis.set_xlabel("decoder layer")
                ticks = list(range(0, len(layers), max(1, len(layers) // 7)))
                axis.set_xticks(ticks, [layers[i] for i in ticks])
    cohort = stats.get("cohort", "all")
    fig.suptitle(
        f"One-fact addition: target is argmax — {cohort} examples "
        f"(filler length {stats['filler_length']})"
    )
    fig.colorbar(image, ax=axes, label="fraction of examples", shrink=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as source:
        for line in source:
            yield json.loads(line)

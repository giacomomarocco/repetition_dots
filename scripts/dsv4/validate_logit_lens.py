#!/usr/bin/env python3
"""Validate a captured DeepSeek V4 final residual against native SGLang output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from filler.dsv4.lens import load_checkpoint_readout, project_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--native-response", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-id", type=int, default=0)
    parser.add_argument("--final-layer", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-logprob-error", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captures = []
    for rank in range(4):
        path = args.capture_root / f"rank{rank}" / f"pass{args.pass_id:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        captures.append(torch.load(path, map_location="cpu", weights_only=True))

    states = [item["states"][args.final_layer] for item in captures]
    rank_max_abs = [float((state.float() - states[0].float()).abs().max()) for state in states[1:]]
    if any(value != 0.0 for value in rank_max_abs):
        raise AssertionError(f"TP ranks disagree on final residual: {rank_max_abs}")

    weights = load_checkpoint_readout(args.checkpoint, device=args.device)
    logits = project_logits(states[0].to(args.device), weights)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    lens_values, lens_ids = torch.topk(logprobs, k=args.top_k)

    native = json.loads(args.native_response.read_text())
    returned_hidden = torch.tensor(native["meta_info"]["hidden_states"][0])
    if returned_hidden.ndim == 2:
        returned_hidden = returned_hidden[-1]
    capture_hidden = states[0].flatten().float()
    returned_hidden_max_abs = float(
        (returned_hidden.float() - capture_hidden).abs().max()
    )
    if returned_hidden_max_abs != 0.0:
        raise AssertionError(
            "SGLang returned final pre-readout state differs from layer-42 hook: "
            f"max_abs={returned_hidden_max_abs}"
        )
    native_id = int(native["output_ids"][0])
    native_top = native["meta_info"]["output_top_logprobs"][0]
    native_top_ids = [int(row[1]) for row in native_top]
    native_logprobs = {int(row[1]): float(row[0]) for row in native_top}
    lens_id = int(lens_ids[0])
    errors = {
        str(token_id): abs(float(logprobs[token_id]) - value)
        for token_id, value in native_logprobs.items()
    }
    result = {
        "passed": lens_id == native_id and max(errors.values()) <= args.max_logprob_error,
        "native_token_id": native_id,
        "lens_token_id": lens_id,
        "argmax_equal": lens_id == native_id,
        "rank_max_abs_differences": rank_max_abs,
        "returned_hidden_max_abs_difference": returned_hidden_max_abs,
        "lens_top_ids": [int(x) for x in lens_ids.cpu()],
        "native_top_ids": native_top_ids,
        "top_k_overlap": len(set(native_top_ids) & set(int(x) for x in lens_ids.cpu())),
        "native_top_logprob_abs_errors": errors,
        "max_native_top_logprob_abs_error": max(errors.values()),
        "max_allowed_logprob_error": args.max_logprob_error,
        "lens_top_logprobs": [float(x) for x in lens_values.cpu()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

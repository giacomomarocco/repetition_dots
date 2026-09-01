"""One-shot SGLang forward hook for DeepSeek V4 Logit Lens captures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def hook_spec(output_dir: str | Path, *, num_layers: int = 43) -> list[dict[str, Any]]:
    """Build a spec that matches decoder blocks and no descendant modules."""
    output_dir = Path(output_dir)
    return [
        {
            "name": "dsv4-logit-lens",
            # SGLang uses fnmatch, where ``model.layers.*`` also matches every
            # descendant. Exact names are required for post-block residuals.
            "target_modules": [f"model.layers.{i}" for i in range(num_layers)],
            "hook_factory": "dsv4_lens_hooks:make_layer_capture_hook",
            "config": {
                "output_dir": str(output_dir),
                "trigger_file": str(output_dir / "CAPTURE_NEXT"),
                "num_layers": num_layers,
                "position": -1,
                "capture_first_pass": True,
            },
        }
    ]


def print_hook_spec(output_dir: str) -> None:
    """Print compact JSON for the shell launcher."""
    print(json.dumps(hook_spec(output_dir), separators=(",", ":")))


def make_layer_capture_hook(config: dict[str, Any]):
    """Create one hook shared by exact ``model.layers.N`` targets in order.

    The first model pass is captured by default.  Later captures can be armed
    by creating or updating ``trigger_file``.  Each distinct file mtime arms
    exactly one pass on every rank, leaving the hook inert between experiments.
    """
    output_dir = Path(config["output_dir"])
    trigger_file = Path(config.get("trigger_file", output_dir / "CAPTURE_NEXT"))
    num_layers = int(config.get("num_layers", 43))
    capture_first_pass = bool(config.get("capture_first_pass", True))
    position = int(config.get("position", -1))
    state: dict[str, Any] = {
        "call": 0,
        "pass": 0,
        "active": False,
        "states": {},
        "trigger_mtime_ns": None,
    }

    def hook(_module: Any, _args: Any, output: Any):
        if getattr(_module, "use_fused_mhc_post_pre", False):
            raise RuntimeError(
                "Logit Lens capture requires SGLANG_OPT_FUSE_MHC_POST_PRE=0"
            )
        layer_id = state["call"] % num_layers
        if layer_id == 0:
            try:
                trigger_mtime_ns = trigger_file.stat().st_mtime_ns
            except FileNotFoundError:
                trigger_mtime_ns = None
            newly_triggered = (
                trigger_mtime_ns is not None
                and trigger_mtime_ns != state["trigger_mtime_ns"]
            )
            state["trigger_mtime_ns"] = trigger_mtime_ns
            state["active"] = (
                capture_first_pass and state["pass"] == 0
            ) or newly_triggered
            state["states"] = {}

        if state["active"]:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(
                    "DeepSeek V4 lens hook expected [tokens, hc_mult, hidden_size], "
                    f"got {type(hidden)!r} shape={getattr(hidden, 'shape', None)}"
                )
            state["states"][layer_id] = hidden[position].detach().to("cpu").clone()

        state["call"] += 1
        if layer_id == num_layers - 1:
            if state["active"]:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                rank_dir = output_dir / f"rank{rank}"
                rank_dir.mkdir(parents=True, exist_ok=True)
                destination = rank_dir / f"pass{state['pass']:05d}.pt"
                temporary = destination.with_suffix(f".tmp.{os.getpid()}")
                torch.save(
                    {
                        "states": state["states"],
                        "metadata": {
                            "rank": rank,
                            "pass": state["pass"],
                            "position": position,
                            "num_layers": num_layers,
                        },
                    },
                    temporary,
                )
                os.replace(temporary, destination)
            state["pass"] += 1
            state["active"] = False
            state["states"] = {}
        return output

    return hook

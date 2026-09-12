"""Serial SGLang full-prompt mHC capture, simultaneous patch and restoration."""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch

from filler.dsv4.factorial import replace_residual_rows, residual_output
from filler.dsv4.patching import atomic_json, file_digest, sync_directory


def campaign_hook_spec(control_root: Path, num_layers: int = 43, *, raw_logits: bool = False) -> list[dict]:
    return [{"name": "dsv4-full-prompt-campaign",
             "target_modules": [f"model.layers.{i}" for i in range(num_layers)],
             "hook_factory": "filler.dsv4.campaign_hook:make_campaign_hook",
             "config": {"control_root": str(control_root), "num_layers": num_layers}}] + ([{
                 "name": "dsv4-candidate-logits", "target_modules": ["logits_processor"],
                 "hook_factory": "filler.dsv4.patching_logits:make_logits_hook",
                 "config": {"control_root": str(control_root)}}] if raw_logits else [])


def atomic_torch_save(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as sink:
        torch.save(value, sink)
        sink.flush()
        os.fsync(sink.fileno())
    checksum = file_digest(temporary)
    os.replace(temporary, path)
    sync_directory(path.parent)
    return checksum


def make_campaign_hook(config: dict):
    import json

    root, n = Path(config["control_root"]), int(config.get("num_layers", 43))
    control_path = root / "NEXT.json"
    seen = None
    active = None
    expected_layer = 0
    captures: dict = {}
    audits: list = []
    cache: OrderedDict = OrderedDict()
    bank_cache: OrderedDict = OrderedDict()

    def random_bank(control):
        ref = control["replacement_bank"]
        key = (ref["path"], ref["sha256"])
        if key not in bank_cache:
            from filler.dsv4.filler_random import load_capture
            bank_cache[key] = load_capture(ref)
            while len(bank_cache) > 4:
                bank_cache.popitem(last=False)
        value = bank_cache[key]
        meta = value["metadata"]
        if (any(meta[k] != control[k] for k in ("runtime_id", "config_hash", "cell_id"))
                or meta["clean_capture"] != control["clean_capture"]
                or meta["baseline_id"] != ref["baseline_id"]):
            raise RuntimeError("random bank runtime/baseline mismatch")
        return value

    def load(ref: dict, rank: int, control: dict) -> dict:
        path = Path(ref["ranks"][str(rank)]["path"])
        checksum = ref["ranks"][str(rank)]["sha256"]
        key = (str(path), checksum)
        if key not in cache:
            if file_digest(path) != checksum:
                raise RuntimeError("capture checksum mismatch")
            value = torch.load(path, map_location="cpu", weights_only=True)
            meta = value["metadata"]
            if (meta["rank"] != rank or meta["runtime_id"] != control["runtime_id"]
                    or meta["config_hash"] != control["config_hash"]
                    or meta["num_tokens"] != control["num_tokens"]
                    or meta["cell_id"] != ref["cell_id"]
                    or meta["positions"] != list(range(control["num_tokens"]))
                    or set(value["states"]) != set(range(n))):
                raise RuntimeError("capture runtime/shape/cell alignment mismatch")
            cache[key] = value
            while len(cache) > 4:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def hook(module, _args, output):
        nonlocal seen, active, expected_layer, captures, audits
        lid = int(module.layer_id)
        if lid != expected_layer:
            raise RuntimeError(f"expected layer {expected_layer}, received {lid}")
        expected_layer = (lid + 1) % n
        if getattr(module, "use_fused_mhc_post_pre", False):
            raise RuntimeError("campaign requires SGLANG_OPT_FUSE_MHC_POST_PRE=0")
        if lid == 0:
            active, captures, audits = None, {}, []
            if control_path.is_file():
                control = json.loads(control_path.read_text())
                if control["request_id"] != seen:
                    active, seen = control, control["request_id"]
                    layers = active["layers"]
                    if len(set(layers)) != len(layers) or not set(layers) <= set(range(n)):
                        raise RuntimeError("invalid layer set")
        if active is None:
            return output
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or len(hidden) != active["num_tokens"]:
            raise RuntimeError("expected serial full-prompt pass; cached/chunked/batched input is invalid")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        clean = load(active["clean_capture"], rank, active)["states"][lid] if active.get("clean_capture") and active["recomputation"] == "answer_only" else None
        donor = load(active["donor_capture"], rank, active)["states"][lid] if lid in active["layers"] else None
        if active.get("replacement_bank") and lid in active["layers"]:
            bank = random_bank(active)
            donor = {p: bank["states"][lid][i] for i, p in enumerate(bank["metadata"]["positions"])}
        positions = active["positions"]
        sources = active.get("source_positions", positions)
        changed = replace_residual_rows(hidden, positions=positions, donor=donor, clean=clean,
            recomputation=active["recomputation"], answer_position=active["num_tokens"] - 1,
            source_positions=sources)
        audit = {"layer": lid, "patched_positions": positions if donor is not None else [],
                 "restored_count": 0, "replacement_exact": True, "restoration_exact": True}
        if donor is not None:
            actual = changed[positions].detach().cpu()
            expected = torch.stack([donor[p] for p in sources]) if isinstance(donor, dict) else donor[sources]
            audit["replacement_exact"] = torch.equal(actual, expected.to(actual.dtype))
            if active.get("replacement_bank"):
                from filler.dsv4.filler_random import norm_error
                native_clean = load(active["clean_capture"], rank, active)["states"][lid]
                audit["max_norm_error"] = max(norm_error(actual[i], native_clean[p]) for i, p in enumerate(positions))
        if "source_positions" in active:
            audit["source_positions"] = sources if donor is not None else []
            unselected = [p for p in range(len(hidden)) if p not in positions]
            audit["unselected_exact"] = torch.equal(changed[unselected], hidden[unselected])
            if active["recomputation"] == "full_downstream" and not audit["unselected_exact"]:
                raise RuntimeError("unselected residual rows changed during replacement")
        if active["recomputation"] == "answer_only":
            frozen = [p for p in range(len(hidden)) if p not in positions and p != len(hidden) - 1]
            actual = changed[frozen].detach().cpu()
            audit["restored_count"] = len(frozen)
            audit["restoration_exact"] = torch.equal(actual, clean[frozen].to(actual.dtype))
        if not audit["replacement_exact"] or not audit["restoration_exact"]:
            raise RuntimeError(f"post-block intervention integrity failure: {audit}")
        audits.append(audit)
        saved_positions = list(range(len(hidden))) if active["capture_all"] else active.get("capture_positions", sorted(set([*positions, *sources, len(hidden) - 1])))
        captures[lid] = changed[saved_positions].detach().cpu().clone()
        if "capture_positions" in active and not torch.isfinite(captures[lid]).all():
            raise RuntimeError("nonfinite captured residual")
        if lid == n - 1:
            path = Path(active["output_root"]) / f"rank{rank}.pt"
            metadata = {key: active[key] for key in ("request_id", "runtime_id", "config_hash", "num_tokens", "cell_id")}
            if "source_positions" in active:
                metadata["source_positions"] = sources
            if active.get("replacement_bank"):
                metadata["replacement_bank"] = active["replacement_bank"]
            metadata.update(rank=rank, positions=saved_positions, num_layers=n,
                            dtype=str(hidden.dtype), shape=list(hidden.shape),
                            torch_version=str(torch.__version__), fused_mhc_post_pre=False)
            checksum = atomic_torch_save(path, {"states": captures, "metadata": metadata, "audits": audits})
            atomic_json(root / "acks" / f"{active['request_id']}.rank{rank}.json", {
                **metadata, "layers": active["layers"], "patch_positions": positions,
                "recomputation": active["recomputation"], "audits": audits,
                "capture": {"path": str(path), "sha256": checksum}})
            active, captures, audits = None, {}, []
        return residual_output(output, changed)
    return hook


def validate_acknowledgements(root: Path, control: dict, tp_size: int = 4, num_layers: int = 43) -> list[dict]:
    """Fail closed on missing/stale/wrong-rank or partially applied operations."""
    import json
    acks = [json.loads((root / "acks" / f"{control['request_id']}.rank{rank}.json").read_text())
            for rank in range(tp_size)]
    validate_ack_records(acks, control, tp_size, num_layers)
    return acks


def validate_ack_records(acks: list[dict], control: dict, tp_size: int = 4, num_layers: int = 43) -> None:
    if len(acks) != tp_size:
        raise ValueError("missing rank acknowledgements")
    for rank in range(tp_size):
        ack = acks[rank]
        for key in ("request_id", "runtime_id", "config_hash", "num_tokens", "cell_id", "layers", "recomputation"):
            if ack[key] != control[key]:
                raise ValueError(f"acknowledgement {key} mismatch on rank {rank}")
        saved = list(range(control["num_tokens"])) if control["capture_all"] else control.get("capture_positions", sorted(set([*control["positions"], *control.get("source_positions", []), control["num_tokens"] - 1])))
        if (ack["rank"] != rank or ack["patch_positions"] != control["positions"]
                or ack["positions"] != saved or ack["fused_mhc_post_pre"] is not False
                or [a["layer"] for a in ack["audits"]] != list(range(num_layers))):
            raise ValueError("rank, position or layer acknowledgements incomplete")
        if "source_positions" in control and ack.get("source_positions") != control["source_positions"]:
            raise ValueError("source position acknowledgement mismatch")
        if ack.get("replacement_bank") != control.get("replacement_bank"):
            raise ValueError("random bank acknowledgement mismatch")
        for audit in ack["audits"]:
            patched = control["positions"] if audit["layer"] in control["layers"] else []
            restored = control["num_tokens"] - len(control["positions"]) - 1 if control["recomputation"] == "answer_only" else 0
            if control.get("replacement_bank") and patched:
                import math
                error = audit.get("max_norm_error", float("inf"))
                if not math.isfinite(error) or not 0 <= error <= .005:
                    raise ValueError("random norm acknowledgement failed")
            if "source_positions" in control:
                sources = control["source_positions"] if patched else []
                if audit.get("source_positions") != sources or (control["recomputation"] == "full_downstream" and audit.get("unselected_exact") is not True):
                    raise ValueError("source mapping or unselected row acknowledgement mismatch")
            if (audit["patched_positions"] != patched or audit["restored_count"] != restored
                    or audit["replacement_exact"] is not True or audit["restoration_exact"] is not True):
                raise ValueError("incomplete replacement/restoration acknowledgement")
        if not Path(ack["capture"]["path"]).is_file():
            raise ValueError("acknowledged capture missing")

"""Prepare exact J-Lens argmax rows for the one-fact heatmap notebook."""
from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import ExitStack
import io
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from filler.dsv4.jlens import (
    JLens, SOURCE_LAYERS, WORKSPACE_SOURCE_LAYERS, load_jlens, load_workspace_jlens,
    project_jlens_logits,
)
from filler.dsv4.workspace_jlens_artifact import DEFAULT_PATH as WORKSPACE_LENS_PATH
from filler.dsv4.lens import load_checkpoint_readout
from filler.dsv4.top_object_heatmap import OBJECTS, canonical_numeric_token_ids

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "runs/deepseek-v4-flash/logit-lens/57804608"
OUTPUT = ROOT / "runs/deepseek-v4-flash/jlens-top-object-heatmap"
ROWS_NAME = "jlens_rows_with_numeric_argmax.jsonl"


def collect_examples(grid_root: Path, capture_root: Path, filler_lengths=None):
    """Inspect small manifests and capture paths without loading any tensors."""
    selected = None if filler_lengths is None else set(filler_lengths)
    examples, manifests, seen, passes = [], {}, set(), set()
    for path in sorted(grid_root.glob("k*/*/manifest.json")):
        raw = path.read_bytes()
        manifest = json.loads(raw)
        length = int(manifest["filler_length"])
        if selected is not None and length not in selected:
            continue
        manifests[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        labels = ["last_question", *[f"filler_{i}" for i in range(length)], "answer_prompt"]
        for cell in manifest["cells"]:
            if cell["kind"] != "one_fact" or type(cell["clean_correct"]) is not bool:
                raise ValueError(f"expected one-fact cells with boolean clean_correct: {path}")
            if [capture["label"] for capture in cell["captures"]] != labels:
                raise ValueError(f"incomplete or misordered positions: {path}, {cell['cell_id']}")
            targets = {name: {"token_id": int(cell["target_token_ids"][name])} for name in OBJECTS}
            for capture in cell["captures"]:
                key = (length, cell["cell_id"], capture["label"])
                pass_id = int(capture["pass_id"])
                if key in seen or pass_id in passes:
                    raise ValueError(f"duplicate example or capture pass: {key}, {pass_id}")
                seen.add(key)
                passes.add(pass_id)
                capture_path = capture_root / "rank0" / f"pass{pass_id:05d}.pt"
                if not capture_path.is_file():
                    raise FileNotFoundError(capture_path)
                examples.append({
                    "panel_id": manifest["panel_id"], "cell_id": cell["cell_id"],
                    "split": manifest["split"], "row": cell["row"], "col": cell["col"],
                    "left_value": cell["left_value"], "right_value": cell["right_value"],
                    "target": cell["target"], "clean_correct": cell["clean_correct"],
                    "filler_length": length, "position_label": capture["label"],
                    "absolute_position": capture["absolute_position"], "pass_id": pass_id,
                    "targets": targets, "capture_path": str(capture_path.resolve()),
                })
    if not examples:
        raise ValueError(f"no selected one-fact captures below {grid_root}")
    if selected is not None and selected != {row["filler_length"] for row in examples}:
        raise ValueError("some requested filler lengths have no captures")
    return examples, manifests


def compact_metrics(logits, targets, numeric_ids, *, target_metrics=False):
    """Ranks count strictly larger logits; argmax ties choose the lowest ID.

    Only three target columns are retained. Log-normalizers use the complete
    fp32 vocabulary, never the numeric subset.
    """
    if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite or non-batched readout logits")
    ids = torch.tensor(sorted(set(numeric_ids)), device=logits.device, dtype=torch.long)
    if not len(ids) or int(ids[0]) < 0 or int(ids[-1]) >= logits.shape[-1]:
        raise ValueError("numeric token IDs outside vocabulary")
    if len(targets) != len(logits):
        raise ValueError("target batch size mismatch")
    top = logits.argmax(-1).tolist()
    numeric = ids[logits.index_select(-1, ids).argmax(-1)].tolist()
    result = [{"top_token_id": winner, "top_numeric_token_id": number}
              for winner, number in zip(top, numeric)]
    if target_metrics:
        tokens = torch.tensor([[row[name]["token_id"] for name in OBJECTS] for row in targets],
                              dtype=torch.long, device=logits.device)
        if bool(((tokens < 0) | (tokens >= logits.shape[-1])).any()):
            raise ValueError("target token outside vocabulary")
        values = logits.gather(1, tokens)
        logprobs = values.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
        # Avoid a batch x targets x vocabulary allocation.
        ranks = torch.stack([(logits > values[:, i:i+1]).sum(-1) + 1
                             for i in range(len(OBJECTS))], -1).tolist()
        for index, row in enumerate(result):
            row["targets"] = {name: {"token_id": int(tokens[index, i]),
                                     "rank": ranks[index][i],
                                     "logprob": float(logprobs[index, i])}
                              for i, name in enumerate(OBJECTS)}
    return result


@torch.inference_mode()
def write_scores_many(examples, lenses, weights, numeric_ids, outputs, *, batch_size=64,
                      target_metrics=False, capture_hashes=None):
    """Score multiple lenses from each shared capture batch; never retain logits.

    Optional hashes describe the exact bytes deserialized, shared by both lenses.
    All output files use exclusive creation. Existing single-lens defaults remain
    unchanged through write_scores below.
    """
    if batch_size < 1 or not lenses or set(lenses) != set(outputs):
        raise ValueError("positive batch size and matching lenses/outputs required")
    if len(set(map(str, outputs.values()))) != len(outputs):
        raise ValueError("output paths must be distinct")
    ids = sorted(set(numeric_ids))
    vocab_size = weights.lm_head_weight.shape[0]
    if not ids or ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError("numeric token IDs must be nonempty and inside the vocabulary")
    for example in examples:
        for target in example["targets"].values():
            if target["token_id"] not in ids:
                raise ValueError("target token is not a canonical integer in this vocabulary")
    device = weights.lm_head_weight.device
    layers = sorted({layer for lens in lenses.values() for layer in lens.source_layers})
    counts = dict.fromkeys(lenses, 0)
    with ExitStack() as stack:
        sinks = {name: stack.enter_context(path.open("x")) for name, path in outputs.items()}
        runtime = {name: replace(lens, jacobians={
            layer: matrix.to(device=device, dtype=torch.float32)
            for layer, matrix in lens.jacobians.items()}) for name, lens in lenses.items()}
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            captures, hashes, batch_cache = [], [], {}
            for example in batch:
                path = Path(example["capture_path"])
                if str(path) not in batch_cache:
                    if capture_hashes is not None or example.get("capture_format") == "full-prompt":
                        raw = path.read_bytes()
                        checksum = hashlib.sha256(raw).hexdigest()
                        expected = example.get("_capture_expected_sha256", checksum)
                        if checksum != expected:
                            raise ValueError(f"capture checksum mismatch: {path}")
                        if capture_hashes is not None:
                            if str(path) in capture_hashes and capture_hashes[str(path)] != checksum:
                                raise ValueError(f"capture changed between batches: {path}")
                            capture_hashes[str(path)] = checksum
                        item = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                        del raw
                    else:
                        checksum = None
                        item = torch.load(path, map_location="cpu", weights_only=True)
                    batch_cache[str(path)] = (item, checksum)
                item, checksum = batch_cache[str(path)]
                if example.get("capture_format") == "full-prompt":
                    from filler.dsv4.jlens_full_prompt import selected_states
                    states_at_position = selected_states(item, example, weights, layers)
                else:
                    metadata = item["metadata"]
                    if (metadata["pass"] != example["pass_id"] or metadata["rank"] != 0
                            or metadata["position"] != -1):
                        raise ValueError(f"capture metadata mismatch: {path}")
                    for layer in layers:
                        state = item["states"].get(layer)
                        if (state is None or state.shape != (weights.hc_mult, weights.hidden_size)
                                or state.dtype != weights.lm_head_weight.dtype):
                            raise ValueError(f"missing or incompatible L{layer} residual: {path}")
                    states_at_position = item["states"]
                captures.append(states_at_position)
                hashes.append(checksum)
            for layer in layers:
                states = torch.stack([capture[layer] for capture in captures]).to(device)
                for name, lens in runtime.items():
                    if layer not in lens.jacobians:
                        continue
                    logits = project_jlens_logits(states, lens, layer, weights)
                    metrics = compact_metrics(logits, [e["targets"] for e in batch], ids,
                                              target_metrics=target_metrics)
                    del logits
                    for example, metric, checksum in zip(batch, metrics, hashes):
                        row = {key: value for key, value in example.items() if key != "capture_path" and not key.startswith("_")}
                        row.update(lens="jlens_workspace_mean" if lens.stream_reduction == "mean" else "jlens",
                                   layer=layer, **metric)
                        if checksum is not None:
                            row["capture_sha256"] = checksum
                        sinks[name].write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                        counts[name] += 1
            for sink in sinks.values():
                sink.flush()
            print(f"J-Lens: {min(start + batch_size, len(examples))}/{len(examples)} "
                  f"captures, rows={counts}", flush=True)
    return counts


def write_scores(examples, lens: JLens, weights, numeric_ids, output: Path, *, batch_size=64,
                 target_metrics=False):
    return write_scores_many(examples, {"lens": lens}, weights, numeric_ids, {"lens": output},
                             batch_size=batch_size, target_metrics=target_metrics)["lens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=SAVED / "filler-grid-expanded")
    parser.add_argument("--capture-root", type=Path, default=SAVED / "captures")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16")
    parser.add_argument("--lens-format", choices=("rectangular", "workspace-mean"), default="rectangular",
                        help="rectangular 0731 fit or camilablank's square fit after stream averaging")
    parser.add_argument("--lens", type=Path, default=None,
                        help="local checkpoint (default selected by --lens-format)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--filler-lengths", type=int, nargs="+", default=None,
                        help="subset to score (default: all saved lengths)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--preflight", action="store_true",
                        help="inspect manifests, files, and square-lens metadata only; safe on a login node")
    parser.add_argument("--allow-incomplete-answer-prefix", action="store_true",
                        help="explicitly reproduce the historical grid without Answer/colon positions")
    args = parser.parse_args()
    if not args.allow_incomplete_answer_prefix:
        parser.error("this historical grid omits Answer/colon; use scripts.dsv4.compare_jlenses "
                     "for complete answer-prefix coverage, or explicitly opt into the legacy grid")
    workspace_mean = args.lens_format == "workspace-mean"
    layers = WORKSPACE_SOURCE_LAYERS if workspace_mean else SOURCE_LAYERS
    if args.lens is None:
        args.lens = WORKSPACE_LENS_PATH if workspace_mean else ROOT / "model/jacobian-lens-deepseek-v4-flash-0731/lens.pt"
    if args.output_dir is None:
        args.output_dir = OUTPUT.with_name("workspace-jlens-top-object-heatmap") if workspace_mean else OUTPUT
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    examples, manifest_hashes = collect_examples(args.grid_root, args.capture_root, args.filler_lengths)
    for path in (args.lens, args.checkpoint / "model-00045-of-00048.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if workspace_mean:
        load_workspace_jlens(args.lens, validate_values=False)
    plan = {"captures": len(examples), "rows": len(examples) * len(layers),
            "lens_format": args.lens_format,
            "lens": "jlens_workspace_mean" if workspace_mean else "jlens",
            "stream_reduction": "mean" if workspace_mean else "flatten",
            "target_layer": 41 if workspace_mean else None,
            "source_layers": list(layers),
            "filler_lengths": sorted({row["filler_length"] for row in examples}),
            "examples_by_filler_length": {
                str(length): len({row["cell_id"] for row in examples if row["filler_length"] == length})
                for length in sorted({row["filler_length"] for row in examples})},
            "output": str(args.output_dir / ROWS_NAME)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight:
        return
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("run scoring through srun on an approved compute allocation; "
                           "use --preflight on a login node")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def save(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    provenance = {**plan, "started_at": datetime.now(timezone.utc).isoformat(),
                  "arguments": {key: str(value) for key, value in vars(args).items()},
                  "command": sys.argv, "node": socket.gethostname(),
                  "job_id": os.environ["SLURM_JOB_ID"], "torch": torch.__version__,
                  "manifest_sha256": manifest_hashes,
                  "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__), ROOT / "filler/dsv4/jlens.py",
                                                 ROOT / "filler/dsv4/lens.py",
                                                 ROOT / "filler/dsv4/workspace_jlens_artifact.py",
                                                 ROOT / "filler/dsv4/top_object_heatmap.py")},
                  "transport_dtype": "float32", "norm_rounding": "published HF",
                  "argmax_ties": "lowest token ID in each scope",
                  "cohorts": "saved model clean_correct; unchanged by the lens"}
    save("STARTED.json", provenance)
    try:
        from transformers import AutoTokenizer
        if workspace_mean:
            from filler.dsv4.workspace_jlens_artifact import REPOSITORY, REVISION, verify_artifact
            provenance["lens_sha256"] = verify_artifact(args.lens)
            provenance["lens_repository"], provenance["lens_revision"] = REPOSITORY, REVISION
            lens = load_workspace_jlens(args.lens)
            provenance["lens_provenance"] = lens.provenance
        else:
            from scripts.dsv4.prepare_jlens import LENS_SHA256, LENS_SIZE, file_digest
            if args.lens.stat().st_size != LENS_SIZE or file_digest(args.lens) != LENS_SHA256:
                raise ValueError("lens differs from the pinned published checkpoint")
            provenance["lens_sha256"] = LENS_SHA256
            lens = load_jlens(args.lens)
        weights = load_checkpoint_readout(args.checkpoint, device=args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True,
                                                  trust_remote_code=False)
        numeric_ids = canonical_numeric_token_ids(tokenizer)
        for example in examples:
            for name, value in zip(OBJECTS, (example["left_value"], example["right_value"], example["target"])):
                if tokenizer.encode(str(value), add_special_tokens=False) != [example["targets"][name]["token_id"]]:
                    raise ValueError(f"target tokenization mismatch: {example['cell_id']}, {name}")
        if weights.lm_head_weight.dtype != torch.bfloat16:
            raise ValueError("expected the validated BF16 checkpoint readout")
        provenance["numeric_token_ids"] = numeric_ids
        save("provenance.json", provenance)
        partial = args.output_dir / (ROWS_NAME + ".partial")
        count = write_scores(examples, lens, weights, numeric_ids, partial, batch_size=args.batch_size)
        if count != plan["rows"]:
            raise ValueError("incomplete J-Lens grid")
        for path, expected in manifest_hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"manifest changed during scoring: {path}")
        partial.rename(args.output_dir / ROWS_NAME)
        save("COMPLETE.json", {**plan, "rows": count, "passed": True,
                               "elapsed_seconds": time.monotonic() - started})
    except Exception as error:
        save("FAILED.json", {"error_type": type(error).__name__, "error": str(error),
                             "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

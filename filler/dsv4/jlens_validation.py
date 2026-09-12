"""Reproduce a small saved-activation comparison with the published J-Lens."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import torch

from filler.dsv4.jlens import load_jlens, published_readout_fixture, transport, unembed_transported
from filler.dsv4.lens import load_checkpoint_readout, project_logits

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "runs/deepseek-v4-flash/logit-lens/57804608"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_provenance(lens_path: Path) -> dict:
    """Check the pinned source and matrix file before importing the reference."""
    from scripts.dsv4.prepare_jlens import CODE_REVISION, LENS_REVISION, LENS_SHA256, LENS_SIZE
    source = ROOT / "ports/jacobian-lens-open-frontier"
    manifest = json.loads((source / "SOURCE.json").read_text())
    if manifest["revision"] != CODE_REVISION:
        raise ValueError("reference source revision differs from the prepared pin")
    for name, expected in manifest["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"reference source checksum mismatch: {name}")
    if lens_path.stat().st_size != LENS_SIZE or digest(lens_path) != LENS_SHA256:
        raise ValueError("lens checkpoint differs from the pinned default fit")
    # These imports are scoped to this command; the serving environment is untouched.
    sys.path.insert(0, str(source))
    import jlens
    if Path(jlens.__file__).resolve().parent != source / "jlens":
        raise ValueError("a different jlens package was imported")
    return {"code": manifest, "lens_revision": LENS_REVISION, "lens_sha256": LENS_SHA256}


def read_examples(capture_root: Path, manifest_path: Path, native_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if [c["label"] for c in manifest["captures"]] != ["last_question", *[f"filler_{i}" for i in range(10)], "answer_prompt"]:
        raise ValueError("expected the corrected twelve-position dots_10 pilot")
    if manifest["target_values"] != {"A": 57, "X": 11, "A+X": 68}:
        raise ValueError("unexpected pilot target values")
    native = json.loads(native_path.read_text())
    examples = [{"case": "paris", "position_label": "last_prompt_token", "pass_id": 0,
                 "absolute_position": -1, "response": str(native_path),
                 "target_token_ids": {"Paris": int(native["output_ids"][0])}}]
    examples.extend({"case": "one_fact", "target_token_ids": manifest["target_token_ids"],
                     **capture, "position_label": capture["label"]} for capture in manifest["captures"])
    hashes = {str(manifest_path): digest(manifest_path), str(native_path): digest(native_path)}
    captures = []
    for example in examples:
        ranks = []
        for rank in range(4):
            path = capture_root / f"rank{rank}/pass{example['pass_id']:05d}.pt"
            hashes[str(path)] = digest(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if set(saved["states"]) != set(range(43)):
                raise ValueError(f"incomplete capture: {path}")
            if (saved["metadata"]["pass"] != example["pass_id"]
                    or saved["metadata"]["rank"] != rank
                    or saved["metadata"]["position"] != -1):
                raise ValueError(f"capture metadata mismatch: {path}")
            ranks.append(saved["states"])
        for layer in range(43):
            for rank in range(4):
                state = ranks[rank][layer]
                if state.shape != (4, 4096) or state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise ValueError("capture shape/dtype/finiteness mismatch")
                if not torch.equal(state, ranks[0][layer]):
                    raise ValueError("tensor-parallel rank mismatch")
        response_path = Path(example["response"])
        if not response_path.is_absolute():
            response_path = ROOT / response_path
        hashes[str(response_path)] = digest(response_path)
        response = json.loads(response_path.read_text())
        generated = int(response["output_ids"][0])
        if "generated_token_id" in example and generated != example["generated_token_id"]:
            raise ValueError("pilot response does not match the capture manifest")
        example["native_token_id"] = generated
        captures.append(ranks[0])
    return examples, captures, native, hashes


def validate_paris(final_state, native, weights) -> dict:
    returned = torch.tensor(native["meta_info"]["hidden_states"][0])
    if returned.ndim == 2:
        returned = returned[-1]
    hidden_error = float((returned.to(final_state.device).float().flatten() - final_state.float().flatten()).abs().max())
    logprobs = project_logits(final_state, weights).log_softmax(-1)
    top = native["meta_info"]["output_top_logprobs"][0]
    errors = {str(row[1]): abs(float(logprobs[int(row[1])]) - float(row[0])) for row in top}
    lens_top = logprobs.topk(10)
    native_ids = {int(row[1]) for row in top}
    overlap = len(set(lens_top.indices.tolist()) & native_ids)
    # BF16 vocabulary logits can have exact ties at the top-k boundary. Device
    # topk implementations need not return the same member of such a tie. Accept
    # only exact boundary ties, never approximate or lower-scoring replacements.
    boundary = lens_top.values[-1]
    strict_above = set(torch.nonzero(logprobs > boundary).flatten().tolist())
    tie_match = (len(native_ids) == 10 and strict_above <= native_ids
                 and all(bool(logprobs[token] >= boundary) for token in native_ids))
    result = {"hidden_max_abs_difference": hidden_error, "rank_max_abs_difference": 0.0,
              "argmax_equal": int(logprobs.argmax()) == native["output_ids"][0],
              "top10_overlap": overlap, "native_top_logprob_errors": errors,
              "top10_equal_up_to_exact_ties": tie_match,
              "top10_boundary_logprob": float(boundary),
              "boundary_tie_count": int((logprobs == boundary).sum()),
              "lens_top10": [{"token_id": int(token), "logprob": float(value)}
                             for token, value in zip(lens_top.indices, lens_top.values)],
              "unmatched_native": [{"token_id": token, "lens_logprob": float(logprobs[token]),
                                    "exactly_at_boundary": bool(logprobs[token] == boundary)}
                                   for token in sorted(native_ids - set(lens_top.indices.tolist()))],
              "max_logprob_error": max(errors.values()), "tolerance": 0.15}
    result["passed"] = hidden_error == 0 and result["argmax_equal"] and tie_match and max(errors.values()) <= 0.15
    return result


def token_scores(logits, targets: dict, tokenizer) -> dict:
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite readout logits")
    logprobs = logits.float().log_softmax(-1)
    top = logprobs.topk(min(10, len(logprobs)))
    metrics = {}
    for label, token in targets.items():
        if not 0 <= token < len(logits):
            raise ValueError("target token outside vocabulary")
        value = logits[token]
        rest = torch.cat((logits[:token], logits[token + 1:]))
        metrics[label] = {"token_id": token, "rank": int((logits > value).sum()) + 1,
                          "logit": float(value), "logprob": float(logprobs[token]),
                          "log_odds_vs_rest": float(value - torch.logsumexp(rest, 0))}
    return {"targets": metrics,
            "top10": [{"token_id": int(token), "text": tokenizer.decode([int(token)]), "logprob": float(value)}
                      for token, value in zip(top.indices, top.values)]}


@torch.inference_mode()
def compare_readouts(examples, captures, lens, weights, tokenizer, *, on_layer=None):
    from jlens.lens import JacobianLens

    reference_readout = published_readout_fixture(weights)
    rows, checks = [], []
    for layer in lens.source_layers:
        states = torch.stack([capture[layer] for capture in captures]).to(weights.lm_head_weight.device)
        reference = JacobianLens({layer: lens.jacobians[layer]}, n_prompts=lens.n_prompts,
                                 d_model=lens.d_model, d_source=lens.d_source)
        expected_transport = reference.transport(states.float(), layer)
        actual_transport = transport(states, lens, layer)
        torch.testing.assert_close(actual_transport, expected_transport, rtol=1e-5, atol=1e-5)
        expected_logits = reference_readout.unembed(expected_transport, collapse=False).float()
        actual_logits = unembed_transported(actual_transport, weights)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)
        ordinary_logits = project_logits(states, weights)
        checks.append({"layer": layer, "transport_max_abs_error": float((actual_transport - expected_transport).abs().max()),
                       "readout_max_abs_error": float((actual_logits - expected_logits).abs().max()),
                       "passed": True})
        for index, example in enumerate(examples):
            rows.append({"case": example["case"], "position_label": example["position_label"],
                         "absolute_position": example["absolute_position"], "pass_id": example["pass_id"],
                         "layer": layer, "native_token_id": example["native_token_id"],
                         "jlens": token_scores(actual_logits[index], example["target_token_ids"], tokenizer),
                         "ordinary": token_scores(ordinary_logits[index], example["target_token_ids"], tokenizer)})
        if on_layer:
            on_layer(rows, checks)
        print(f"L{layer}: reference parity passed; {len(examples)} positions scored", flush=True)
    return rows, checks


def export_results(output: Path, rows: list[dict], provenance: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fields = ["case", "position_label", "absolute_position", "pass_id", "layer", "lens", "target",
              "token_id", "rank", "logit", "logprob", "log_odds_vs_rest"]
    with (output / "target_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for name in ("ordinary", "jlens"):
                for target, scores in row[name]["targets"].items():
                    writer.writerow({**{key: row[key] for key in fields[:5]}, "lens": name,
                                     "target": target, **scores})
    labels = ["last_question", *[f"filler_{i}" for i in range(10)], "answer_prompt"]
    layers = sorted({row["layer"] for row in rows})
    lookup = {(row["layer"], row["position_label"]): row for row in rows if row["case"] == "one_fact"}
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for i, target in enumerate(("A", "X", "A+X")):
        for j, name in enumerate(("ordinary", "jlens")):
            values = np.array([[lookup[layer, label][name]["targets"][target]["rank"] for label in labels] for layer in layers])
            axis = axes[i, j]
            chart = axis.imshow(np.log10(values), aspect="auto", origin="lower", vmin=0,
                                vmax=np.log10(provenance["vocab_size"]), cmap="viridis_r")
            axis.set_title(f"{target}: {'Ordinary Logit Lens' if name == 'ordinary' else 'J-Lens'}")
            axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
            axis.set_yticks(range(0, len(layers), 2), layers[::2])
            axis.set_ylabel("Layer (zero-based)")
    figure.colorbar(chart, ax=axes, label="log10(target rank); lower is better", shrink=0.65)
    figure.suptitle("Atatürk addition pilot: A=57, X=11, A+X=68")
    figure.savefig(output / "target_ranks.png", dpi=170)
    figure.savefig(output / "target_ranks.pdf")
    plt.close(figure)
    paris = [row for row in rows if row["case"] == "paris"]
    lines = ["# J-Lens compatibility smoke test", "", f"Date: {provenance['started_at']}", "",
             "Implementation checks passed: pinned file/source integrity, captured-rank agreement, native Paris readout, "
             "and published J-Lens transport/readout parity at every fitted layer.", "",
             "## Paris", "", "| Readout | Best Paris rank | Layer | L30 top five |", "|---|---:|---:|---|"]
    for name in ("ordinary", "jlens"):
        best = min(paris, key=lambda row: row[name]["targets"]["Paris"]["rank"])
        at30 = next(row for row in paris if row["layer"] == 30)
        tokens = ", ".join(repr(item["text"]).replace("|", "\\|") for item in at30[name]["top10"][:5])
        lines.append(f"| {name} | {best[name]['targets']['Paris']['rank']} | {best['layer']} | {tokens} |")
    lines += ["", "## Addition pilot", "", "Best sum rank within layers 19–39; layer ties use the earliest layer.", "",
              "| Position | Ordinary best rank (layer) | J-Lens best rank (layer) |", "|---|---:|---:|"]
    for label in labels:
        selected = [lookup[layer, label] for layer in layers]
        cells = []
        for name in ("ordinary", "jlens"):
            best = min(selected, key=lambda row: row[name]["targets"]["A+X"]["rank"])
            cells.append(f"{best[name]['targets']['A+X']['rank']} (L{best['layer']})")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "![Target ranks](target_ranks.png)", "", "## Interpretation and reproduction", "",
              "This is a two-prompt compatibility pilot using existing SGLang activations. It does not test causal validity, "
              "population performance, or equality with the author's original inference backend. The local model uses "
              "the A100 conversion/runtime; the matrices were fitted on the author's backend. No model was loaded or fitted.", "",
              "J-Lens follows the published HF norm rounding (cast before norm-weight multiplication); ordinary Logit Lens "
              "retains the previously validated SGLang native rounding. Both use the same local norm and vocabulary weights. "
              "Reference parity validates the adapter, not the transfer of a fitted lens between numerical backends.", "",
              "This diagnostic does not select scientific intervention sites; the existing J-Lens causal eligibility gate is unchanged.", "",
              "See `provenance.json` for exact inputs, hashes, revisions, precision, allocation and command; "
              "`validation.json` for checks; `results.json` for both top-ten readouts and all target metrics; "
              "`target_scores.csv` for tabular scores.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, default=ROOT / "model/jacobian-lens-deepseek-v4-flash-0731/lens.pt")
    parser.add_argument("--capture-root", type=Path, default=SAVED / "captures")
    parser.add_argument("--pilot-manifest", type=Path, default=SAVED / "filler-pilot-v2/manifest.json")
    parser.add_argument("--native-response", type=Path, default=SAVED / "native_response.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/deepseek-v4-flash/jlens-smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("run the real checkpoint comparison through srun on an approved compute allocation")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("results.json", "STARTED.json", "FAILED.json")):
        raise FileExistsError("output already contains a run; choose a fresh --output-dir")
    started = time.monotonic()
    provenance = {"started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
                  "python": platform.python_version(), "torch": torch.__version__, "node": socket.gethostname(),
                  "job_id": os.environ["SLURM_JOB_ID"], "device": args.device, "threads": args.threads,
                  "environment": {key: os.environ.get(key) for key in
                                  ("SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
                  "arguments": {key: str(value) for key, value in vars(args).items()},
                  "readout_dtype": "bfloat16", "transport_dtype": "float32",
                  "jlens_norm_rounding": "HF: normalized vector cast before weight multiply",
                  "ordinary_norm_rounding": "SGLang: normalized vector and weight multiplied in float32 then cast"}
    write_json(args.output_dir / "STARTED.json", provenance)
    try:
        provenance.update(source_provenance(args.lens))
        examples, captures, native, hashes = read_examples(args.capture_root, args.pilot_manifest, args.native_response)
        for path in [args.checkpoint / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]:
            hashes[str(path)] = digest(path)
        for path in (Path(__file__), ROOT / "filler/dsv4/jlens.py", ROOT / "filler/dsv4/lens.py"):
            hashes[str(path)] = digest(path)
        provenance["input_and_source_sha256"] = hashes
        shard = args.checkpoint / "model-00045-of-00048.safetensors"
        provenance["readout_shard"] = {"path": str(shard), "size": shard.stat().st_size, "mtime_ns": shard.stat().st_mtime_ns}
        config = json.loads((args.checkpoint / "config.json").read_text())
        if (config["num_hidden_layers"], config["hidden_size"], config["hc_mult"]) != (43, 4096, 4):
            raise ValueError("unexpected base-model configuration")
        provenance["vocab_size"] = config["vocab_size"]
        lens = load_jlens(args.lens)
        weights = load_checkpoint_readout(args.checkpoint, device=args.device)
        if weights.lm_head_weight.dtype != torch.bfloat16 or len(weights.lm_head_weight) != config["vocab_size"]:
            raise ValueError("unexpected local vocabulary head")
        provenance["readout_tensor_sha256"] = {
            name: hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
            for name, tensor in vars(weights).items() if isinstance(tensor, torch.Tensor)}
        write_json(args.output_dir / "provenance.json", provenance)
        native_check = validate_paris(captures[0][42].to(args.device), native, weights)
        write_json(args.output_dir / "native_validation.json", native_check)
        if not native_check["passed"]:
            raise AssertionError("saved Paris native validation failed")
        from transformers import AutoTokenizer
        import transformers
        provenance["transformers"] = transformers.__version__
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        for label, value in {"A": 57, "X": 11, "A+X": 68}.items():
            if tokenizer.encode(str(value), add_special_tokens=False) != [examples[1]["target_token_ids"][label]]:
                raise ValueError("pilot target tokenization changed")
        write_json(args.output_dir / "provenance.json", provenance)

        def checkpoint(rows, checks):
            write_json(args.output_dir / "partial_results.json", {"rows": rows, "checks": checks})

        rows, checks = compare_readouts(examples, captures, lens, weights, tokenizer, on_layer=checkpoint)
        if len(rows) != 273:
            raise AssertionError("incomplete comparison grid")
        for path, expected in hashes.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"input changed during comparison: {path}")
        provenance["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "provenance.json", provenance)
        write_json(args.output_dir / "results.json", {"comparison_count": len(rows), "rows": rows})
        write_json(args.output_dir / "validation.json", {"passed": True, "capture_passes": len(examples),
                   "ranks": 4, "capture_layers": 43, "native": native_check, "reference": checks,
                   "rtol": 1e-5, "atol": 1e-5, "inputs_unchanged": True})
        export_results(args.output_dir, rows, provenance)
        write_json(args.output_dir / "COMPLETE.json", {"passed": True, "elapsed_seconds": time.monotonic() - started,
                   "comparison_count": len(rows), "report": str(args.output_dir / "REPORT.md")})
        print(f"Complete: {args.output_dir / 'REPORT.md'}", flush=True)
    except Exception as error:
        write_json(args.output_dir / "FAILED.json", {"error_type": type(error).__name__, "error": str(error),
                                                    "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

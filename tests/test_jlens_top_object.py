import json

import pytest
import torch

from filler.dsv4.jlens import JLens
from filler.dsv4.jlens_top_object import collect_examples, write_scores
from filler.dsv4.lens import DeepseekV4LensWeights
from filler.dsv4.top_object_heatmap import aggregate_rows, read_jsonl


def saved_grid(tmp_path):
    grid = tmp_path / "grid"
    captures = tmp_path / "captures"
    (captures / "rank0").mkdir(parents=True)
    manifest_path = grid / "k0/panel/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    cells = []
    for cell_index in range(2):
        items = []
        for position_index, label in enumerate(("last_question", "answer_prompt")):
            pass_id = 2 * cell_index + position_index
            items.append({"pass_id": pass_id, "label": label, "absolute_position": position_index})
            state = torch.zeros(4, 3)
            state[0, pass_id % 3] = 4
            torch.save({"metadata": {"pass": pass_id, "rank": 0, "position": -1},
                        "states": {19: state, 39: state}},
                       captures / "rank0" / f"pass{pass_id:05d}.pt")
        cells.append({"cell_id": f"panel:{cell_index}", "kind": "one_fact",
                      "clean_correct": cell_index == 0, "row": 0, "col": cell_index,
                      "left_value": 1, "right_value": 2, "target": 3,
                      "target_token_ids": {"A": 0, "X": 1, "A+X": 2}, "captures": items})
    manifest = {"panel_id": "panel", "split": "discovery", "filler_length": 0, "cells": cells}
    manifest_path.write_text(json.dumps(manifest))
    return grid, captures, manifest_path


def readout():
    # Complete-stream transport selects the first three coordinates, with a
    # different permutation at L39. Poison mHC weights detect a second collapse.
    matrix = torch.eye(3, 12).half()
    lens = JLens({19: matrix, 39: matrix[[2, 0, 1]]}, 1000, 3, 12)
    weights = DeepseekV4LensWeights(
        torch.full((4, 12), float("nan")), torch.full((4,), float("nan")),
        torch.full((1,), float("nan")), torch.ones(3),
        torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [3., 3., 3.], [0., 0., 0.]]),
    )
    return lens, weights


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_saved_grid_to_exact_both_argmaxes_and_cohorts(tmp_path, batch_size):
    grid, captures, manifest_path = saved_grid(tmp_path)
    examples, hashes = collect_examples(grid, captures)
    assert len(examples) == 4 and str(manifest_path.resolve()) in hashes
    lens, weights = readout()
    output = tmp_path / "rows.jsonl"
    # Deliberately unsorted IDs: ties/order must still follow token ID order.
    assert write_scores(examples, lens, weights, [2, 0, 1], output, batch_size=batch_size) == 8
    rows = list(read_jsonl(output))
    assert {row["lens"] for row in rows} == {"jlens"}
    assert {row["top_token_id"] for row in rows} == {3}
    lookup = {(row["pass_id"], row["layer"]): row for row in rows}
    assert [lookup[i, 19]["top_numeric_token_id"] for i in range(4)] == [0, 1, 2, 0]
    assert [lookup[i, 39]["top_numeric_token_id"] for i in range(4)] == [1, 2, 0, 1]
    assert all(row["targets"]["A"] == {"token_id": 0} for row in rows)
    stats = aggregate_rows(rows, filler_length=0)
    assert stats["positions"] == ["last_question", "answer_prompt"]
    assert stats["layers"] == [19, 39]
    assert stats["rates"]["all"]["A"] == [[0., 0.], [0., 0.]]
    assert stats["rates"]["numeric"]["A"] == [[0.5, 0.5], [0.5, 0.]]
    assert set(stats["example_counts"].values()) == {2}
    correct = aggregate_rows(rows, filler_length=0, correct=True)
    wrong = aggregate_rows(rows, filler_length=0, correct=False)
    assert correct["rates"]["numeric"]["A"] == [[1., 0.], [0., 0.]]
    assert wrong["rates"]["numeric"]["A"] == [[0., 1.], [1., 0.]]


def test_argmax_ties_choose_lowest_id_and_existing_output_is_preserved(tmp_path):
    grid, captures, _ = saved_grid(tmp_path)
    examples, _ = collect_examples(grid, captures)
    path = captures / "rank0/pass00000.pt"
    item = torch.load(path, weights_only=True)
    item["states"] = {layer: torch.zeros(4, 3) for layer in (19, 39)}
    torch.save(item, path)
    lens, weights = readout()
    output = tmp_path / "rows.jsonl"
    write_scores(examples[:1], lens, weights, [2, 1, 0], output)
    assert all(row["top_token_id"] == row["top_numeric_token_id"] == 0 for row in read_jsonl(output))
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_scores(examples, lens, weights, [0, 1, 2], output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem,match", [
    ("metadata", "metadata mismatch"), ("layer", "missing or incompatible"),
    ("nan", "nonfinite residual"), ("shape", "incompatible"),
])
def test_incompatible_captures_fail(tmp_path, problem, match):
    grid, captures, _ = saved_grid(tmp_path)
    examples, _ = collect_examples(grid, captures)
    path = captures / "rank0/pass00000.pt"
    item = torch.load(path, weights_only=True)
    if problem == "metadata":
        item["metadata"]["pass"] = 100
    elif problem == "layer":
        del item["states"][19]
    elif problem == "nan":
        item["states"][19][0, 0] = float("nan")
    else:
        item["states"][19] = torch.zeros(3)
    torch.save(item, path)
    lens, weights = readout()
    with pytest.raises(ValueError, match=match):
        write_scores(examples, lens, weights, [0, 1, 2], tmp_path / "rows.jsonl")


@pytest.mark.parametrize("problem,match", [
    ("positions", "positions"), ("duplicate", "duplicate"),
    ("correctness", "boolean"), ("missing_file", "pass00099"),
])
def test_manifest_validation(tmp_path, problem, match):
    grid, captures, path = saved_grid(tmp_path)
    manifest = json.loads(path.read_text())
    cell = manifest["cells"][0]
    if problem == "positions":
        cell["captures"].reverse()
    elif problem == "duplicate":
        manifest["cells"].append(cell)
    elif problem == "correctness":
        cell["clean_correct"] = "false"
    else:
        cell["captures"][0]["pass_id"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        collect_examples(grid, captures)


def test_requested_lengths_must_exist_and_numeric_targets_must_be_valid(tmp_path):
    grid, captures, _ = saved_grid(tmp_path)
    with pytest.raises(ValueError, match="requested filler lengths"):
        collect_examples(grid, captures, [0, 20])
    examples, _ = collect_examples(grid, captures, [0])
    lens, weights = readout()
    with pytest.raises(ValueError, match="target token"):
        write_scores(examples, lens, weights, [0, 1], tmp_path / "rows.jsonl")


@pytest.mark.parametrize("batch_size", [1, 4])
def test_square_lens_saved_grid_keeps_mean_readout_distinct(tmp_path, batch_size):
    grid, captures, _ = saved_grid(tmp_path)
    examples, _ = collect_examples(grid, captures)
    _, weights = readout()
    lens = JLens({19: torch.eye(3).half(), 39: torch.eye(3).half()[[2, 0, 1]]},
                 25, 3, 3, stream_reduction="mean", target_layer=39)
    output = tmp_path / "square.jsonl"
    assert write_scores(examples, lens, weights, [2, 0, 1], output, batch_size=batch_size) == 8
    rows = list(read_jsonl(output))
    assert {row["lens"] for row in rows} == {"jlens_workspace_mean"}
    lookup = {(row["pass_id"], row["layer"]): row for row in rows}
    assert [lookup[i, 19]["top_numeric_token_id"] for i in range(4)] == [0, 1, 2, 0]
    assert [lookup[i, 39]["top_numeric_token_id"] for i in range(4)] == [1, 2, 0, 1]
    assert set(aggregate_rows(rows, filler_length=0)["example_counts"].values()) == {2}


@pytest.mark.parametrize("preflight", [True, False])
def test_workspace_cli_preflight_and_login_node_guard(tmp_path, monkeypatch, capsys, preflight):
    from filler.dsv4 import jlens_top_object as command
    grid, captures, _ = saved_grid(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model-00045-of-00048.safetensors").touch()
    lens_path = tmp_path / "lens.pt"
    lens_path.touch()
    calls = []
    monkeypatch.setattr(command, "WORKSPACE_LENS_PATH", lens_path)
    monkeypatch.setattr(command, "load_workspace_jlens",
                        lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setattr(command, "load_checkpoint_readout",
                        lambda *_a, **_k: pytest.fail("must not load readout weights on login node"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = ["prepare_jlens_top_object", "--allow-incomplete-answer-prefix", "--lens-format", "workspace-mean",
            "--grid-root", str(grid), "--capture-root", str(captures),
            "--checkpoint", str(checkpoint)]
    monkeypatch.setattr(command.sys, "argv", args + (["--preflight"] if preflight else []))
    if preflight:
        command.main()
    else:
        with pytest.raises(RuntimeError, match="approved compute allocation"):
            command.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["rows"] == 4 * 42 and plan["source_layers"] == list(range(42))
    assert plan["stream_reduction"] == "mean" and plan["lens"] == "jlens_workspace_mean"
    assert "workspace-jlens-top-object-heatmap" in plan["output"]
    assert calls == [(lens_path, {"validate_values": False})]

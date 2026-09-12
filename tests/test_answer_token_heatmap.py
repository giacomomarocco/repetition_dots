import pytest
import torch

from filler.dsv4.answer_token_heatmap import score_capture, select_positions
from filler.dsv4.lens import DeepseekV4LensWeights
from filler.dsv4.top_object_heatmap import aggregate_rows, position_tick_label


class Tokenizer:
    def decode(self, ids):
        return {0: "</think>", 1: ".", 2: "Answer", 3: ":", 4: " "}[ids[0]]


@pytest.mark.parametrize("k", [0, 20])
def test_complete_suffix_and_zero_filler_boundary(k):
    ids = [0, *([1] * k), 2, 3, 4]
    positions = {"last_question": 0, "fillers": list(range(1, k + 1)), "answer_prompt": k + 3}
    selected = select_positions(Tokenizer(), ids, positions, k)
    assert [p["absolute_position"] for p in selected] == list(range(k + 4))
    assert [p["token"] for p in selected[-3:]] == ["Answer", ":", " "]
    rows = [{"filler_length": k, "position_label": p["label"], "layer": 0,
             "top_token_id": 10, "top_numeric_token_id": 10,
             "targets": {o: {"token_id": 10} for o in ("A", "X", "A+X")}}
            for p in reversed(selected)]
    stats = aggregate_rows(rows, filler_length=k)
    assert stats["positions"] == [p["label"] for p in selected]
    assert [position_tick_label(p) for p in stats["positions"][-3:]] == ["Answer", ":", "space (answer_prompt)"]


def test_refuse_merged_answer_tokens_and_wrong_filler_count():
    positions = {"last_question": 0, "fillers": [], "answer_prompt": 3}
    with pytest.raises(ValueError, match="three distinct tokens"):
        select_positions(Tokenizer(), [0, 2, 4, 3], positions, 0)
    with pytest.raises(ValueError, match="filler positions"):
        select_positions(Tokenizer(), [0, 2, 3, 4], positions, 20)


def score_fixture():
    cell = {"cell_id": "test:00", "panel_id": "test", "split": "test", "row": 0, "col": 0,
            "left_value": 1, "right_value": 2, "target": 3, "filler_length": 0,
            "input_ids": [99, 0, 2, 3, 4], "target_token_ids": {"A": 0, "X": 1, "A+X": 2},
            "positions": [{"label": label, "absolute_position": i + 1} for i, label in enumerate(
                ["last_question", "answer_word", "answer_colon", "answer_prompt"])]}
    weights = DeepseekV4LensWeights(
        hc_head_fn=torch.zeros(1, 2), hc_head_base=torch.zeros(1), hc_head_scale=torch.ones(1),
        norm_weight=torch.ones(2), lm_head_weight=torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
    # Position-dependent winners catch accidentally projecting the final row everywhere.
    states = torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[-1., 0.]]])
    saved = {"states": {layer: states.clone() for layer in range(43)},
             "metadata": {"positions": [1, 2, 3, 4], "cell_id": "test:00", "num_tokens": 5}}
    return cell, saved, {"output_ids": [2]}, weights, torch.tensor([0, 2])


def test_scoring_selects_each_absolute_position_and_checks_native_answer():
    args = score_fixture()
    rows = list(score_capture(*args))
    assert len(rows) == 4 * 43
    assert [rows[i * 43]["top_token_id"] for i in range(4)] == [0, 1, 2, 2]
    assert [rows[i * 43]["top_numeric_token_id"] for i in range(4)] == [0, 0, 2, 2]
    assert all(r["clean_correct"] for r in rows)
    args[2]["output_ids"] = [1]
    with pytest.raises(ValueError, match="differs from native"):
        list(score_capture(*args))


def test_scoring_rejects_partial_and_nonfinite_captures():
    args = score_fixture()
    args[1]["states"].pop(42)
    with pytest.raises(ValueError, match="missing captured layers"):
        list(score_capture(*args))
    args = score_fixture()
    args[1]["states"][10][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        list(score_capture(*args))


def test_scoring_full_prompt_capture_matches_selected_position_capture():
    cell, saved, response, weights, numeric_ids = score_fixture()
    selected = list(score_capture(cell, saved, response, weights, numeric_ids))
    # Historical baselines also contain earlier prompt positions not plotted.
    saved["states"] = {layer: torch.cat([torch.zeros(1, 1, 2), state])
                       for layer, state in saved["states"].items()}
    saved["metadata"]["positions"] = [0, 1, 2, 3, 4]
    full = list(score_capture(cell, saved, response, weights, numeric_ids))
    assert full == selected


def test_sglang_projection_backend_preserves_batch_and_stream_shapes(monkeypatch):
    import sys
    from types import SimpleNamespace
    from filler.dsv4.lens import collapse_mhc, project_logits, project_sglang_logits

    torch.manual_seed(2)
    weights = DeepseekV4LensWeights(torch.randn(2, 6), torch.randn(2), torch.randn(1),
                                   torch.ones(3), torch.randn(12, 3))
    calls = []
    def fused(x, fn, scale, base, **kwargs):
        calls.append(tuple(x.shape))
        return collapse_mhc(x, weights)
    def norm(x, weight, eps):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.mhc_head", SimpleNamespace(fused_hc_head=fused))
    monkeypatch.setitem(sys.modules, "sgl_kernel", SimpleNamespace(rmsnorm=norm))
    states = torch.randn(2, 5, 2, 3)
    expected = project_logits(states, weights)
    torch.testing.assert_close(project_sglang_logits(states, weights), expected)
    torch.testing.assert_close(project_sglang_logits(states.flatten(-2), weights), expected)
    torch.testing.assert_close(project_sglang_logits(states[0, 0], weights), expected[0, 0])
    assert calls == [(10, 2, 3), (10, 2, 3), (1, 2, 3)]


def test_score_capture_uses_requested_projector():
    from filler.dsv4.lens import project_logits
    args = score_fixture()
    calls = []
    def projector(states, weights):
        calls.append(tuple(states.shape))
        return project_logits(states, weights)
    assert list(score_capture(*args, projector=projector)) == list(score_capture(*args))
    assert calls == [(43, 1, 2)] * 4

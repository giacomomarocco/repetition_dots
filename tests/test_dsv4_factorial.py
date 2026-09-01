import pytest
import torch

from deepseek_v4_logit_lens import DeepseekV4LensWeights
from dsv4_factorial import (
    candidate_sites, causal_effect, copy_cache_rows, crossed_panels, donor_roles,
    factorial_contrasts, locate_positions, score_numeric_targets, should_run_jlens,
)


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_positions_keep_answer_slot_in_user_turn_before_generation_prefix():
    prompt = "question\n. .\nAnswer:<Assistant></think>"
    ids, positions = locate_positions(
        _CharacterTokenizer(), prompt, question_prefix="question\n",
        filler_text=". .\n", answer_prefix="Answer:",
        generation_prefix="<Assistant></think>",
    )
    assert len(ids) == len(prompt)
    assert positions.last_question == len("question\n") - 1
    assert positions.answer_prompt == len("question\n. .\nAnswer:") - 1
    assert positions.answer_prompt < len(ids) - 1


def test_factorial_panels_rotate_all_targets():
    cells = crossed_panels([("a0", 1), ("a1", 4)], [("x0", 10), ("x1", 20)], split="discovery", kind="one_fact")
    assert sorted(x.target for x in cells) == [11, 14, 21, 24]
    roles = donor_roles(cells)
    assert len(roles) == 16
    for target in {r["target_id"] for r in roles}:
        assert {r["role"] for r in roles if r["target_id"] == target} == {"identity", "left", "right", "both"}


def test_repeated_sums_rejected():
    with pytest.raises(ValueError, match="repeated sums"):
        crossed_panels([("a0", 1), ("a1", 2)], [("x0", 5), ("x1", 4)], split="discovery", kind="one_fact")


def test_exact_numeric_scores():
    torch.manual_seed(2)
    w = DeepseekV4LensWeights(torch.randn(2, 6), torch.randn(2), torch.randn(1), torch.randn(3), torch.randn(7, 3))
    scores = score_numeric_targets(torch.randn(2, 3), w, {"A": 2, "X": 4, "sum": 6})
    assert set(scores) == {"A", "X", "sum"}
    assert all(1 <= row["rank"] <= 7 for row in scores.values())
    assert all(isinstance(row["log_odds_vs_rest"], float) for row in scores.values())


def test_site_selection_uses_panel_means_and_rejects_confirmation():
    rows = [
        {"split": "discovery", "layer": 2, "position_label": "filler_0", "role": "left", "panel_id": p, "target_log_odds_shift": v}
        for p, v in (("p0", 1.0), ("p1", 3.0))
    ]
    assert candidate_sites(rows, top_k=1)[0]["mean_panel_shift"] == 2.0
    rows[0]["split"] = "confirmation"
    with pytest.raises(ValueError, match="discovery"):
        candidate_sites(rows)


class _PackedPool:
    page_size = 2
    start_layer = 0
    def __init__(self):
        self.kv_buffer = [torch.arange(32, dtype=torch.uint8).reshape(2, 16)]
    def get_bytes_per_token(self):
        return 4


def test_packed_fp8_cache_copy_moves_one_complete_record():
    pool = _PackedPool()
    before = pool.kv_buffer[0].clone()
    copy_cache_rows(pool, 0, torch.tensor([1]), torch.tensor([2]))
    torch.testing.assert_close(pool.kv_buffer[0][1, :4], before[0, 4:8])
    torch.testing.assert_close(pool.kv_buffer[0][1, 4:], before[1, 4:])


def test_causal_effect_is_patched_minus_clean():
    clean = {"sum": {"log_odds_vs_rest": -2.0}}
    patched = {"sum": {"log_odds_vs_rest": 1.5}}
    assert causal_effect(clean, patched) == {"sum": 3.5}


def test_factorial_contrasts_and_jlens_gate():
    assert factorial_contrasts({(0, 0): 1, (1, 0): 3, (0, 1): 4, (1, 1): 10}) == {
        "left_main": 4.0, "right_main": 5.0, "interaction": 4.0}
    assert should_run_jlens(-0.1, 0.5)
    assert not should_run_jlens(0.1, 0.5)

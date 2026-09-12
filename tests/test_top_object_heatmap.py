from filler.dsv4.top_object_heatmap import aggregate_rows, canonical_numeric_token_ids
from filler.dsv4.lens import DeepseekV4LensWeights, project_logits, project_selected_logits
import torch


class FakeTokenizer:
    values = ["7", " 12", "01", "-3", "3.0", "x", "<eos>"]
    def __len__(self): return len(self.values)
    def decode(self, ids, skip_special_tokens=False): return self.values[ids[0]]


def test_canonical_numeric_token_ids():
    assert canonical_numeric_token_ids(FakeTokenizer()) == [0, 1]


def test_aggregate_rows_computes_both_argmax_rates():
    base = {"filler_length": 1, "position_label": "filler_0", "layer": 0,
            "clean_correct": True,
            "targets": {"A": {"token_id": 10}, "X": {"token_id": 11}, "A+X": {"token_id": 12}}}
    rows = [dict(base, top_token_id=10, top_numeric_token_id=12),
            dict(base, top_token_id=99, top_numeric_token_id=12)]
    stats = aggregate_rows(rows, filler_length=1)
    assert stats["rates"]["all"]["A"] == [[0.5]]
    assert stats["rates"]["all"]["A+X"] == [[0.0]]
    assert stats["rates"]["numeric"]["A+X"] == [[1.0]]


def test_aggregate_rows_splits_correct_and_wrong():
    base = {"filler_length": 1, "position_label": "filler_0", "layer": 0,
            "targets": {"A": {"token_id": 10}, "X": {"token_id": 11}, "A+X": {"token_id": 12}},
            "top_numeric_token_id": 12}
    rows = [dict(base, clean_correct=True, top_token_id=10),
            dict(base, clean_correct=False, top_token_id=12)]
    correct = aggregate_rows(rows, filler_length=1, correct=True)
    wrong = aggregate_rows(rows, filler_length=1, correct=False)
    assert correct["cohort"] == "correct"
    assert wrong["cohort"] == "wrong"
    assert correct["rates"]["all"]["A"] == [[1.0]]
    assert wrong["rates"]["all"]["A+X"] == [[1.0]]


def test_selected_projection_matches_full_projection_columns():
    torch.manual_seed(0)
    weights = DeepseekV4LensWeights(
        hc_head_fn=torch.randn(2, 6), hc_head_base=torch.randn(2),
        hc_head_scale=torch.randn(2), norm_weight=torch.randn(3),
        lm_head_weight=torch.randn(7, 3),
    )
    states = torch.randn(4, 2, 3)
    ids = torch.tensor([1, 5, 6])
    assert torch.allclose(project_selected_logits(states, weights, ids),
                          project_logits(states, weights)[:, ids])

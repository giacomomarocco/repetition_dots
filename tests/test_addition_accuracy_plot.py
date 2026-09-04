import json

from filler.addition.accuracy_plot import (
    load_accuracy,
    load_paired_accuracy_changes,
    wilson_interval,
)


def test_wilson_interval_contains_observed_accuracy():
    low, high = wilson_interval(9, 130)
    assert low < 9 / 130 < high


def test_completed_one_fact_summary_is_loaded_in_length_order():
    points = load_accuracy(
        __import__("pathlib").Path(
            "runs/deepseek-v4-flash/one-fact-addition-full-batched/summary.json"
        )
    )
    assert [point["filler_length"] for point in points] == [0, 10, 20, 50, 100]
    assert all(point["count"] == 262 for point in points)


def test_paired_accuracy_changes_count_transitions(tmp_path):
    results = [
        {"condition": "baseline", "pair_id": "a", "correct": False},
        {"condition": "baseline", "pair_id": "b", "correct": True},
        {"condition": "baseline", "pair_id": "c", "correct": False},
        {"condition": "dots_10", "pair_id": "a", "correct": True},
        {"condition": "dots_10", "pair_id": "b", "correct": False},
        {"condition": "dots_10", "pair_id": "c", "correct": True},
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))

    [point] = load_paired_accuracy_changes(path, n_bootstrap=100, seed=1)

    assert point["accuracy_change"] == 1 / 3
    assert point["wrong_to_right"] == 2
    assert point["right_to_wrong"] == 1
    assert point["unchanged"] == 0

"""Numerical winners, paired cohorts and complete notebook-position coverage."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from filler.dsv4.five_shot_jlens_heatmap import (
    OBJECTS, argmax_ids, heatmap_stats, numeric_ids, summarize)
from filler.dsv4.lens_positions import prompt_position_labels


def test_numeric_winner_outside_full_top_ten_and_exact_ties():
    tokens = ['word'] * 12 + ['0', ' 12', '01', '-2', '2.0', '999']
    ids = numeric_ids(tokens)
    assert ids == [12, 13, 17]
    logits = torch.tensor([[10.] * 12 + [2., 3., 9., 9., 9., 3.],
                           [0.] * 12 + [4., 2., 0., 0., 0., 1.]])
    full, number = argmax_ids(logits, torch.tensor(ids))
    assert full.tolist() == [0, 12]
    assert number.tolist() == [13, 12]
    with pytest.raises(ValueError):
        argmax_ids(logits, torch.tensor([17, 13, 12]))
    logits[0, 0] = float('nan')
    with pytest.raises(ValueError):
        argmax_ids(logits, torch.tensor(ids))


def fixture_records():
    records, arrays = [], []
    for k in (0, 20):
        for i, correct in enumerate((True, False)):
            cid = f'fact{i}-k{k}'
            positions = prompt_position_labels(k, suffix='five_shot')
            records.append({'correct': correct, 'cell': {'cell_id': cid, 'filler_length': k,
                'positions': [{'label': p} for p in positions], 'target_token_ids': dict(zip(OBJECTS, (1, 2, 3)))}})
            arrays.append({'cell_id': cid, 'config_hash': 'fixture',
                'top_token_id': np.full((42, len(positions)), 3 if correct else 0, dtype=np.int32),
                'top_numeric_token_id': np.full((42, len(positions)), 3 if correct else 2, dtype=np.int32),
                'logit_top_token_id': np.full((43, len(positions)), 1, dtype=np.int32),
                'logit_top_numeric_token_id': np.full((43, len(positions)), 2, dtype=np.int32)})
    return records, arrays


@pytest.mark.parametrize('lens,layers', [('square', 42), ('logit', 43)])
def test_summaries_keep_all_positions_and_native_cohorts(lens, layers):
    records, arrays = fixture_records()
    summary, counts = summarize(records, arrays, [1, 2, 3], 'fixture', lens=lens)
    completion = {'cohort_counts': counts}
    for k in (0, 20):
        for cohort in ('all', 'correct', 'wrong'):
            stats = heatmap_stats(summary, completion, k, cohort, lens=lens)
            assert stats['positions'] == prompt_position_labels(k, suffix='five_shot')
            assert np.asarray(stats['rates']['numeric']['A+X']).shape == (k+5, layers)
            assert set(stats['example_counts'].values()) == {2 if cohort == 'all' else 1}
    value = heatmap_stats(summary, completion, 20, 'all', lens=lens)['rates']['numeric']['A+X'][0][0]
    assert value == (.5 if lens == 'square' else 0)
    with pytest.raises(ValueError, match='incomplete/duplicate'):
        heatmap_stats(summary[1:], completion, summary[0]['filler_length'], summary[0]['cohort'], lens=lens)
    changed = deepcopy(completion)
    changed['cohort_counts']['20/all'] += 1
    with pytest.raises(ValueError, match='cohort'):
        heatmap_stats(summary, changed, 20, 'all', lens=lens)


def test_reject_duplicate_missing_and_mislabeled_scores():
    records, arrays = fixture_records()
    with pytest.raises(ValueError):
        summarize(records + records[:1], arrays + arrays[:1], [1, 2, 3], 'fixture')
    with pytest.raises(ValueError):
        summarize(records, arrays[:-1], [1, 2, 3], 'fixture')
    arrays[0]['top_numeric_token_id'] = arrays[0]['top_numeric_token_id'][:, :-1]
    with pytest.raises(ValueError, match='incomplete'):
        summarize(records, arrays, [1, 2, 3], 'fixture')
    records, arrays = fixture_records()
    records[0]['cell']['positions'].pop(1)
    with pytest.raises(ValueError, match='positions'):
        summarize(records, arrays, [1, 2, 3], 'fixture')


def test_notebook_code_compiles_and_has_explicit_dataset_selection():
    root = Path(__file__).resolve().parents[1]
    for name in ('one_fact_top_object_heatmap.ipynb', 'one_fact_top_object_jlens_heatmap.ipynb'):
        nb = json.loads((root / 'notebooks' / name).read_text())
        sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        for source in sources:
            compile(source, name, 'exec')
        joined = '\n'.join(sources)
        assert "DATASET = 'five_shot' if" in joined
        assert 'heatmap_stats(' in joined

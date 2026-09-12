"""Small CPU fixtures for paired lens metrics and result-only visualization."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from filler.dsv4.jlens import JLens, load_jlens
from filler.dsv4.jlens_top_object import compact_metrics, collect_examples, write_scores_many
from filler.dsv4.jlens_comparison import (index_rows, pair_rows, summarize, square_summary,
    compare_baseline, check_baseline_provenance, plot_comparison, write_tables, load_completed, validate_token_metrics)
from test_jlens_top_object import saved_grid, readout


def fixture_rows(tmp_path, monkeypatch=None):
    grid, captures, _ = saved_grid(tmp_path)
    examples, _ = collect_examples(grid, captures)
    rectangular, weights = readout()
    square = JLens({19: torch.eye(3).half(), 39: torch.eye(3).half()}, 25, 3, 3, stream_reduction='mean')
    paths = {n: tmp_path / f'{n}.jsonl' for n in ('square', 'rectangular')}
    hashes = {}
    loads = []
    original = torch.load
    if monkeypatch:
        def counted(*a, **kw):
            loads.append(1)
            return original(*a, **kw)
        monkeypatch.setattr(torch, 'load', counted)
    counts = write_scores_many(examples, {'square': square, 'rectangular': rectangular}, weights,
                              [2, 0, 1], paths, batch_size=3, target_metrics=True, capture_hashes=hashes)
    if monkeypatch:
        assert len(loads) == len(examples)  # one deserialization, both readouts
    assert counts == {'square': 8, 'rectangular': 8}
    rows = {n: [json.loads(line) for line in p.read_text().splitlines()] for n, p in paths.items()}
    indexes = {n: index_rows(rows[n], examples, (19, 39),
                            'jlens_workspace_mean' if n == 'square' else 'jlens', hashes=hashes) for n in rows}
    return examples, hashes, rows, indexes


def test_metric_arithmetic_and_tied_rank_one():
    logits = torch.tensor([[2., 2., -1., 0.], [-1000., -1002., -1001., -999.]])
    targets = [{'A': {'token_id': 1}, 'X': {'token_id': 2}, 'A+X': {'token_id': 3}}] * 2
    scores = compact_metrics(logits, targets, [3, 2, 1, 0], target_metrics=True)
    assert scores[0]['top_token_id'] == 0
    assert scores[0]['targets']['A']['rank'] == 1  # rank 1 but not the selected winner
    assert scores[0]['targets']['X']['rank'] == 4
    assert scores[0]['targets']['A+X']['rank'] == 3
    for i, row in enumerate(scores):
        for name, target in row['targets'].items():
            token = targets[i][name]['token_id']
            assert target['logprob'] == pytest.approx(float(logits[i].log_softmax(-1)[token]), abs=5e-5)
            assert target['rank'] == 1 + sum(float(v) > float(logits[i, token]) for v in logits[i])
    equal = compact_metrics(torch.zeros(1, 4), targets[:1], [3, 2, 1], target_metrics=True)[0]
    assert equal['top_token_id'] == 0 and equal['top_numeric_token_id'] == 1
    assert equal['targets']['A']['logprob'] == pytest.approx(-math.log(4))
    assert all(t['rank'] == 1 for t in equal['targets'].values())


def test_shared_batches_pairing_cohorts_and_differences(tmp_path, monkeypatch):
    _, _, rows, indexes = fixture_rows(tmp_path, monkeypatch)
    paired = list(pair_rows(indexes['square'], indexes['rectangular'], layers=(19, 39)))
    assert len(paired) == 24
    for row in paired:
        assert row['square_mrr'] == 1 / row['square_rank']
        for metric in ('top1', 'numeric_top1', 'mrr', 'logprob'):
            assert row[f'difference_{metric}'] == row[f'square_{metric}'] - row[f'rectangular_{metric}']
    stats = summarize(paired)
    assert {r['n'] for r in stats if r['cohort'] == 'all'} == {2}
    assert {r['n'] for r in stats if r['cohort'] != 'all'} == {1}
    all_a = next(r for r in stats if r['cohort'] == 'all' and r['target_name'] == 'A'
                 and r['layer'] == 39 and r['position_label'] == 'last_question')
    assert all_a['square_numeric_top1'] == .5
    assert all_a['rectangular_numeric_top1'] == .5
    assert all_a['numeric_square_only'] == 1 and all_a['numeric_rectangular_only'] == 1
    assert all_a['numeric_agreement'] == 0
    assert compare_baseline(indexes['rectangular'], indexes['rectangular'])['passed']
    changed = deepcopy(indexes['rectangular'])
    next(iter(changed.values()))['top_token_id'] = 0
    check = compare_baseline(indexes['rectangular'], changed)
    assert not check['passed'] and len(check['mismatches']) == 1
    with pytest.raises(ValueError, match='duplicate'):
        summarize(paired + paired[:1])


@pytest.mark.parametrize('issue', ['duplicate', 'missing', 'target', 'hash', 'cohort', 'rank', 'probability', 'identity'])
def test_invalid_rows_fail(tmp_path, issue):
    examples, hashes, rows, _ = fixture_rows(tmp_path)
    square = deepcopy(rows['square'])
    if issue == 'duplicate': square.append(square[0])
    if issue == 'missing': square.pop()
    if issue == 'target': square[0]['targets']['A']['token_id'] = 99
    if issue == 'hash': square[0]['capture_sha256'] = 'wrong'
    if issue == 'cohort': square[0]['clean_correct'] = 1
    if issue == 'rank': square[0]['targets']['A']['rank'] = 0
    if issue == 'probability': square[0]['targets']['A']['logprob'] = float('nan')
    if issue == 'identity': square[0]['pass_id'] = 100
    with pytest.raises(ValueError):
        index_rows(square, examples, (19, 39), 'jlens_workspace_mean', hashes=hashes)


@pytest.mark.parametrize('issue', ['missing', 'target', 'hash', 'identity'])
def test_pair_mismatch_rejected(tmp_path, issue):
    _, _, _, indexes = fixture_rows(tmp_path)
    rectangular = deepcopy(indexes['rectangular'])
    row = next(iter(rectangular.values()))
    if issue == 'missing': rectangular.pop(next(iter(rectangular)))
    if issue == 'target': row['targets']['A']['token_id'] = 99
    if issue == 'hash': row['capture_sha256'] = 'wrong'
    if issue == 'identity': row['pass_id'] = 100
    with pytest.raises(ValueError):
        list(pair_rows(indexes['square'], rectangular, layers=(19, 39)))


@pytest.mark.parametrize('field', ['manifest_sha256', 'numeric_token_ids', 'norm_rounding',
    'transport_dtype', 'lens_sha256', 'checkpoint', 'capture_root', 'grid_root', 'batch_size', 'threads', 'device'])
def test_incompatible_baseline_provenance(field):
    baseline = {'manifest_sha256': {'a': 'abc'}, 'numeric_token_ids': [0, 1],
                'norm_rounding': 'published HF', 'transport_dtype': 'float32', 'lens_sha256': '123',
                'arguments': {'checkpoint': '/tmp/checkpoint', 'capture_root': '/tmp/captures',
                              'grid_root': '/tmp/grid', 'batch_size': '64', 'threads': '16', 'device': 'cpu'}}
    check_baseline_provenance(baseline, baseline)
    changed = deepcopy(baseline)
    (changed if field in changed else changed['arguments'])[field] = 'different'
    with pytest.raises(ValueError, match='provenance'):
        check_baseline_provenance(baseline, changed)


def test_plots_tables_empty_cohort_and_completed_guard(tmp_path):
    import matplotlib.pyplot as plt
    _, _, _, indexes = fixture_rows(tmp_path)
    paired = list(pair_rows(indexes['square'], indexes['rectangular'], layers=(19, 39)))
    stats = summarize(paired)
    for metric in ('top1', 'numeric_top1', 'mrr', 'logprob'):
        fig = plot_comparison(stats, 0, 'all', metric)
        images = [ax.images[0] for ax in fig.axes if ax.images]
        assert len(images) == 9
        assert images[0].get_clim() == images[1].get_clim() == images[3].get_clim()
        lo, hi = images[2].get_clim()
        assert lo == -hi and images[2].norm(0) == .5
        if metric == 'top1':
            fig.savefig(tmp_path / 'test.png')
            fig.savefig(tmp_path / 'test.pdf')
        plt.close(fig)
    empty = plot_comparison(stats, 20, 'wrong')
    assert any('No wrong examples' in t.get_text() for ax in empty.axes for t in ax.texts)
    plt.close(empty)
    fig = plot_comparison(square_summary(indexes['square']), 0, 'correct', square_only=True)
    assert len([ax for ax in fig.axes if ax.images]) == 3
    plt.close(fig)
    write_tables(tmp_path / 'summary', stats)
    write_tables(tmp_path / 'paired', paired)
    write_tables(tmp_path / 'square_summary', square_summary(indexes['square']))
    assert json.loads((tmp_path / 'paired.json').read_text()) == paired
    (tmp_path / 'COMPLETE.json').write_text('{"passed": true, "executed_notebook": "executed.ipynb"}')
    with pytest.raises(ValueError, match='notebook export'):
        load_completed(tmp_path)
    (tmp_path / 'executed.ipynb').write_text('{}')
    assert load_completed(tmp_path)[0] == stats
    (tmp_path / 'FAILED.json').write_text('{}')
    with pytest.raises(ValueError, match='completed'):
        load_completed(tmp_path)


def test_rectangular_metadata_defers_values_only(tmp_path):
    path = tmp_path / 'rect.pt'
    torch.save({'J': {19: torch.full((3, 12), float('nan')).half()}, 'n_prompts': 1000,
                'd_model': 3, 'd_source': 12, 'source_layers': [19]}, path)
    assert load_jlens(path, expected_shape=(3, 12), expected_layers=(19,), validate_values=False).source_layers == (19,)
    with pytest.raises(ValueError, match='nonfinite'):
        load_jlens(path, expected_shape=(3, 12), expected_layers=(19,))


def test_cli_preflight_and_compute_guard(tmp_path, monkeypatch):
    from filler.dsv4 import compare_jlenses as command
    monkeypatch.setattr(command, 'preflight', lambda args: ([], {}, {'metadata_only': True}))
    monkeypatch.setattr(command, 'load_checkpoint_readout', lambda *a, **kw: pytest.fail('weights loaded on login node'))
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    command.main(['--legacy-grid', '--preflight', '--output-dir', str(tmp_path / 'new')])
    with pytest.raises(RuntimeError, match='approved compute allocation'):
        command.main(['--legacy-grid', '--output-dir', str(tmp_path / 'new')])
    assert not (tmp_path / 'new').exists()


def test_executed_notebook_embeds_default_figure_and_exact_example(tmp_path):
    from filler.dsv4.compare_jlenses import execute_notebook
    stats, pairs = [], []
    for position in ['last_question', *[f'filler_{i}' for i in range(20)], 'answer_prompt']:
        for target in ('A', 'X', 'A+X'):
            row = {'filler_length': 20, 'cohort': 'all', 'position_label': position,
                   'layer': 39, 'target_name': target, 'n': 1}
            for lens in ('square', 'rectangular', 'difference'):
                for metric in ('top1', 'numeric_top1', 'mrr', 'logprob'):
                    row[f'{lens}_{metric}'] = 0.0
            for scope in ('full', 'numeric'):
                for metric in ('agreement', 'square_only', 'rectangular_only'):
                    row[f'{scope}_{metric}'] = 0
            stats.append(row)
            pair = {**row, 'cell_id': 'synthetic', 'clean_correct': True, 'target_token_text': '1'}
            for lens in ('square', 'rectangular'):
                pair[f'{lens}_rank'] = 2
                pair[f'{lens}_logprob'] = -2.0
                pair[f'{lens}_top_token_text'] = ' 2'
                pair[f'{lens}_top_numeric_token_text'] = '2'
            pairs.append(pair)
    write_tables(tmp_path / 'summary', stats)
    write_tables(tmp_path / 'square_summary', stats)
    write_tables(tmp_path / 'paired', pairs)
    write_tables(tmp_path / 'overview_k20_all', [{'target': 'A', 'pairs': 22}])
    (tmp_path / 'validation.json').write_text('{"passed":true}')
    execute_notebook(tmp_path)
    nb = json.loads((tmp_path / 'one_fact_jlens_comparison.executed.ipynb').read_text())
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert all(c['execution_count'] for c in code_cells)
    outputs = [o for c in code_cells for o in c['outputs']]
    assert any('image/png' in o.get('data', {}) for o in outputs)
    assert any('square_rank' in o.get('data', {}).get('text/html', '') for o in outputs)
    assert not any(o['output_type'] == 'error' for o in outputs)
    assert not (tmp_path / 'COMPLETE.json').exists()


@pytest.mark.parametrize('issue', [None, 'rank', 'winner', 'numeric', 'probability'])
def test_cached_vocabulary_validation(tmp_path, issue):
    _, _, _, indexes = fixture_rows(tmp_path)
    rows = indexes['square']
    row = next(iter(rows.values()))
    if issue == 'rank': row['targets']['A']['rank'] = 6
    if issue == 'winner': row['top_token_id'] = 5
    if issue == 'numeric': row['top_numeric_token_id'] = 4
    if issue == 'probability': row['targets']['A']['logprob'] = float('inf')
    # The interface accepts an integer size; there is no tokenizer dependency
    # and therefore no vocabulary reconstruction in the per-row loop.
    if issue is None:
        validate_token_metrics(rows, vocab_size=5, numeric_ids=[0, 1, 2])
    else:
        with pytest.raises(ValueError):
            validate_token_metrics(rows, vocab_size=5, numeric_ids=[0, 1, 2])

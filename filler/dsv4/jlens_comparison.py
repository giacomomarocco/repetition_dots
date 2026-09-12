"""Strict pairing, descriptive summaries and saved-result views for two J-Lenses.

This module never loads models or captures. Notebook users read completed exports.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

OBJECTS = ('A', 'X', 'A+X')
METRICS = ('top1', 'numeric_top1', 'mrr', 'logprob')
COHORTS = ('all', 'correct', 'wrong')
CAPTURE_IDENTITY = ('capture_format', 'runtime_id', 'request_id', 'capture_config_hash',
                    'input_ids_sha256', 'num_tokens', 'position_token_id', 'position_token')
KEY = ('filler_length', 'cell_id', 'position_label', 'layer')
IDENTITY = ('panel_id', 'split', 'row', 'col', 'left_value', 'right_value', 'target',
            'clean_correct', 'absolute_position', 'pass_id')
LIMITATION = ('The published artifacts have different fitting recipes, prompt counts, '
              'target bases and model/backend provenance. Differences cannot be attributed '
              'solely to averaging versus flattening. These are descriptive readouts, '
              'not causal effects or evidence of backend equivalence.')


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def key(row):
    return tuple(row[name] for name in KEY)


def index_rows(rows, examples, layers, lens_name, *, metrics=True, hashes=None):
    """Reject duplicates, missing cells and any disagreement with saved manifests."""
    expected = {tuple(e[n] for n in KEY[:-1]): e for e in examples}
    if len(expected) != len(examples):
        raise ValueError('duplicate expected captures')
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f'duplicate row: {k}')
        if k[:-1] not in expected or k[-1] not in layers or row['lens'] != lens_name:
            raise ValueError(f'incompatible row identity: {k}')
        example = expected[k[:-1]]
        for name in (*IDENTITY, *(n for n in CAPTURE_IDENTITY if n in example)):
            if row[name] != example[name] or type(row[name]) is not type(example[name]):
                raise ValueError(f'row/manifest mismatch: {k}, {name}')
        if type(row['clean_correct']) is not bool:
            raise ValueError('cohort requires saved boolean correctness')
        if set(row['targets']) != set(OBJECTS):
            raise ValueError('missing target labels')
        for name in OBJECTS:
            target = row['targets'][name]
            if target['token_id'] != example['targets'][name]['token_id']:
                raise ValueError(f'target ID mismatch: {k}, {name}')
            if metrics and (type(target.get('rank')) is not int or target['rank'] < 1
                            or not math.isfinite(target.get('logprob', float('nan')))
                            or target['logprob'] > 0):
                raise ValueError(f'invalid target metrics: {k}, {name}')
        if hashes is not None and row.get('capture_sha256') != hashes[example['capture_path']]:
            raise ValueError(f'capture hash mismatch: {k}')
        result[k] = row
    if len(result) != len(examples) * len(layers):
        raise ValueError('missing rows in complete capture/layer grid')
    return result


def check_baseline_provenance(baseline, current):
    for name in ('manifest_sha256', 'numeric_token_ids', 'transport_dtype', 'norm_rounding', 'lens_sha256'):
        if baseline.get(name) != current.get(name) or name not in current:
            raise ValueError(f'incompatible baseline provenance: {name}')
    for name in ('checkpoint', 'capture_root', 'grid_root'):
        if Path(baseline['arguments'][name]).resolve() != Path(current['arguments'][name]).resolve():
            raise ValueError(f'incompatible baseline provenance: {name}')
    for name in ('batch_size', 'threads', 'device'):
        if str(baseline['arguments'][name]) != str(current['arguments'][name]):
            raise ValueError(f'incompatible baseline provenance: {name}')


def validate_token_metrics(rows, *, vocab_size, numeric_ids):
    """Validate compact rows with cached vocabulary metadata, never tokenizer calls."""
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('invalid vocabulary size')
    numeric = set(numeric_ids)
    for row in rows.values():
        if row['top_numeric_token_id'] not in numeric or not 0 <= row['top_token_id'] < vocab_size:
            raise ValueError('invalid winning token ID')
        if any(type(t['rank']) is not int or not 1 <= t['rank'] <= vocab_size
               or not math.isfinite(t['logprob']) or t['logprob'] > 0
               for t in row['targets'].values()):
            raise ValueError('invalid target metrics or rank outside vocabulary')


def compare_baseline(rectangular, baseline):
    if rectangular.keys() != baseline.keys():
        raise ValueError('baseline capture/layer keys differ')
    mismatches = []
    for k, row in rectangular.items():
        old = baseline[k]
        for name in ('top_token_id', 'top_numeric_token_id'):
            if row[name] != old[name]:
                mismatches.append({'key': list(k), 'scope': name,
                                   'baseline': old[name], 'rescored': row[name]})
    return {'passed': not mismatches, 'rows': len(rectangular), 'mismatches': mismatches}


def pair_rows(square, rectangular, *, layers=tuple(range(19, 40))):
    matched = {k for k in square if k[-1] in layers}
    if matched != set(rectangular):
        raise ValueError('missing or extra paired rows')
    for k in sorted(matched):
        s, r = square[k], rectangular[k]
        for name in (*IDENTITY, 'capture_sha256', *(n for n in CAPTURE_IDENTITY if n in s or n in r)):
            if name not in s or s[name] != r.get(name):
                raise ValueError(f'incompatible paired identity/provenance: {k}, {name}')
        for target in OBJECTS:
            token = s['targets'][target]['token_id']
            if token != r['targets'][target]['token_id']:
                raise ValueError(f'paired target ID mismatch: {k}')
            row = {name: s[name] for name in (*KEY, *IDENTITY, 'capture_sha256')}
            row.update({name: s[name] for name in CAPTURE_IDENTITY if name in s})
            row.update(target_name=target, target_token_id=token)
            for name, source in (('square', s), ('rectangular', r)):
                score = source['targets'][target]
                row.update({f'{name}_top_token_id': source['top_token_id'],
                            f'{name}_top_numeric_token_id': source['top_numeric_token_id'],
                            f'{name}_rank': score['rank'], f'{name}_logprob': score['logprob'],
                            f'{name}_mrr': 1.0 / score['rank'],
                            f'{name}_top1': int(source['top_token_id'] == token),
                            f'{name}_numeric_top1': int(source['top_numeric_token_id'] == token)})
            for metric in METRICS:
                row[f'difference_{metric}'] = row[f'square_{metric}'] - row[f'rectangular_{metric}']
            for scope, winner in (('full', 'top_token_id'), ('numeric', 'top_numeric_token_id')):
                row[f'{scope}_agreement'] = int(s[winner] == r[winner])
                row[f'{scope}_square_only'] = int(s[winner] == token and r[winner] != token)
                row[f'{scope}_rectangular_only'] = int(r[winner] == token and s[winner] != token)
            yield row


def summarize(paired):
    """Equal weight per saved example within each position/layer/cohort."""
    groups = {}
    value_names = None
    seen = set()
    for row in paired:
        k = (*key(row), row['target_name'])
        if k in seen:
            raise ValueError('duplicate paired target row')
        seen.add(k)
        if type(row['clean_correct']) is not bool:
            raise ValueError('invalid correctness label')
        if value_names is None:
            value_names = [f'{lens}_{metric}' for lens in ('square', 'rectangular', 'difference') for metric in METRICS]
            value_names += [f'{scope}_{metric}' for scope in ('full', 'numeric')
                            for metric in ('agreement', 'square_only', 'rectangular_only')]
        for cohort in ('all', 'correct' if row['clean_correct'] else 'wrong'):
            group_key = (row['filler_length'], cohort, row['position_label'], row['layer'], row['target_name'])
            if group_key not in groups:
                groups[group_key] = {'n': 0, **dict.fromkeys(value_names, 0.0)}
            group = groups[group_key]
            group['n'] += 1
            for name in value_names:
                group[name] += row[name]
    result = []
    for k, values in sorted(groups.items()):
        row = dict(zip(('filler_length', 'cohort', 'position_label', 'layer', 'target_name'), k))
        row['n'] = values['n']
        for name in value_names:
            row[name] = int(values[name]) if name.endswith('_only') else values[name] / values['n']
        result.append(row)
    # Every position/layer must use the same saved cohort denominator.
    denominators = defaultdict(set)
    for row in result:
        denominators[row['filler_length'], row['cohort']].add(row['n'])
    if any(len(counts) != 1 for counts in denominators.values()):
        raise ValueError('inconsistent cohort denominators; missing paired rows')
    return result


def square_summary(square):
    groups = {}
    for row in square.values():
        for cohort in ('all', 'correct' if row['clean_correct'] else 'wrong'):
            for target in OBJECTS:
                k = (row['filler_length'], cohort, row['position_label'], row['layer'], target)
                group = groups.setdefault(k, {'n': 0, **dict.fromkeys(METRICS, 0.0)})
                score = row['targets'][target]
                group['n'] += 1
                group['top1'] += row['top_token_id'] == score['token_id']
                group['numeric_top1'] += row['top_numeric_token_id'] == score['token_id']
                group['mrr'] += 1.0 / score['rank']
                group['logprob'] += score['logprob']
    return [{**dict(zip(('filler_length', 'cohort', 'position_label', 'layer', 'target_name'), k)),
             'n': v['n'], **{f'square_{m}': v[m] / v['n'] for m in METRICS}}
            for k, v in sorted(groups.items())]


def write_tables(stem, rows):
    """CSV plus standard JSON (no NaN); paired rows retain exact per-example ranks."""
    rows = list(rows)
    Path(stem).with_suffix('.json').write_text(json.dumps(rows, allow_nan=False) + '\n')
    with Path(stem).with_suffix('.csv').open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_comparison(summary, filler_length=20, cohort='all', metric='top1', *, square_only=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    if metric not in METRICS or cohort not in COHORTS:
        raise ValueError('unknown metric or cohort')
    selected = [r for r in summary if r['filler_length'] == filler_length and r['cohort'] == cohort]
    with plt.rc_context({'text.usetex': False}):
        if not selected:
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.text(.5, .5, f'No {cohort} examples (filler length {filler_length})', ha='center')
            ax.axis('off')
            return fig
        from filler.dsv4.top_object_heatmap import position_order, position_tick_label
        from filler.dsv4.lens_positions import prompt_position_labels
        positions = sorted({r['position_label'] for r in selected}, key=lambda p: position_order(p, filler_length))
        legacy = not any(p in positions for p in ('answer_word', 'answer_colon'))
        if positions != prompt_position_labels(filler_length, legacy=legacy):
            raise ValueError('incomplete prompt position coverage in plotted data')
        layers = sorted({r['layer'] for r in selected})
        lookup = {(r['target_name'], r['position_label'], r['layer']): r for r in selected}
        columns = ('square',) if square_only else ('square', 'rectangular', 'difference')
        arrays = {(target, name): np.array([[lookup[target, p, l][f'{name}_{metric}'] for l in layers]
                                           for p in positions]) for target in OBJECTS for name in columns}
        absolute = np.concatenate([v.ravel() for (_, n), v in arrays.items() if n != 'difference'])
        lo, hi = (0., 1.) if metric != 'logprob' else (float(absolute.min()), float(absolute.max()))
        if lo == hi:
            lo, hi = lo - .5, hi + .5
        limit = max([float(np.abs(v).max()) for (_, n), v in arrays.items() if n == 'difference'] + [1e-8])
        fig, axes = plt.subplots(3, len(columns), figsize=(6 * len(columns), 12),
                                 squeeze=False, constrained_layout=True)
        for i, target in enumerate(OBJECTS):
            for j, name in enumerate(columns):
                axis = axes[i, j]
                kwargs = {'cmap': 'coolwarm', 'norm': TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)} if name == 'difference' else {'cmap': 'magma', 'vmin': lo, 'vmax': hi}
                chart = axis.imshow(arrays[target, name], origin='lower', aspect='auto', interpolation='nearest', **kwargs)
                axis.set_title(f'{target}: ' + ('square minus rectangular' if name == 'difference' else name))
                axis.set_yticks(range(len(positions)), [position_tick_label(p) for p in positions], fontsize=8)
                ticks = range(0, len(layers), max(1, len(layers) // 8))
                axis.set_xticks(list(ticks), [layers[t] for t in ticks])
                axis.set_xlabel('Post-block layer (zero-based)')
                fig.colorbar(chart, ax=axis, shrink=.8)
        n = selected[0]['n']
        fig.suptitle(f'J-Lens {metric} | filler {filler_length} | {cohort} | n={n} examples per cell')
        return fig


def load_completed(run, *, allow_validated=False):
    run = Path(run)
    marker = run / 'COMPLETE.json'
    if allow_validated and not marker.exists():
        marker = run / 'validation.json'
    complete = json.loads(marker.read_text())
    if complete.get('passed') is not True or (run / 'FAILED.json').exists():
        raise ValueError('comparison has not completed successfully')
    if not allow_validated and not (run / complete.get('executed_notebook', '__missing_notebook__')).is_file():
        raise ValueError('comparison has not completed notebook export')
    return json.loads((run / 'summary.json').read_text()), json.loads((run / 'square_summary.json').read_text())


def inspect_example(run, filler_length, cell_id, position_label=None, layer=None):
    import pandas as pd
    # CSV reading keeps notebook inspection independent of Torch/tokenizer imports.
    rows = []
    with (Path(run) / 'paired.csv').open() as handle:
        for row in csv.DictReader(handle):
            if (int(row['filler_length']) == filler_length and row['cell_id'] == cell_id
                    and (position_label is None or row['position_label'] == position_label)
                    and (layer is None or int(row['layer']) == layer)):
                rows.append(row)
    return pd.DataFrame(rows)

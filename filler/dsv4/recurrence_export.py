"""Validated tables, figures and artifact-only notebook for five-shot recurrence."""
from __future__ import annotations

import base64
import contextlib
import csv
from datetime import datetime, timezone
import gzip
import io
from pathlib import Path

import numpy as np

from filler.dsv4.five_shot_recurrence import NOTEBOOK, ROOT, read, verify_plan
from filler.dsv4.patching import atomic_json, file_digest
from filler.dsv4.recurrence_analysis import (
    REGIONS, annotation_matrix, bootstrap_weights, grouped_top, interval, load_scores,
    region_indices, representative, subsets, summary_arrays)

LIMITATION = ('These are recurring lens readouts, not generated sentences or evidence of causal computation. '
              'The k20-minus-k0 contrast changes filler length in all five demonstrations and the target, '
              'so it describes the complete prompt condition. Intervals are descriptive paired-fact bootstrap '
              'intervals (2,000 resamples, seed 42), not simultaneous or post-selection inference. '
              'Words/wordpieces strip surrounding whitespace and case-fold; they are not reconstructed words.')
LAB_LOG = ROOT / 'FIVE_SHOT_SQUARE_RECURRENCE.md'


def save_csv(path, rows, fields=None):
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None and fields is None:
        raise ValueError('cannot infer empty table columns')
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=fields or list(first))
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(iterator)


def summary_rows(data, vocab, weights):
    """All observed IDs/groups, cohort/category summaries with fact-level intervals."""
    for k, block in data.items():
        records = block['records']
        for kind in ('token', 'group'):
            top = block['top_ids'] if kind == 'token' else grouped_top(block['top_ids'], vocab['token_group'])
            for region in REGIONS:
                pos = region_indices(records[0]['cell']['positions'], region)
                if not pos:
                    continue
                items, metrics, cells = summary_arrays(top, pos)
                annotations = {scope: annotation_matrix(records, vocab, items, scope, kind)
                               for scope in ('demonstration', 'question')}
                for category, cohort, mask in subsets(records):
                    n = int(mask.sum())
                    if not n:
                        continue  # Explicit empty cohorts are retained in cohorts.csv.
                    w = weights * mask[None, :]
                    denom = w.sum(-1)
                    valid = denom > 0
                    for start in range(0, len(items), 256):
                        stop = min(start+256, len(items))
                        means, cis = {}, {}
                        for metric, values in metrics.items():
                            batch = values[:, start:stop]
                            means[metric] = np.asarray(batch[mask].mean(0)).ravel()
                            samples = (batch.T @ w[valid].T).T / denom[valid, None]
                            cis[metric] = np.quantile(samples, [.025, .975], axis=0)
                        for j, item in enumerate(items[start:stop]):
                            index = start+j
                            text = vocab['tokens' if kind == 'token' else 'groups'][item]
                            group = vocab['token_group'][item] if kind == 'token' else item
                            row = {'k': k, 'region': region, 'category': category, 'cohort': cohort,
                                'kind': kind, 'item_id': int(item), 'text': text, 'filter': vocab['group_kind'][group],
                                'n_prompts': n, 'cells_per_prompt': cells, 'bootstrap_nonempty': int(valid.sum()),
                                'in_demonstrations_fraction': float(annotations['demonstration'][mask, index].mean()),
                                'in_question_fraction': float(annotations['question'][mask, index].mean())}
                            for metric in metrics:
                                row[metric] = float(means[metric][j])
                                row[metric+'_low'], row[metric+'_high'] = map(float, cis[metric][:, j])
                            yield row
                print(f'Summary k{k} {kind} {region}: {len(items)} items', flush=True)


def cell_frequencies(top, mask, vocab_size):
    """Sparse exact per-cell counts: code = cell_offset * vocabulary_size + ID."""
    x = top[mask]
    if not len(x):
        return {'code': np.empty(0, np.int64), 'top10_count': np.empty(0, np.int32),
                'top1_count': np.empty(0, np.int32), 'n': np.int32(0)}
    cell = np.arange(x.shape[1]*x.shape[2]).reshape(1, x.shape[1], x.shape[2], 1)
    codes = cell*vocab_size + x
    selected = x >= 0
    keys, counts = np.unique(codes[selected], return_counts=True)
    first, first_counts = np.unique(codes[..., 0], return_counts=True)
    top1 = np.zeros(len(keys), np.int32)
    top1[np.searchsorted(keys, first)] = first_counts
    return {'code': keys, 'top10_count': counts.astype(np.int32), 'top1_count': top1, 'n': np.int32(len(x))}


def export_frequencies(output, data, vocab):
    root = output / 'frequencies'
    root.mkdir(exist_ok=True)
    manifest = []
    for k, block in data.items():
        for kind in ('token', 'group'):
            top = block['top_ids'] if kind == 'token' else grouped_top(block['top_ids'], vocab['token_group'])
            size = len(vocab['tokens' if kind == 'token' else 'groups'])
            for basis in ('condition', 'paired_k20'):
                cohort_records = block['records'] if basis == 'condition' else data[20]['records']
                for category, cohort, mask in subsets(cohort_records):
                    path = root / f'{basis}_k{k}_{kind}_{category}_{cohort}.npz'
                    np.savez_compressed(path, **cell_frequencies(top, mask, size), vocab_size=np.int32(size),
                                        shape=np.array(top.shape[1:3], np.int32))
                    manifest.append({'k': k, 'kind': kind, 'category': category, 'cohort': cohort, 'basis': basis,
                                     'path': str(path.relative_to(output)), 'sha256': file_digest(path)})
    atomic_json(output / 'frequencies.json', manifest)


def frequencies(output, k, item_id, *, kind='group', cohort='all', category='all', metric='top10', basis='condition'):
    output = Path(output)
    entries = read(output / 'frequencies.json')
    entry = next(r for r in entries if (r['k'], r['kind'], r['category'], r['cohort'], r['basis']) == (k, kind, category, cohort, basis))
    if file_digest(output / entry['path']) != entry['sha256']:
        raise ValueError('frequency artifact checksum mismatch')
    with np.load(output / entry['path'], allow_pickle=False) as x:
        shape, n = tuple(x['shape']), int(x['n'])
        result = np.full(shape, np.nan) if not n else np.zeros(shape)
        if n:
            mask = x['code'] % int(x['vocab_size']) == item_id
            result.ravel()[x['code'][mask] // int(x['vocab_size'])] = x[metric+'_count'][mask]/n
        return result


def shared(block):
    return [i for i, p in enumerate(block['records'][0]['cell']['positions']) if not p['label'].startswith('filler_')]


def paired_rows(data, mass, groups, vocab, weights):
    """k20 membership defines correct/wrong paired cohorts; both outcomes retained."""
    grouped = {k: grouped_top(v['top_ids'], vocab['token_group']) for k, v in data.items()}
    records = data[20]['records']
    labels = [data[0]['records'][0]['cell']['positions'][p]['label'] for p in shared(data[0])]
    for gi, group in enumerate(groups):
        gid = group['group_id']
        for region, indices in [('shared_all', list(range(5))), ('question', [0]),
                                ('answer_prefix', [1, 2]), ('assistant_transition', [3, 4]),
                                *[(label, [i]) for i, label in enumerate(labels)]]:
            for metric in ('top1', 'top10', 'probability_mass'):
                values = {}
                for k in (0, 20):
                    pos = np.asarray(shared(data[k]))[indices]
                    if metric == 'probability_mass':
                        x = np.exp(mass[k]['group_logmass'][:, :, pos, gi].astype(np.float64))
                    else:
                        x = grouped[k][:, :, pos, :1 if metric == 'top1' else 10]
                        x = (x == gid).any(-1)
                    values[k] = x.mean(axis=(1, 2))
                delta = values[20] - values[0]
                for category, cohort, mask in subsets(records):
                    if not mask.any():
                        continue
                    low, high = interval(delta, weights, mask)[:, 0]
                    yield {'wordpiece': group['wordpiece'], 'group_id': gid, 'region': region, 'metric': metric,
                           'category': category, 'cohort_at_k20': cohort, 'n_pairs': int(mask.sum()),
                           'k0': float(values[0][mask].mean()), 'k20': float(values[20][mask].mean()),
                           'difference': float(delta[mask].mean()), 'low': float(low), 'high': float(high)}


def probability_rows(data, mass, groups, weights):
    for k, block in data.items():
        for region in REGIONS:
            pos = region_indices(block['records'][0]['cell']['positions'], region)
            if not pos:
                continue
            values = np.exp(mass[k]['group_logmass'][:, :, pos].astype(np.float64)).mean((1, 2))
            for category, cohort, mask in subsets(block['records']):
                if not mask.any():
                    continue
                cis = interval(values, weights, mask)
                for j, group in enumerate(groups):
                    yield {'k': k, 'region': region, 'category': category, 'cohort': cohort,
                           'wordpiece': group['wordpiece'], 'group_id': group['group_id'], 'n_prompts': int(mask.sum()),
                           'probability_mass': float(values[mask, j].mean()), 'low': float(cis[0, j]), 'high': float(cis[1, j])}


def paired_heatmap(k0, k20, labels0, labels20, title, *, annotate=None, probability=False):
    import matplotlib.pyplot as plt
    shared20 = [i for i, label in enumerate(labels20) if not label.startswith('filler_')]
    delta = k20[:, shared20] - k0
    fig, axes = plt.subplots(1, 3, figsize=(20, 12), gridspec_kw={'width_ratios': [5, 25, 5]}, constrained_layout=True)
    for i, (ax, values, labels, name) in enumerate(zip(axes, (k0, k20, delta), (labels0, labels20, labels0),
                                                               ('k=0', 'k=20', 'k20 − k0 (shared)'))):
        limit = max(float(np.nanmax(np.abs(delta))), .001) if i == 2 and np.isfinite(delta).any() else 1
        im = ax.imshow(values, origin='lower', aspect='auto', interpolation='none',
                       cmap='coolwarm' if i == 2 else 'viridis', vmin=-limit if i == 2 else 0, vmax=limit if i == 2 else 1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(0, 42, 3))
        ax.set_ylabel('Square J-Lens post-block layer')
        ax.set_title(name)
        if i == 1:
            ax.axvline(.5, color='white', lw=1)
            ax.axvline(20.5, color='white', lw=1)
        if annotate is not None and i < 2:
            for layer in range(len(values)):
                for p in range(len(labels)):
                    ax.text(p, layer, annotate[i][layer, p], ha='center', va='center', fontsize=5, color='white')
        fig.colorbar(im, ax=ax, shrink=.5, label='Probability mass' if probability else 'Rate')
    fig.suptitle(title)
    return fig


def labels(block):
    return [p['label'] for p in block['records'][0]['cell']['positions']]


def save_figure(fig, stem):
    import matplotlib.pyplot as plt
    fig.savefig(str(stem)+'.png', dpi=160)
    fig.savefig(str(stem)+'.pdf')
    plt.close(fig)


def plot_word(output, group_id, *, cohort='all', category='all', metric='top10'):
    output = Path(output)
    plan = read(output / 'preflight.json')
    positions = {k: [p['label'] for p in next(c for c in plan['cells'] if c['k'] == k)['positions']] for k in (0, 20)}
    values = {k: frequencies(output, k, group_id, cohort=cohort, category=category, metric=metric, basis='paired_k20') for k in (0, 20)}
    word = read(output / 'vocabulary.json')['groups'][group_id]
    return paired_heatmap(values[0], values[20], positions[0], positions[20], f'{word!r}: {metric}, {category}, {cohort} (cohort fixed at k20)')


def inspect_cell(output, pair_id, k, layer, position):
    import pandas as pd
    output = Path(output)
    records = read(output / 'captures.json')
    record = next(r for r in records if r['cell']['pair_id'] == pair_id and r['cell']['k'] == k)
    cell = record['cell']
    offset = next(i for i, p in enumerate(cell['positions']) if p['label'] == position)
    vocab = read(output / 'vocabulary.json')
    with np.load(output / 'scores' / f"{cell['cell_id']}.npz", allow_pickle=False) as scores:
        ids, lp = scores['top_ids'][layer, offset], scores['top_logprob'][layer, offset]
        return pd.DataFrame([{'rank': j+1, 'token_id': int(t), 'decoded': repr(vocab['tokens'][t]),
                              'wordpiece': vocab['groups'][vocab['token_group'][t]], 'full_vocab_logprob': float(v),
                              'in_demonstrations': int(t) in cell['demonstration_token_ids'],
                              'in_question': int(t) in cell['question_token_ids']} for j, (t, v) in enumerate(zip(ids, lp))])


def load_completed(output):
    output = Path(output)
    complete = read(output / 'COMPLETE.json')
    if complete.get('status') != 'complete' or not complete.get('passed'):
        raise ValueError('completed, validated recurrence artifact required')
    if 'runtime' in complete:
        return load_completed(complete['runtime'])
    for name, sha in complete['export_sha256'].items():
        if file_digest(output / name) != sha:
            raise ValueError(f'export checksum mismatch: {name}')
    return output


def notebook_export(output):
    """Execute trusted artifact-reading notebook cells, embedding actual tables/figures."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure
    notebook = read(NOTEBOOK)
    namespace = {}
    count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        outputs = []
        def display(value):
            if isinstance(value, Figure):
                buffer = io.BytesIO()
                value.savefig(buffer, format='png', dpi=120)
                data = {'image/png': base64.b64encode(buffer.getvalue()).decode()}
            elif isinstance(value, pd.DataFrame):
                data = {'text/html': value.to_html(index=False), 'text/plain': value.to_string(index=False)}
            else:
                data = {'text/plain': repr(value)}
            outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': data})
        code = ''.join(cell['source']).replace('RUN = DEFAULT_ROOT', f'RUN = Path({str(output)!r})')
        executable = code.replace('from IPython.display import display', '')
        # Only the exporter may view validated artifacts before publishing COMPLETE.
        executable = executable.replace('RUN = load_completed(RUN)', "assert read(RUN / 'validation.json')['passed']")
        namespace['display'] = display
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(executable, f'{NOTEBOOK}:cell{i}', 'exec'), namespace)
        cell.update(source=code.splitlines(keepends=True), execution_count=count,
                    outputs=([{'output_type': 'stream', 'name': 'stdout', 'text': stdout.getvalue()}] if stdout.getvalue() else [])+outputs)
    plt.close('all')
    atomic_json(output / 'five_shot_square_recurrence.executed.ipynb', notebook)


def export(output):
    import pandas as pd
    import matplotlib.pyplot as plt
    output = Path(output)
    if (output / 'COMPLETE.json').exists():
        # A completed artifact is immutable; repeat export verifies it instead.
        load_completed(output)
        return
    plan, data = load_scores(output)
    verify_plan(plan)
    _, mass = load_scores(output, mass=True)
    vocab, groups = read(output / 'vocabulary.json'), read(output / 'selected_groups.json')
    weights = bootstrap_weights(len(data[20]['records']))
    tables = output / 'tables'
    figures = output / 'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    # Save the bootstrap weights: every summary and paired difference uses this one draw.
    np.savez_compressed(output / 'bootstrap.npz', weights=weights.astype(np.int16),
                        pair_ids=np.array([r['cell']['pair_id'] for r in data[20]['records']]))
    save_csv(tables / 'recurrence.csv.gz', summary_rows(data, vocab, weights))
    export_frequencies(output, data, vocab)
    save_csv(tables / 'paired_differences.csv', paired_rows(data, mass, groups, vocab, weights))
    save_csv(tables / 'probability_mass.csv', probability_rows(data, mass, groups, weights))
    cohorts = [{'k': k, 'category': category, 'cohort': cohort, 'n_prompts': int(mask.sum())}
               for k, block in data.items() for category, cohort, mask in subsets(block['records'])]
    save_csv(tables / 'cohorts.csv', cohorts)
    outcomes = [{**{name: r['cell'][name] for name in ('pair_id', 'prompt_id', 'fact_id', 'category', 'k', 'target', 'historical_correct', 'historical_response')},
                 'response': r['response_text'], 'correct': r['correct'],
                 'disagreement': r['correct'] != r['cell']['historical_correct']}
                for block in data.values() for r in block['records']]
    save_csv(tables / 'outcomes.csv', outcomes)
    records = data[20]['records']
    best, overlap, numerators = representative(data[20]['top_ids'], [r['correct'] for r in records], [r['cell']['pair_id'] for r in records])
    save_csv(tables / 'representative_candidates.csv', [{'pair_id': r['cell']['pair_id'], 'correct_k20': r['correct'],
              'mean_top10_overlap': float(overlap[i]), 'integer_overlap_sum': int(numerators[i]), 'selected': i == best}
              for i, r in enumerate(records)])
    representative_info = {'pair_id': records[best]['cell']['pair_id'], 'mean_top10_overlap': float(overlap[best]),
                           'integer_overlap_sum': int(numerators[best]), 'criterion': 'maximum mean corresponding-cell top10 overlap with every k20 ensemble prompt; tie by pair_id',
                           'k0': data[0]['records'][best], 'k20': data[20]['records'][best]}
    atomic_json(output / 'representative.json', representative_info)
    for k in (0, 20):
        r = data[k]['records'][best]
        (output / f'representative_k{k}_prompt.txt').write_text(r['cell']['rendered_prompt'])
    # Ranked exact-token and wordpiece tables remain separate. The default order
    # is persistence, breadth, decoded group spelling (never a selected example).
    frame = pd.read_csv(tables / 'recurrence.csv.gz', keep_default_na=False)
    default = frame[(frame.k == 20) & (frame.region == 'filler') & (frame.category == 'all') & (frame.cohort == 'all')]
    for kind in ('token', 'group'):
        ranking = default[default.kind == kind].sort_values(['persistence', 'breadth', 'text'], ascending=[False, False, True])
        ranking.to_csv(tables / f'filler_{kind}_leaderboard.csv', index=False)
    textual = default[(default.kind == 'group') & (default['filter'] == 'textual')].sort_values(
        ['persistence', 'breadth', 'text'], ascending=[False, False, True]).head(20)
    if textual.item_id.tolist() != [g['group_id'] for g in groups]:
        raise ValueError('rescored groups differ from the final filler leaderboard')
    textual.to_csv(tables / 'top20_textual.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    y = np.arange(len(textual))
    ax.barh(y, textual.persistence, label='Persistence')
    ax.scatter(textual.breadth, y, marker='|', s=100, color='red', label='Breadth')
    ax.errorbar(textual.persistence, y, xerr=np.maximum(0, [textual.persistence-textual.persistence_low,
                textual.persistence_high-textual.persistence]), fmt='none', ecolor='black', capsize=2)
    ax.set_yticks(y, textual.text)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction; k20 filler region, all 262 facts')
    ax.set_title('Recurring words/wordpieces: persistence and breadth')
    ax.legend()
    save_figure(fig, figures / 'filler_leaderboard')
    for gi, group in enumerate(groups):
        for metric in ('top1', 'top10'):
            save_figure(plot_word(output, group['group_id'], metric=metric), figures / f'word_{gi:02d}_{metric}')
        values = {k: np.exp(mass[k]['group_logmass'][..., gi].astype(np.float64)).mean(0) for k in (0, 20)}
        save_figure(paired_heatmap(values[0], values[20], labels(data[0]), labels(data[20]),
                                  f"{group['wordpiece']!r}: exact group probability mass, all facts", probability=True),
                    figures / f'word_{gi:02d}_probability')
        values = {k: np.exp(mass[k]['group_logmass'][best, ..., gi].astype(np.float64)) for k in (0, 20)}
        save_figure(paired_heatmap(values[0], values[20], labels(data[0]), labels(data[20]),
                                  f"Representative: {group['wordpiece']!r}, exact group probability mass", probability=True),
                    figures / f'representative_word_{gi:02d}')
    values = {k: np.exp(data[k]['top_logprob'][best, ..., 0]) for k in (0, 20)}
    annotations = [np.vectorize(lambda t: repr(vocab['tokens'][t])[1:-1])(data[k]['top_ids'][best, ..., 0]) for k in (0, 20)]
    save_figure(paired_heatmap(values[0], values[20], labels(data[0]), labels(data[20]),
                              'Representative top token (color: exact top-token probability)', annotate=annotations, probability=True),
                figures / 'representative_top_token')
    for ti, target in enumerate(('A', 'X', 'A+X')):
        values = {k: np.exp(data[k]['target_logprob'][best, ..., ti].astype(np.float64)) for k in (0, 20)}
        save_figure(paired_heatmap(values[0], values[20], labels(data[0]), labels(data[20]),
                                  f'Representative: exact {target} probability', probability=True), figures / f'representative_target_{ti}')
    # Recheck compact score bytes after aggregation; captures were hashed when scored.
    for mode in ('score', 'mass'):
        for rank in range(4):
            for row in read(output / f'{mode}_rank{rank}.json'):
                if file_digest(Path(row['path'])) != row['sha256']:
                    raise ValueError('scores changed during export')
    verify_plan(plan)
    native = [row['native'] for rank in range(4) for row in read(output / f'score_rank{rank}.json')]
    validation = {'passed': True, 'prompts': len(outcomes), 'readouts': plan['readouts'], 'square_layers': list(range(42)),
                  'native_checks': len(native), 'max_native_logprob_error': max(r['max_logprob_error'] for r in native),
                  'historical_label_disagreements': sum(r['disagreement'] for r in outcomes),
                  'selected_groups': len(groups), 'complete_five_shot_position_coverage': True,
                  'cohort_source': 'strict integer answers from this capture run', 'config_hash': plan['config_hash']}
    atomic_json(output / 'validation.json', validation)
    atomic_json(output / 'provenance.json', {'plan': plan, 'runtime': read(output / 'runtime.json'),
                'selected_groups': groups, 'limitation': LIMITATION,
                'pair_cohorts': 'k20 correct/wrong membership held fixed for both paired conditions',
                'annotation': 'exact vocabulary tokens occurring in saved demonstration turns or target user question; group annotation unions token variants',
                'bootstrap_empty_subsets': 'resamples containing no subgroup members are omitted; counts recorded',
                'frequency_format': 'frequencies/*.npz: code = (layer * n_positions + position) * vocab_size + item_id; divide counts by n'})
    report = ['# Five-shot square J-Lens recurrence', '', datetime.now(timezone.utc).isoformat(), '', LIMITATION, '',
              f"Validated {len(outcomes)} prompts and {plan['readouts']:,} readouts. Native final-readout gates passed for every prompt.",
              f"Historical strict-correctness disagreements: {validation['historical_label_disagreements']}.", '',
              f"Representative: `{representative_info['pair_id']}`, mean top-ten overlap {overlap[best]:.6f}.", '',
              '## Leading filler words/wordpieces', '', '| Group | Persistence | Breadth |', '|---|---:|---:|']
    for row in textual.itertuples():
        report.append(f'| {str(row.text).replace("|", "&#124;")} | {row.persistence:.4f} | {row.breadth:.4f} |')
    report += ['', '![Filler leaderboard](figures/filler_leaderboard.png)', '', '## Representative paired prompts', '']
    for k in (0, 20):
        r = data[k]['records'][best]
        report += [f"### k={k}", '', f"A={r['cell']['answer_value']}; X={r['cell']['addend']}; expected={r['cell']['target']}; response={r['response_text']!r}; correct={r['correct']}.",
                   '', '```text', r['cell']['rendered_prompt'], '```', '']
    report += ['## Outputs', '', '`tables/recurrence.csv.gz`: exact token and grouped region rates, breadth, persistence, confidence intervals and annotations for each cohort/category.',
               '`frequencies/`: compact per-position/layer top-one and top-ten counts for all tokens/groups and cohort/category filters.',
               '`scores/`: vocabulary-wide top ten with full-vocabulary log probabilities and exact A/X/A+X logits, log probabilities and ordinal ranks.',
               '`mass/`: exact full-vocabulary mass for all twenty selected groups at every position, including absence from top ten.',
               '`tables/paired_differences.csv`: shared-position/region paired differences; cohort membership is fixed at k20.',
               '`representative.json`, `tables/representative_candidates.csv`: objective selection and paired native outputs.',
               '`figures/`: PNG/PDF figures. `five_shot_square_recurrence.executed.ipynb`: executed artifact-only notebook.', '']
    (output / 'REPORT.md').write_text('\n'.join(report))
    notebook_export(output)
    log_entry = ('\n\n## ' + datetime.now(timezone.utc).isoformat() + ' — completed capture and analysis\n\n'
                 f"Run: [{output.name}]({output / 'REPORT.md'}). "
                 f"Config `{plan['config_hash']}`; {len(outcomes)} prompts, {plan['readouts']:,} readouts; "
                 f"native readout passed for all prompts, maximum log-probability error {validation['max_native_logprob_error']:.6g}. "
                 f"Historical label disagreements: {validation['historical_label_disagreements']}. "
                 f"Representative `{representative_info['pair_id']}` (overlap {overlap[best]:.6f}).\n\n"
                 'Top five filler groups by persistence: ' + ', '.join(
                     f"{r.text!r} ({r.persistence:.4f}; breadth {r.breadth:.4f})" for r in textual.head(5).itertuples()) + '.\n\n'
                 'Allocation, GPU capacities, timestamps, exact launch, source hashes, native/reference checks, '
                 'paired bootstrap definitions, complete prompts and output links are in the run provenance and report. '
                 + LIMITATION + '\n')
    with LAB_LOG.open('a') as sink:
        sink.write(log_entry)
    # Only now can a consumer regard this run as complete.
    outputs = [p for p in [*tables.iterdir(), *figures.iterdir(), output / 'frequencies.json',
               output / 'representative.json', output / 'validation.json', output / 'provenance.json',
               output / 'REPORT.md', output / 'five_shot_square_recurrence.executed.ipynb',
               *[output / name for name in ('preflight.json', 'captures.json', 'CAPTURED.json',
                    'vocabulary.json', 'selected_groups.json', 'bootstrap.npz', 'runtime.json')],
               *[output / f'{mode}_rank{rank}.json' for mode in ('score', 'mass') for rank in range(4)],
               *[output / f'worker_provenance{rank}.json' for rank in range(4)]] if p.is_file()]
    checksums = {str(p.relative_to(output)): file_digest(p) for p in outputs}
    for mode in ('score', 'mass'):
        for rank in range(4):
            for row in read(output / f'{mode}_rank{rank}.json'):
                checksums[str(Path(row['path']).relative_to(output))] = row['sha256']
    for row in read(output / 'frequencies.json'):
        checksums[row['path']] = row['sha256']
    atomic_json(output / 'COMPLETE.json', {**validation, 'status': 'complete',
        'finished_at': datetime.now(timezone.utc).isoformat(), 'export_sha256': checksums})
    print(f'Completed {output / "REPORT.md"}', flush=True)

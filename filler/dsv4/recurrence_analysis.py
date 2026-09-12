"""Fact-level recurrence, pairing, exact grouping and representative selection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse

from filler.dsv4.five_shot_recurrence import read, LAYERS, normalize
from filler.dsv4.patching import atomic_json, file_digest, digest

REGIONS = ('question', 'filler', 'answer_prefix', 'assistant_transition')


def region_indices(positions, region):
    def member(label):
        return {'question': label == 'last_question', 'filler': label.startswith('filler_'),
                'answer_prefix': label in ('answer_word', 'answer_colon'),
                'assistant_transition': label in ('assistant', 'end_think')}[region]
    return [i for i, p in enumerate(positions) if member(p['label'])]


def grouped_top(ids, token_group):
    """Preserve top-one identity and count each textual variant group once per cell."""
    groups = np.asarray(token_group, dtype=np.int32)[ids].copy()
    original = groups.copy()
    for j in range(1, groups.shape[-1]):
        groups[..., j] = np.where((original[..., :j] == original[..., j:j+1]).any(-1), -1, original[..., j])
    return groups


def bootstrap_weights(n, *, resamples=2000, seed=42):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    counts = np.zeros((resamples, n), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(resamples), n), draws.ravel()), 1)
    return counts


def interval(values, weights, mask=None):
    """Resample paired facts once; cohort membership is fixed before resampling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    mask = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
    w = weights * mask[None, :]
    denom = w.sum(-1)
    valid = denom > 0
    if not mask.any():
        return np.full((2, values.shape[1]), np.nan)
    rates = (w[valid] @ values) / denom[valid, None]
    return np.quantile(rates, [.025, .975], axis=0)


def region_matrix(top, positions):
    """Prompt-by-observed-token cell counts; zeros include all eligible prompts."""
    x = top[:, :, positions, :]
    n, layers, width, depth = x.shape
    if width == 0:
        raise ValueError('empty region has no recurrence denominator')
    flat = x.reshape(n, -1)
    valid = flat >= 0
    items, inverse = np.unique(flat[valid], return_inverse=True)
    rows = np.broadcast_to(np.arange(n)[:, None], flat.shape)[valid]
    counts = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, inverse)),
                              shape=(n, len(items))).tocsr()
    counts.sum_duplicates()
    return items, counts, layers*width


def summary_arrays(top, positions):
    items, counts, cells = region_matrix(top, positions)
    rates = counts / cells
    breadth = counts.copy()
    breadth.data[:] = 1
    top1_items, first, _ = region_matrix(top[..., :1], positions)
    mapping = np.searchsorted(items, top1_items)
    first = first.tocoo()
    first = sparse.coo_matrix((first.data/cells, (first.row, mapping[first.col])), shape=counts.shape).tocsr()
    return items, {'persistence': rates, 'breadth': breadth, 'top1': first}, cells


def representative(top, correct, pair_ids):
    """Exact mean |top10_i ∩ top10_j|/10 over corresponding cells and all j.

    Include each candidate itself, as 'every ensemble prompt' requires. Its
    identical self term cannot change ordering. Integer sums avoid float ties.
    """
    n, layers, positions, depth = top.shape
    if depth != 10 or len(pair_ids) != n or len(set(pair_ids)) != n:
        raise ValueError('representative requires unique paired facts and top ten')
    if np.any(np.diff(np.sort(top, -1), axis=-1) == 0):
        raise ValueError('duplicate top-ten token')
    scores = np.zeros(n, dtype=np.int64)
    for l in range(layers):
        for p in range(positions):
            values = top[:, l, p]
            freq = np.bincount(values.ravel())
            scores += freq[values].sum(-1)
    eligible = np.flatnonzero(correct)
    if not len(eligible):
        raise ValueError('no correctly answered k20 representative exists in this run')
    best = min(eligible, key=lambda i: (-int(scores[i]), pair_ids[i]))
    return int(best), scores / (n*layers*positions*depth), scores


def load_scores(output, *, mass=False):
    """Validate all prompt/score links and the complete dense scoring grid."""
    from filler.dsv4.recurrence_runtime import validate_records
    output = Path(output)
    plan = read(output / 'preflight.json')
    records = read(output / 'captures.json')
    validate_records(plan, records)
    captured = read(output / 'CAPTURED.json')
    if captured['config_hash'] != plan['config_hash'] or captured['captures_sha256'] != file_digest(output / 'captures.json') or captured['count'] != len(records):
        raise ValueError('capture completion checksum mismatch')
    reports = [r for rank in range(4) for r in read(output / f"{'mass' if mass else 'score'}_rank{rank}.json")]
    by_id = {r['cell_id']: r for r in reports}
    if len(by_id) != len(reports) or set(by_id) != {r['cell']['cell_id'] for r in records}:
        raise ValueError('missing or duplicate scoring report')
    vocab = read(output / 'vocabulary.json')
    if digest(vocab) != plan['vocabulary_sha256']:
        raise ValueError('vocabulary mapping checksum mismatch')
    vocab_size = len(vocab['tokens'])
    groups = read(output / 'selected_groups.json') if mass else []
    results = {0: [], 20: []}
    count = 0
    for record in sorted(records, key=lambda r: (r['cell']['pair_id'], r['cell']['k'])):
        cell = record['cell']
        report = by_id[cell['cell_id']]
        path = output / ('mass' if mass else 'scores') / f"{cell['cell_id']}.npz"
        if Path(report['path']).resolve() != path.resolve() or file_digest(path) != report['sha256']:
            raise ValueError('scoring file checksum/path mismatch')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['config_hash']) != plan['config_hash'] or str(saved['cell_id']) != cell['cell_id']
                    or str(saved['capture_sha256']) != record['capture']['ranks']['0']['sha256']
                    or str(saved['vocabulary_sha256']) != plan['vocabulary_sha256']
                    or (mass and str(saved['selected_groups_sha256']) != digest(groups))):
                raise ValueError('scoring metadata/capture mismatch')
            names = ('group_logmass',) if mass else ('top_ids', 'top_logprob', 'target_logprob', 'target_logits', 'target_rank')
            arrays = {name: saved[name] for name in names}
        p = len(cell['positions'])
        for name, a in arrays.items():
            depth = len(groups) if mass else (10 if name.startswith('top_') else 3)
            if a.shape != (len(LAYERS), p, depth) or not np.isfinite(a).all():
                raise ValueError('incomplete scoring shape/nonfinite values')
            if name.endswith('logprob') or mass:
                if (a > 1e-5).any():
                    raise ValueError('probabilities exceed one')
        if not mass:
            ids, lp, ranks = arrays['top_ids'], arrays['top_logprob'], arrays['target_rank']
            if ids.dtype != np.int32 or ranks.dtype != np.int32 or (ids < 0).any() or (ids >= vocab_size).any():
                raise ValueError('invalid token IDs/dtypes')
            if (np.diff(np.sort(ids, -1), axis=-1) == 0).any() or (np.diff(lp, axis=-1) > 0).any():
                raise ValueError('invalid top-ten selection')
            if ((np.diff(lp, axis=-1) == 0) & (np.diff(ids, axis=-1) <= 0)).any():
                raise ValueError('top-ten ties are not ordered by ascending token ID')
            if (ranks < 1).any() or (ranks > vocab_size).any() or not report['native']['passed']:
                raise ValueError('rank/native gate failed')
            for j, key in enumerate(('A', 'X', 'A+X')):
                hit = ids == cell['target_token_ids'][key]
                present = hit.any(-1)
                if not np.array_equal(ranks[..., j] <= 10, present):
                    raise ValueError('target rank/top-ten membership disagree')
                if present.any():
                    index = hit.argmax(-1)
                    if not np.array_equal(ranks[..., j][present], (index+1)[present]):
                        raise ValueError('target rank/tie mismatch')
                    if not np.allclose(arrays['target_logprob'][..., j][present], np.take_along_axis(lp, index[..., None], -1)[..., 0][present], atol=1e-6, rtol=0):
                        raise ValueError('target log probability mismatch')
        results[cell['k']].append({'record': record, **arrays})
        count += p*len(LAYERS)
    if count != plan['readouts']:
        raise ValueError('scoring coverage differs from plan')
    if not mass:
        worker_provenance = [read(output / f'worker_provenance{rank}.json') for rank in range(4)]
        head_hashes = [w['readout_tensor_sha256'] for w in worker_provenance]
        if not head_hashes[0] or any(h != head_hashes[0] for h in head_hashes):
            raise ValueError('workers used different readout tensors')
        refs = [c for r in reports for c in r.get('reference', [])]
        if {c['layer'] for c in refs if c['passed']} != set(LAYERS):
            raise ValueError('missing all-layer reference validation')
        for k in (0, 20):
            checks = [by_id[r['record']['cell']['cell_id']]['native'] for r in results[k]]
            if not any(c.get('hidden_max_abs_error') == 0 for c in checks):
                raise ValueError('missing returned-hidden/native equivalence validation for a condition')
    for k, rows in results.items():
        results[k] = {'records': [r.pop('record') for r in rows],
                      **{name: np.stack([r[name] for r in rows]) for name in rows[0]}}
    if [r['cell']['pair_id'] for r in results[0]['records']] != [r['cell']['pair_id'] for r in results[20]['records']]:
        raise ValueError('paired condition order mismatch')
    return plan, results


def select_groups(output):
    _, data = load_scores(output)
    vocab = read(output / 'vocabulary.json')
    top = grouped_top(data[20]['top_ids'], vocab['token_group'])
    positions = region_indices(data[20]['records'][0]['cell']['positions'], 'filler')
    items, metrics, _ = summary_arrays(top, positions)
    persistence = np.asarray(metrics['persistence'].mean(0)).ravel()
    breadth = np.asarray(metrics['breadth'].mean(0)).ravel()
    indices = [i for i, item in enumerate(items) if vocab['group_kind'][item] == 'textual']
    indices.sort(key=lambda i: (-persistence[i], -breadth[i], vocab['groups'][items[i]]))
    selected = [{'group_id': int(items[i]), 'wordpiece': vocab['groups'][items[i]],
                 'persistence': float(persistence[i]), 'breadth': float(breadth[i])} for i in indices[:20]]
    if len(selected) != 20:
        raise ValueError('fewer than twenty textual groups observed')
    atomic_json(output / 'selected_groups.json', selected)
    return selected


def subsets(records, *, paired=False):
    categories = sorted({r['cell']['category'] for r in records})
    correct = np.array([r['correct'] for r in records])
    for category in ['all', *categories]:
        category_mask = np.array([category == 'all' or r['cell']['category'] == category for r in records])
        for cohort in ('all', 'correct', 'wrong'):
            mask = category_mask & (np.ones(len(records), bool) if cohort == 'all' else correct == (cohort == 'correct'))
            yield category, cohort, mask


def annotation_matrix(records, vocab, items, scope, kind):
    lookup = {int(token): j for j, token in enumerate(items)}
    result = np.zeros((len(records), len(items)), bool)
    for i, record in enumerate(records):
        ids = record['cell'][scope + '_token_ids']
        ids = {vocab['token_group'][t] for t in ids} if kind == 'group' else set(ids)
        for token in ids:
            if token in lookup:
                result[i, lookup[token]] = True
    return result

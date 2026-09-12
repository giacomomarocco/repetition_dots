"""Score existing five-shot residuals for square J-Lens and Logit Lens heatmaps.

No model server or prompt execution is used. Historical captures/scores are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from filler.dsv4.patching import atomic_json, digest, file_digest
from filler.dsv4.lens_positions import prompt_position_labels, validate_positions

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34'
DEST = ROOT / 'runs/deepseek-v4-flash/five-shot-jlens-heatmap'
NOTEBOOK = ROOT / 'notebooks/one_fact_top_object_jlens_heatmap.ipynb'
LOGIT_NOTEBOOK = ROOT / 'notebooks/one_fact_top_object_heatmap.ipynb'
OBJECTS = ('A', 'X', 'A+X')
NUMERICAL_SOURCES = ('filler/dsv4/recurrence_runtime.py', 'filler/dsv4/jlens.py',
                     'filler/dsv4/lens.py', 'filler/dsv4/lens_positions.py',
                     'filler/dsv4/workspace_jlens_artifact.py')


def read(path):
    return json.loads(Path(path).read_text())


def numeric_ids(tokens):
    from filler.dsv4.top_object_heatmap import canonical_numeric_token_ids
    class Vocabulary:
        def __len__(self):
            return len(tokens)

        def decode(self, ids, **kwargs):
            return ''.join(tokens[i] for i in ids)
    return canonical_numeric_token_ids(Vocabulary())


def argmax_ids(logits, numeric):
    """Sorted vocabulary IDs give lowest-ID tie breaking in both scopes."""
    import torch
    if (numeric.ndim != 1 or numeric.numel() == 0
            or not bool(torch.all(numeric[1:] > numeric[:-1]))
            or int(numeric[0]) < 0 or int(numeric[-1]) >= logits.shape[-1]
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite logits and sorted unique numerical token IDs required')
    return logits.argmax(-1), numeric[logits.index_select(-1, numeric).argmax(-1)]


def prepare(root):
    from filler.dsv4.recurrence_runtime import validate_records
    complete, historical = read(SOURCE / 'COMPLETE.json'), read(SOURCE / 'preflight.json')
    if (complete.get('passed') is not True or complete.get('status') != 'complete'
            or complete['prompts'] != 524 or complete['readouts'] != 330120
            or complete.get('complete_five_shot_position_coverage') is not True
            or complete['config_hash'] != historical['config_hash']
            or digest({k: v for k, v in historical.items() if k != 'config_hash'}) != historical['config_hash']):
        raise ValueError('source run is not a complete validated five-shot ensemble')
    for name in ('captures.json', 'vocabulary.json', 'validation.json'):
        if file_digest(SOURCE / name) != complete['export_sha256'][name]:
            raise ValueError(f'completed source checksum mismatch: {name}')
    records = read(SOURCE / 'captures.json')
    validate_records(historical, records)
    vocab = read(SOURCE / 'vocabulary.json')
    if digest(vocab) != historical['vocabulary_sha256']:
        raise ValueError('vocabulary differs from capture plan')
    for record in records:
        validate_positions(record['cell'], lambda ids: ''.join(vocab['tokens'][i] for i in ids))
    for name in NUMERICAL_SOURCES:
        if file_digest(ROOT / name) != historical['source_hashes'][name]:
            raise ValueError(f'numerical source differs from validated scoring: {name}')
    names = set(NUMERICAL_SOURCES) | {
        'filler/dsv4/five_shot_jlens_heatmap.py', 'filler/dsv4/five_shot_recurrence.py',
        'filler/dsv4/campaign_hook.py', 'filler/dsv4/factorial.py',
        'filler/addition/one_fact.py', 'filler/dsv4/top_object_heatmap.py',
        'filler/dsv4/jlens_comparison.py', 'scripts/dsv4/five_shot_jlens_heatmap.py',
        'run_five_shot_jlens_heatmap_allocation.sh', str(NOTEBOOK.relative_to(ROOT)), str(LOGIT_NOTEBOOK.relative_to(ROOT))}
    names |= {n for n in historical['source_hashes'] if n.startswith(('ports/', 'model/'))}
    hashes = {str(ROOT / name): file_digest(ROOT / name) for name in names}
    frozen = DEST / 'source/patching.py'
    hashes[str(frozen)] = file_digest(frozen)
    hashes.update({str(SOURCE / n): file_digest(SOURCE / n) for n in
                   ('COMPLETE.json', 'preflight.json', 'captures.json', 'vocabulary.json', 'validation.json')})
    # Large tensors are stat-checked here and checksum-checked by compute workers.
    capture_files = {ref['path']: {'size': Path(ref['path']).stat().st_size,
                                  'mtime_ns': Path(ref['path']).stat().st_mtime_ns}
                     for r in records for ref in r['capture']['ranks'].values()}
    cohorts = Counter(f"{r['cell']['filler_length']}/{'correct' if r['correct'] else 'wrong'}" for r in records)
    cohorts.update(Counter(f"{r['cell']['filler_length']}/all" for r in records))
    if cohorts['0/all'] != 262 or cohorts['20/all'] != 262:
        raise ValueError('expected 262 examples per length')
    plan = {'source': str(SOURCE), 'created_at': datetime.now(timezone.utc).isoformat(),
            'source_hashes': hashes, 'capture_files': capture_files,
            'checkpoint_files': historical['checkpoint_files'], 'square': historical['square'],
            'numeric_token_ids': numeric_ids(vocab['tokens']), 'vocab_size': len(vocab['tokens']),
            'cohort_counts': dict(cohorts), 'positions': 7860, 'readouts': 330120, 'logit_readouts': 337980,
            'forward_passes': 0, 'position_suffix': 'five_shot', 'layers': list(range(42)),
            'logit_layers': list(range(43)), 'readout': {
                'square': 'FP32 stream mean/transport; published BF16 HF norm/head',
                'logit': 'native SGLang fused mHC/RMSNorm and TP=4 vocabulary GEMMs'},
            'ties': 'lowest vocabulary ID wins exact ties'}
    plan['config_hash'] = digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'COMPLETE.json').exists():
        raise ValueError('use a new output root for a new completed run')
    atomic_json(root / 'preflight.json', plan)
    print(json.dumps({k: plan[k] for k in ('config_hash', 'forward_passes', 'readouts', 'cohort_counts')}, indent=2))


def verify(plan):
    if digest({k: v for k, v in plan.items() if k != 'config_hash'}) != plan['config_hash']:
        raise ValueError('scoring plan checksum mismatch')
    for path, checksum in plan['source_hashes'].items():
        if file_digest(Path(path)) != checksum:
            raise ValueError(f'pinned source/input changed: {path}')
    from filler.dsv4.five_shot_recurrence import CHECKPOINT, DEFAULT_PATH
    paths = {**plan['capture_files'], **{str(CHECKPOINT / n): v for n, v in plan['checkpoint_files'].items()},
             str(DEFAULT_PATH): {k: plan['square'][k] for k in ('size', 'mtime_ns')}}
    for path, expected in paths.items():
        stat = Path(path).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError(f'pinned tensor changed: {path}')


def score_worker(output, rank):
    import numpy as np
    import torch
    from filler.dsv4.five_shot_recurrence import CHECKPOINT, DEFAULT_PATH
    from filler.dsv4.jlens import load_workspace_jlens, project_jlens_logits
    from filler.dsv4.lens import load_checkpoint_readout, project_sglang_logits
    from filler.dsv4.recurrence_runtime import load_capture, native_check, reference_checks, require_compute
    from filler.dsv4.workspace_jlens_artifact import verify_artifact
    require_compute()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    plan = read(output / 'preflight.json')
    verify(plan)
    source = Path(plan['source'])
    complete = read(source / 'COMPLETE.json')
    records = read(source / 'captures.json')
    verify_artifact(DEFAULT_PATH)
    lens = load_workspace_jlens(DEFAULT_PATH)
    lens = replace(lens, jacobians={l: t.to(device=device, dtype=torch.float32) for l, t in lens.jacobians.items()})
    weights = load_checkpoint_readout(CHECKPOINT, device=device)
    if weights.lm_head_weight.dtype != torch.bfloat16 or len(weights.lm_head_weight) != plan['vocab_size']:
        raise ValueError('head precision/vocabulary mismatch')
    numeric = torch.tensor(plan['numeric_token_ids'], device=device)
    reports = []
    with torch.inference_mode():
        for i, record in enumerate(records):
            if (i // 2) % 4 != rank:
                continue
            if time.time() > read(output / 'runtime.json')['deadline']:
                raise RuntimeError('allocation deadline reached')
            saved = load_capture(record)
            cid = record['cell']['cell_id']
            report = {'cell_id': cid, 'native': native_check(record, saved, weights)}
            if len(reports) < 2:
                report['reference'] = reference_checks(lens, weights, saved)
            old_path = source / 'scores' / f'{cid}.npz'
            if file_digest(old_path) != complete['export_sha256'][f'scores/{cid}.npz']:
                raise ValueError('historical score checksum mismatch')
            with np.load(old_path, allow_pickle=False) as old:
                if (str(old['cell_id']) != cid or str(old['config_hash']) != complete['config_hash']
                        or str(old['capture_sha256']) != record['capture']['ranks']['0']['sha256']):
                    raise ValueError('historical scores use different capture provenance')
                winners, numbers, ordinary_full, ordinary_numeric = [], [], [], []
                for layer in range(43):
                    state = saved['states'][layer].to(device)
                    if layer < 42:
                        logits = project_jlens_logits(state, lens, layer, weights)
                        full, number = argmax_ids(logits, numeric)
                        full, number = full.cpu().numpy(), number.cpu().numpy()
                        if not np.array_equal(full, old['top_ids'][layer, :, 0]):
                            raise ValueError(f'historical full argmax disagreement: {cid}/{layer}')
                        target_ids = [record['cell']['target_token_ids'][t] for t in OBJECTS]
                        if not np.array_equal(logits[:, target_ids].float().cpu().numpy(), old['target_logits'][layer]):
                            raise ValueError(f'historical target logit disagreement: {cid}/{layer}')
                        winners.append(full)
                        numbers.append(number)
                    logits = project_sglang_logits(state, weights)
                    full, number = argmax_ids(logits, numeric)
                    ordinary_full.append(full.cpu().numpy())
                    ordinary_numeric.append(number.cpu().numpy())
                if int(ordinary_full[42][-1]) != read(record['response'])['output_ids'][0]:
                    raise ValueError('ordinary final-layer argmax differs from native output')
                path = output / 'scores' / f'{cid}.npz'
                np.savez_compressed(path, top_token_id=np.asarray(winners, dtype=np.int32),
                                    top_numeric_token_id=np.asarray(numbers, dtype=np.int32),
                                    logit_top_token_id=np.asarray(ordinary_full, dtype=np.int32),
                                    logit_top_numeric_token_id=np.asarray(ordinary_numeric, dtype=np.int32),
                                    cell_id=cid, config_hash=plan['config_hash'],
                                    capture_sha256=record['capture']['ranks']['0']['sha256'])
            reports.append({**report, 'historical_argmax_and_target_logits_exact': True,
                            'path': str(path), 'sha256': file_digest(path)})
            atomic_json(output / f'worker{rank}-progress.json', reports)
            print(f'GPU {rank}: scored {len(reports)} prompts; {cid}', flush=True)
    verify(plan)
    atomic_json(output / f'worker{rank}.json', {'rank': rank, 'torch': str(torch.__version__),
                'cuda': torch.version.cuda, 'python': sys.version, 'reports': reports})


def summarize(records, arrays, numeric, config_hash, *, lens='square'):
    """Stream per-prompt arrays, rejecting missing positions, labels and outcomes."""
    import numpy as np
    if lens not in ('square', 'logit'):
        raise ValueError('unknown lens')
    layers = 42 if lens == 'square' else 43
    prefix = '' if lens == 'square' else 'logit_'
    sums, counts, seen = {}, Counter(), set()
    for record, score in zip(records, arrays, strict=True):
        c = record['cell']
        cid, k = c['cell_id'], c['filler_length']
        if cid in seen or str(score['cell_id']) != cid or str(score['config_hash']) != config_hash:
            raise ValueError('duplicate/mismatched scored prompt')
        seen.add(cid)
        positions = prompt_position_labels(k, suffix='five_shot')
        if [p['label'] for p in c['positions']] != positions or type(record['correct']) is not bool:
            raise ValueError('incomplete positions or missing native cohort')
        shape = (layers, len(positions))
        full, number = score[prefix+'top_token_id'], score[prefix+'top_numeric_token_id']
        if (full.shape != shape or number.shape != shape or not np.isin(number, numeric).all()
                or not np.issubdtype(full.dtype, np.integer)
                or not np.issubdtype(number.dtype, np.integer) or (full < 0).any()):
            raise ValueError('invalid/incomplete score arrays')
        targets = np.asarray([c['target_token_ids'][t] for t in OBJECTS])
        metrics = np.stack((full[..., None] == targets, number[..., None] == targets), -1)
        for cohort in ('all', 'correct' if record['correct'] else 'wrong'):
            key = k, cohort
            if key not in sums:
                sums[key] = np.zeros(metrics.shape, dtype=np.float64)
            sums[key] += metrics
            counts[f'{k}/{cohort}'] += 1
    rows = []
    for (k, cohort), total in sorted(sums.items()):
        n = counts[f'{k}/{cohort}']
        for l in range(layers):
            for j, p in enumerate(prompt_position_labels(k, suffix='five_shot')):
                for t, target in enumerate(OBJECTS):
                    rows.append({'filler_length': k, 'cohort': cohort, 'layer': l,
                                 'position_label': p, 'target_name': target, 'n': n,
                                 **dict(zip((f'{lens}_top1', f'{lens}_numeric_top1'),
                                            (total[l, j, t] / n).tolist()))})
    return rows, dict(counts)


def load_completed(run, *, validated=False, lens='square'):
    if lens not in ('square', 'logit'):
        raise ValueError('unknown lens')
    run = Path(run)
    completion = read(run / ('validation.json' if validated else 'COMPLETE.json'))
    if (completion.get('passed') is not True or completion.get('position_suffix') != 'five_shot'
            or (run / 'FAILED.json').exists()
            or file_digest(run / f'{lens}_summary.json') != completion[f'{lens}_summary_sha256']):
        raise ValueError('five-shot heatmap is incomplete or has changed')
    return completion, read(run / f'{lens}_summary.json')


def heatmap_stats(summary, completion, filler_length, cohort, *, lens='square'):
    """Notebook adapter with exact coverage and saved-cohort checks."""
    import math
    if lens not in ('square', 'logit'):
        raise ValueError('unknown lens')
    positions = prompt_position_labels(filler_length, suffix='five_shot')
    layers = list(range(42 if lens == 'square' else 43))
    selected = [r for r in summary if r['filler_length'] == filler_length and r['cohort'] == cohort]
    indexed = {(r['position_label'], r['layer'], r['target_name']): r for r in selected}
    if len(indexed) != len(selected) or set(indexed) != {(p, l, t) for p in positions for l in layers for t in OBJECTS}:
        raise ValueError('incomplete/duplicate five-shot summary positions or layers')
    n = completion['cohort_counts'][f'{filler_length}/{cohort}']
    if n <= 0 or any(r['n'] != n for r in selected):
        raise ValueError('inconsistent native cohort counts')
    rates = {scope: {t: [[indexed[p, l, t][f'{lens}_{metric}'] for l in layers] for p in positions]
                     for t in OBJECTS} for scope, metric in (('all', 'top1'), ('numeric', 'numeric_top1'))}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for objects in rates.values()
           for matrix in objects.values() for row in matrix for v in row):
        raise ValueError('invalid heatmap rates')
    return {'filler_length': filler_length, 'cohort': cohort, 'positions': positions, 'layers': layers,
            'rates': rates, 'example_counts': {f'{p}:{l}': n for p in positions for l in layers}}


def export(output):
    import numpy as np
    plan = read(output / 'preflight.json')
    records = read(Path(plan['source']) / 'captures.json')
    reports = [r for rank in range(4) for r in read(output / f'worker{rank}.json')['reports']]
    by_id = {r['cell_id']: r for r in reports}
    if len(reports) != len(by_id) or set(by_id) != {r['cell']['cell_id'] for r in records}:
        raise ValueError('missing/duplicate worker reports')
    def arrays():
        for record in records:
            report = by_id[record['cell']['cell_id']]
            if (report['native']['passed'] is not True or report['historical_argmax_and_target_logits_exact'] is not True
                    or file_digest(Path(report['path'])) != report['sha256']):
                raise ValueError('failed score/native validation')
            with np.load(report['path'], allow_pickle=False) as values:
                if str(values['capture_sha256']) != record['capture']['ranks']['0']['sha256']:
                    raise ValueError('score/capture mismatch')
                yield values
    summary, cohorts = summarize(records, arrays(), plan['numeric_token_ids'], plan['config_hash'])
    if cohorts != plan['cohort_counts'] or sum(len(r['cell']['positions'])*42 for r in records) != 330120:
        raise ValueError('incomplete cohort or readout coverage')
    atomic_json(output / 'square_summary.json', summary)
    logit_summary, logit_cohorts = summarize(records, arrays(), plan['numeric_token_ids'], plan['config_hash'], lens='logit')
    if logit_cohorts != cohorts:
        raise ValueError('lens cohort mismatch')
    atomic_json(output / 'logit_summary.json', logit_summary)
    validation = {'passed': True, 'config_hash': plan['config_hash'], 'prompts': len(records),
                  'readouts': 330120, 'cohort_counts': cohorts, 'position_suffix': 'five_shot',
                  'complete_answer_prefix': True, 'square_summary_sha256': file_digest(output / 'square_summary.json'),
                  'logit_summary_sha256': file_digest(output / 'logit_summary.json'), 'logit_readouts': 337980,
                  'source': plan['source'], 'native_checks': len(reports),
                  'max_native_logprob_error': max(r['native']['max_logprob_error'] for r in reports)}
    for lens, rows in (('square', summary), ('logit', logit_summary)):
        for k in (0, 20):
            for cohort in ('all', 'correct', 'wrong'):
                heatmap_stats(rows, validation, k, cohort, lens=lens)
    atomic_json(output / 'validation.json', validation)
    return validation


def run(root):
    from filler.dsv4.recurrence_runtime import allocation, stop_process
    plan = read(root / 'preflight.json')
    verify(plan)
    info, gpus, deadline = allocation()
    output = root / 'runtimes' / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scores').mkdir()
    atomic_json(output / 'preflight.json', plan)
    atomic_json(output / 'runtime.json', {'allocation': info, 'gpus': gpus, 'deadline': deadline,
                'started_at': datetime.now(timezone.utc).isoformat(), 'command': sys.argv})
    for path in plan['source_hashes']:
        p = Path(path)
        if p.suffix in ('.py', '.sh', '.ipynb'):
            dest = output / 'source' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    children, logs = [], []
    def interrupted(*_):
        raise KeyboardInterrupt('allocation interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        for rank in range(4):
            log = (output / f'gpu{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-m', 'scripts.dsv4.five_shot_jlens_heatmap',
                            'worker', '--root', str(output), '--rank', str(rank)], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(c.poll() is None for c in children):
            if time.time() > deadline or any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(f'scoring worker failed/deadline reached; see {output}/gpu*.log')
            time.sleep(5)
        if any(c.returncode for c in children):
            raise RuntimeError('scoring worker failed')
        validation = export(output)
        nersc_python = '/global/common/software/nersc/pe/conda-envs/26.8.0/python-3.13/nersc-python/bin/python'
        subprocess.run([nersc_python, '-m', 'scripts.dsv4.five_shot_jlens_heatmap', 'notebook',
                        '--root', str(output)], cwd=ROOT, check=True, timeout=min(600, max(1, deadline-time.time())))
        verify(plan)
        complete = {**validation, 'status': 'complete', 'runtime': str(output),
                    'executed_notebook': 'one_fact_top_object_jlens_heatmap.ipynb',
                    'finished_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(output / 'COMPLETE.json', complete)
        atomic_json(root / 'COMPLETE.json', complete)
        print(json.dumps(complete, indent=2), flush=True)
    except BaseException as error:
        atomic_json(output / 'FAILED.json', {'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        for child in children:
            stop_process(child)
        for log in logs:
            log.close()
        atomic_json(output / 'ended.json', {'ended_at': datetime.now(timezone.utc).isoformat()})


def execute_notebook(output):
    import nbformat
    import re
    import matplotlib
    original_rc = matplotlib.matplotlib_fname()
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for notebook in (NOTEBOOK, LOGIT_NOTEBOOK):
        nb = nbformat.read(notebook, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                cell.source = re.sub(r'^DATASET = .*$', "DATASET = 'five_shot'", cell.source, flags=re.M)
                cell.source = cell.source.replace('FIVE_SHOT_RUN = None', f'FIVE_SHOT_RUN = Path({str(output)!r})')
                cell.source = cell.source.replace('VALIDATED_EXPORT = False', 'VALIDATED_EXPORT = True')
        nb.cells.insert(0, nbformat.v4.new_code_cell(
            'import matplotlib\nmatplotlib.rc_file(' + repr(original_rc) + ')\n'
            + "get_ipython().run_line_magic('matplotlib', 'inline')"))
        km = KernelManager(kernel_name=nb.metadata.kernelspec.name)
        km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(ROOT)}}).execute()
        images = [o for c in nb.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
        if len(images) < 3:
            raise ValueError('notebook did not emit its heatmap images')
        import base64
        for i, item in enumerate(images):
            encoded = item['data']['image/png']
            (output / f'{notebook.stem}-{i:02d}.png').write_bytes(base64.b64decode(encoded))
        nbformat.write(nb, output / notebook.name)
        print(f'{notebook.name} verified with {sys.executable}: {len(images)} inline PNGs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'notebook'))
    parser.add_argument('--root', type=Path, default=DEST)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'worker':
        score_worker(root, args.rank)
    else:
        {'prepare': prepare, 'run': run, 'notebook': execute_notebook}[args.action](root)

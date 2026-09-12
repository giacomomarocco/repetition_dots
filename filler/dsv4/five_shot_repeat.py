"""Five-shot repeat intervention, with indivisible paired-example checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filler.addition.one_fact import parse_answer
from filler.dsv4.five_shot_recurrence import SOURCE, ROOT, CHECKPOINT, make_cells, read, tokenizer
from filler.dsv4.patching import Journal, atomic_bytes, atomic_json, digest, file_digest, invariant, requested_scores
from filler.dsv4.patching_campaign import Campaign, WalltimeReached, native_equivalence, read_response
from filler.dsv4.patching_logits import first_prediction, validate_logits

DEFAULT_ROOT = ROOT / 'runs/deepseek-v4-flash/one-fact-patching-five-shot-repeat'
CONDITIONS = ['baseline', 'no_intervention'] + [f'identity_{s}' for s in range(6)] + [f'repeat_{s}' for s in range(6)]


def mapping(cell, source):
    if cell['k'] != 20 or source not in range(6):
        raise ValueError('requires k=20 and source 0..5')
    positions = {p['label']: p['absolute_position'] for p in cell['positions']}
    dest = [positions[f'filler_{i}'] for i in range(source + 1, 20)]
    return dest, [positions[f'filler_{source}']] * len(dest)


def score(cell, response):
    ids = list(dict.fromkeys(cell['target_token_ids'].values()))
    first = first_prediction(response, 8)
    values = requested_scores(first, ids)
    target = cell['target_token_ids']['A+X']
    return {'text': response['text'], 'parsed_answer': parse_answer(response['text']),
            'correct': parse_answer(response['text']) == cell['target'],
            'first_token_correct': response['output_ids'][0] == target,
            'target_logprob': values[target], 'output_ids': response['output_ids']}


def control_gate(clean, changed, ids, *, full_response):
    gate = invariant(first_prediction(clean, 8), first_prediction(changed, 8), ids)
    # Layer 42 leaves the initial prediction invariant. Its edited KV context can
    # affect later generated tokens; full-response equality is an identity gate.
    gate['complete_response_equal'] = clean['text'] == changed['text'] and clean['output_ids'] == changed['output_ids']
    gate['passed'] = gate['passed'] and (not full_response or gate['complete_response_equal'])
    return gate


class FiveShotRepeat(Campaign):
    def __init__(self, manifest, root, runtime_id, transport, checkpoint, *, deadline, validation=native_equivalence):
        self.manifest, self.root, self.runtime_id = manifest, root, runtime_id
        self.transport, self.checkpoint, self.deadline = transport, checkpoint, deadline
        self.validation = validation
        self.journal = Journal(root, manifest['config_hash'])
        self.runtime_root = root / 'runtimes' / runtime_id
        self.durations = []
        self.stop_requested, self.runtime_validated, self.current = False, False, None
        self.planned_passes = 14 * sum(p['record_id'] not in self.journal.records for p in manifest['pairs']) + 6

    def check_time(self):
        mean = sum(self.durations) / len(self.durations) if self.durations else 30
        remaining = self.deadline - time.time()
        self.journal.progress(runtime_id=self.runtime_id, completed_examples=len(self.journal.records),
                              seconds_remaining=remaining, mean_request_seconds=mean,
                              projected_remaining_seconds=max(0, self.planned_passes-len(self.durations))*mean)
        if self.stop_requested or remaining < max(180, 2 * mean):
            raise WalltimeReached('checkpointed; incomplete example retained as diagnostic attempt')

    def run(self):
        for pair in self.manifest['pairs']:
            if pair['record_id'] in self.journal.records:
                continue
            self.check_time()
            a, b = pair['cells']
            ids = list(dict.fromkeys(b['target_token_ids'].values()))
            results = {}
            results['baseline'] = self.execute(cell=a, token_ids=ids, mode='full_downstream')
            results['no_intervention'] = self.execute(cell=b, token_ids=ids, mode='full_downstream', capture_all=True)
            clean = read_response(results['no_intervention'])
            capture = results['no_intervention']['capture']
            attempt = self.runtime_root / 'attempts' / results['no_intervention']['request_id']
            if not self.runtime_validated:
                native = self.validation(capture, first_prediction(clean, 8), self.checkpoint, ids)
                atomic_json(attempt / 'native.json', native)
                if not native['passed']:
                    raise RuntimeError('native final-layer equivalence failed')
            else:
                native = self.native
            gates, diagnostics = {}, []
            for source in range(6):
                dest, src = mapping(b, source)
                result = self.execute(cell=b, token_ids=ids, mode='full_downstream', layers=range(43),
                                      positions=dest, source_positions=dest, donor=capture)
                results[f'identity_{source}'] = result
                gate = control_gate(clean, read_response(result), ids, full_response=True)
                gates[f'identity_{source}'] = gate
                atomic_json(attempt / f'identity-{source}.json', gate)
                if not gate['passed']:
                    raise RuntimeError('identity control failed')
            if not self.runtime_validated:
                for source in range(6):
                    dest, src = mapping(b, source)
                    result = self.execute(cell=b, token_ids=ids, mode='full_downstream', layers=[42],
                                          positions=dest, source_positions=src, donor=capture)
                    gate = control_gate(clean, read_response(result), ids, full_response=False)
                    diagnostics.append({'source': source, 'result': result, 'gate': gate})
                    atomic_json(attempt / f'layer42-{source}.json', diagnostics[-1])
                    if not gate['passed']:
                        raise RuntimeError('layer-42-only initial prediction invariance failed')
                self.native = native
                self.runtime_gate = {'native': native, 'baseline': results['no_intervention'], 'diagnostics': diagnostics}
                atomic_json(self.runtime_root / 'validation.json', self.runtime_gate)
                self.runtime_validated = True
            for source in range(6):
                dest, src = mapping(b, source)
                results[f'repeat_{source}'] = self.execute(cell=b, token_ids=ids, mode='full_downstream',
                    layers=range(43), positions=dest, source_positions=src, donor=capture)
            # No baselines or individual trials are journaled before all 14 pass.
            self.journal.append({'record_id': pair['record_id'], 'kind': 'example_block',
                'runtime_id': self.runtime_id, 'pair_id': b['pair_id'], 'results': results,
                'gates': gates, 'runtime_gate': self.runtime_gate,
                'scores': {k: score(a if k == 'baseline' else b, read_response(v)) for k, v in results.items()}})
            print(f"Completed {len(self.journal.records)}/{len(self.manifest['pairs'])} examples; "
                  f"last response={read_response(results['no_intervention'])['text']!r}", flush=True)
        return {'passed': True, 'examples': len(self.journal.records)}


def checked_result(result, cell, manifest, *, kind, source=None, donor=None, runtime=None):
    from filler.dsv4.campaign_hook import validate_ack_records
    for path_key, hash_key in [('control_path', 'control_sha256'), ('ack_path', 'ack_sha256')]:
        if file_digest(Path(result[path_key])) != result[hash_key]:
            raise ValueError('request artifact changed')
    control = read(result['control_path'])
    dest, src = ([], []) if source is None else mapping(cell, source)
    expected_layers = [] if source is None else ([42] if kind == 'diagnostic' else list(range(43)))
    expected_src = dest if kind == 'identity' else src
    if (control['config_hash'] != manifest['config_hash'] or control['cell_id'] != cell['cell_id']
            or control['runtime_id'] != runtime or result['runtime_id'] != runtime
            or control['request_id'] != result['request_id']
            or control['input_ids_hash'] != digest(cell['input_ids'])
            or control['num_tokens'] != len(cell['input_ids']) or control['max_new_tokens'] != 8
            or control['layers'] != expected_layers or control['positions'] != dest
            or control.get('source_positions', []) != expected_src
            or control['recomputation'] != 'full_downstream' or control['donor_capture'] != donor
            or control['clean_capture'] is not None
            or control['capture_all'] != (kind == 'no_intervention')):
        raise ValueError('request protocol or final-question mapping mismatch')
    acks = read(result['ack_path'])
    validate_ack_records(acks, control)
    if result['capture']['ranks'] != {str(a['rank']): a['capture'] for a in acks}:
        raise ValueError('capture references differ from acknowledgements')
    for ack in acks:
        ref = ack['capture']
        if file_digest(Path(ref['path'])) != ref['sha256']:
            raise ValueError('residual capture changed')
    response = read_response(result)
    if response['meta_info'].get('cached_tokens') != 0:
        raise ValueError('cached tokens in capture')
    validate_logits(response, control)
    return response


def summarize(manifest, journal, root, *, integrity=None):
    from filler.addition.accuracy_plot import wilson_interval
    expected = {p['record_id'] for p in manifest['pairs']}
    if set(journal.records) != expected:
        raise ValueError('campaign incomplete or contains unexpected/duplicate blocks')
    rows, validated_runtimes = [], set()
    for pair in manifest['pairs']:
        block = journal.records[pair['record_id']]
        a, b = pair['cells']
        if set(block['results']) != set(CONDITIONS) or block['pair_id'] != b['pair_id']:
            raise ValueError('incomplete or misaligned example block')
        runtime = block['runtime_id']
        capture = block['results']['no_intervention']['capture']
        responses = {}
        for name in CONDITIONS:
            source = int(name.rsplit('_', 1)[1]) if name.startswith(('repeat_', 'identity_')) else None
            kind = name.split('_')[0] if source is not None else name
            cell = a if name == 'baseline' else b
            response = checked_result(block['results'][name], cell, manifest, kind=kind,
                source=source, donor=capture if source is not None else None, runtime=runtime)
            responses[name] = response
            scores = score(cell, response)
            if scores != block['scores'][name]:
                raise ValueError('raw-response score reconciliation failed')
            rows.append({'pair_id': b['pair_id'], 'condition': name, 'prompt_id': cell['prompt_id'],
                         'runtime_id': runtime, **scores})
        ids = list(dict.fromkeys(b['target_token_ids'].values()))
        for s in range(6):
            name = f'identity_{s}'
            gate = control_gate(responses['no_intervention'], responses[name], ids, full_response=True)
            if not gate['passed'] or gate != block['gates'][name]:
                raise ValueError('identity gate reconstruction failed')
        if runtime not in validated_runtimes:
            gate = block['runtime_gate']
            if not gate['native']['passed'] or len(gate['diagnostics']) != 6:
                raise ValueError('missing native/diagnostic validation')
            # Validation may belong to an earlier example in this runtime.
            cid = gate['baseline']['capture']['cell_id']
            gc = next(c for p in manifest['pairs'] for c in p['cells'] if c['cell_id'] == cid)
            gcids = list(dict.fromkeys(gc['target_token_ids'].values()))
            baseline = checked_result(gate['baseline'], gc, manifest, kind='no_intervention', runtime=runtime)
            for s, diagnostic in enumerate(gate['diagnostics']):
                if diagnostic['source'] != s:
                    raise ValueError('duplicate or missing diagnostic source')
                response = checked_result(diagnostic['result'], gc, manifest, kind='diagnostic', source=s,
                    donor=gate['baseline']['capture'], runtime=runtime)
                check = control_gate(baseline, response, gcids, full_response=False)
                if not check['passed'] or check != diagnostic['gate']:
                    raise ValueError('layer-42 gate reconstruction failed')
            validated_runtimes.add(runtime)
    points = []
    for name in ['baseline', 'no_intervention'] + [f'repeat_{s}' for s in range(6)]:
        selected = [r for r in rows if r['condition'] == name]
        correct, count = sum(r['correct'] for r in selected), len(selected)
        lo, hi = wilson_interval(correct, count)
        points.append({'condition': name, 'untouched': int(name[-1])+1 if name.startswith('repeat') else None,
                       'correct': correct, 'count': count, 'accuracy': correct/count, 'ci_low': lo, 'ci_high': hi})
    transitions = {}
    base = {r['pair_id']: r['correct'] for r in rows if r['condition'] == 'no_intervention'}
    for s in range(6):
        selected = [r for r in rows if r['condition'] == f'repeat_{s}']
        transitions[str(s)] = {f'{old}_to_{new}': sum(base[r['pair_id']] == old and r['correct'] == new for r in selected)
                              for old in (False, True) for new in (False, True)}
    report = {'config_hash': manifest['config_hash'], 'points': points, 'transitions': transitions,
              'historical': manifest['historical'], 'integrity': {'passed': True, 'examples': len(expected),
              'conditions_per_example': 14, 'runtimes': sorted(validated_runtimes)}}
    atomic_json(root / 'analysis/raw_scores.json', rows)
    atomic_json(root / 'analysis/plotted_counts.json', points)
    atomic_json(root / 'analysis/summary.json', report)
    atomic_json(root / 'COMPLETE.json', {'status': 'complete', 'config_hash': manifest['config_hash'],
                'integrity': report['integrity'], 'finished_at': datetime.now(timezone.utc).isoformat()})
    journal.progress(status='complete', integrity=report['integrity'])
    lab(root, 'Completed capture and independent raw-response/rank integrity reconciliation. '
              'Fresh and historical reference counts: ' + json.dumps({'fresh': points[:2], 'historical': manifest['historical']}))
    return report


def lab(root, text):
    path = root / 'LAB_LOG.md'
    atomic_bytes(path, ((path.read_text() if path.exists() else '# Five-shot repeat lab log\n') +
                       f'\n## {datetime.now(timezone.utc).isoformat()}\n\n{text}\n').encode())


def prepare(root):
    from scripts.dsv4.one_fact_patching import SOURCES, CHECKS, checkpoint_files, launch_command, verify_manifest
    if (root / 'results.jsonl').exists():
        raise ValueError('existing campaign: run/analyze instead of replacing frozen inputs')
    config = read(SOURCE / 'run_config.json')
    if config['decoding'] != {'temperature': 0, 'max_new_tokens': 8} or len(config['demonstrations']) != 5:
        raise ValueError('saved decoding/demonstrations changed')
    prompts, history = read(SOURCE / 'prompts.json'), read(SOURCE / 'results.json')
    selected = [r for r in prompts if r['k'] in (0, 20)]
    outcomes = [r for r in history if r['k'] in (0, 20)]
    if len(selected) != 524 or len(outcomes) != 524 or len({r['prompt_id'] for r in outcomes}) != 524:
        raise ValueError('missing/duplicate five-shot inputs')
    cells = make_cells(prompts, history, tokenizer())
    pairs = [{'record_id': f"example|{a['pair_id']}", 'cells': [a, b]} for a, b in zip(cells[::2], cells[1::2])]
    for pair in pairs:
        for source in range(6):
            dest, src = mapping(pair['cells'][1], source)
            if len(dest) != 19-source or len(set(src)) != 1:
                raise ValueError('wrong replacement count')
    paths = [ROOT / n for n in SOURCES] + [ROOT / n for n in (
        'filler/dsv4/five_shot_repeat.py', 'scripts/dsv4/five_shot_repeat.py',
        'filler/dsv4/five_shot_recurrence.py', 'tests/test_five_shot_repeat.py',
        'run_five_shot_repeat_allocation.sh')]
    paths += [SOURCE / n for n in ('prompts.json', 'results.json', 'run_config.json')]
    paths += [CHECKPOINT / n for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')]
    manifest = {'pairs': pairs, 'trials': [{'record_id': p['record_id']} for p in pairs], 'identity_controls': [],
        'raw_logits': True, 'max_new_tokens': 8, 'source_hashes': {str(p.relative_to(ROOT)): file_digest(p) for p in paths},
        'checkpoint_files': checkpoint_files(), 'created_at': datetime.now(timezone.utc).isoformat(),
        'runtime_requirements': {'tp_size': 4, 'disable_radix_cache': True, 'disable_cuda_graph': True,
            'chunked_prefill_size': -1, 'max_running_requests': 1, 'enable_return_hidden_states': True,
            'fused_mhc_post_pre': False},
        'historical': {str(k): {'correct': sum(c['historical_correct'] for c in cells if c['k'] == k), 'count': 262} for k in (0,20)},
        'revisions': {n: subprocess.check_output(['git', '-C', str(ROOT/p), 'rev-parse', 'HEAD'], text=True).strip()
                      for n,p in [('workspace','.'), ('sglang','ports/sglang'), ('port','ports/deepseek-v4-a100-sglang')]},
        'semantics': 'Post-block complete mHC residual; all ranks/layers 0..42; final-question fillers only; '
                     'sources 0..5 copied from same-example clean layer states; no clean correctness filter; '
                     'whole 14-condition example checkpoints; strict complete-answer parsing; layer42 first prediction invariant.'}
    if len(manifest['checkpoint_files']) != 48:
        raise ValueError('expected 48 checkpoint shards')
    manifest['config_hash'] = digest(manifest)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'manifest.json', manifest)
    for p in paths:
        atomic_bytes(root / 'source_snapshot' / p.relative_to(ROOT), p.read_bytes())
    cfg = read(CHECKPOINT / 'config.json')
    row_bytes = 2 * cfg['hc_mult'] * cfg['hidden_size'] * 43 * 4
    # Full k20 donor captures, one-row k0 captures, selected trial/control rows.
    nrows = sum(len(p['cells'][1]['input_ids']) + 1 + 2*sum(21-s for s in range(6)) for p in pairs) + sum(21-s for s in range(6))
    estimate = nrows*row_bytes + 3674*1024**2
    stat = os.statvfs(root)
    storage = {'estimated_bytes': estimate, 'recommended_free_bytes': int(estimate*1.25),
               'filesystem_free_bytes': stat.f_bavail*stat.f_frsize,
               'scope': 'one uninterrupted campaign; partial attempts add storage; filesystem free is not user quota'}
    if storage['recommended_free_bytes'] > storage['filesystem_free_bytes']:
        raise ValueError('insufficient filesystem free space')
    checks = ['tests/test_dsv4_factorial.py', 'tests/test_deepseek_v4_logit_lens.py', 'tests/test_dsv4_transplant_hook.py', 'tests/test_activation_patching_runner.py', 'tests/test_five_shot_repeat.py',
              'tests/test_five_shot_recurrence.py::test_actual_524_prompt_coverage',
              'tests/test_five_shot_recurrence.py::test_bad_position_coverage',
              'tests/test_five_shot_recurrence.py::test_eight_token_response_uses_first_scores']
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           'TOKENIZERS_PARALLELISM':'false', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', *checks], cwd=ROOT, env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    atomic_bytes(root / 'preflight-tests.log', test.stdout.encode())
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError('CPU preflight failed')
    subprocess.run(['bash', '-n', str(ROOT/'run_five_shot_repeat_allocation.sh')], check=True)
    verify_manifest(manifest)
    report = {'passed': True, 'config_hash': manifest['config_hash'], 'examples':262, 'requests':3674,
              'storage':storage, 'tests':checks, 'gpu_gates':'pending; required before interventions',
              'server_command':launch_command(root/'runtimes/RUNTIME/control', raw_logits=True)}
    atomic_json(root/'preflight.json', report)
    lab(root, 'Prepared exact 524 saved prompts for 262 paired examples, sources 0–5. 3,668 block requests '
        '+ six layer-42 diagnostics = 3,674 generation requests per uninterrupted runtime. '
        'CPU checks passed. Notebook/plots deferred until after GPU launch. See manifest.json, preflight.json '
        'and source_snapshot/ for inputs, source hashes, revisions, decoding and storage estimate.')
    print(json.dumps(report, indent=2), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','analyze'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=30002)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'run':
        from scripts.dsv4.one_fact_patching import run
        run(root, args.port, campaign_class=FiveShotRepeat, finish_callback=summarize)
    else:
        manifest = read(root/'manifest.json')
        summarize(manifest, Journal(root, manifest['config_hash']), root)

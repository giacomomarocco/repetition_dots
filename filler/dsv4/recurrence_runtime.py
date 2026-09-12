"""Allocation-owned capture and distributed scoring for five-shot recurrence."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from filler.dsv4.five_shot_recurrence import (
    ROOT, CHECKPOINT, DEFAULT_PATH, SHA256, LAYERS, read, tokenizer, vocabulary, verify_plan)
from filler.dsv4.patching import atomic_json, digest, file_digest, requested_scores


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('run/export require srun in an explicitly approved compute allocation')


def allocation():
    require_compute()
    import torch
    raw = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'],
                                  text=True, env={**os.environ, 'TZ': 'UTC'})
    info = dict(x.split('=', 1) for x in shlex.split(raw) if '=' in x)
    if info.get('Account') != 'm5258_g' or info.get('NumNodes') != '1' or info.get('QOS') not in ('interactive', 'gpu_interactive'):
        raise RuntimeError('requires one interactive node, account m5258_g')
    if torch.cuda.device_count() != 4:
        raise RuntimeError('requires four visible GPUs')
    devices = [torch.cuda.get_device_properties(i) for i in range(4)]
    if any('A100' not in d.name or d.total_memory < 79 * 1024**3 for d in devices):
        raise RuntimeError('requires four A100 80 GB GPUs')
    deadline = datetime.fromisoformat(info['EndTime']).replace(tzinfo=timezone.utc).timestamp() - 180
    return info, [{'name': d.name, 'bytes': d.total_memory} for d in devices], deadline


def first_token_scores(response, token_ids):
    """Validate native candidate scores at step zero of an eight-token completion."""
    if not 1 <= len(response['output_ids']) <= 8:
        raise ValueError('expected one to eight generated tokens')
    rows = response['meta_info']['output_token_ids_logprobs']
    if len(rows) != len(response['output_ids']):
        raise ValueError('missing native candidate scoring steps')
    first = {**response, 'output_ids': response['output_ids'][:1],
             'meta_info': {**response['meta_info'], 'output_token_ids_logprobs': rows[:1]}}
    return requested_scores(first, list(dict.fromkeys(token_ids)))


def request(control_root, control, cell, port):
    """One full prefill, selected residuals, greedy eight-token native completion."""
    from filler.dsv4.campaign_hook import validate_acknowledgements
    from scripts.dsv4.run_activation_patching import flush_cache, post_json
    url = f'http://127.0.0.1:{port}'
    flush_cache(url)
    atomic_json(Path(control['output_root']) / 'control.json', control)
    atomic_json(control_root / 'NEXT.json', control)
    response = post_json(url + '/generate', {
        'input_ids': cell['input_ids'], 'sampling_params': {'temperature': 0, 'max_new_tokens': 8},
        'return_logprob': True, 'top_logprobs_num': 20,
        'token_ids_logprob': list(dict.fromkeys(cell['target_token_ids'].values())),
        'return_hidden_states': control['validate_returned_hidden']}, timeout=600)
    response_path = Path(control['output_root']) / 'response.json'
    atomic_json(response_path, response)
    if response['meta_info'].get('cached_tokens') != 0 or not response['output_ids']:
        raise ValueError('cached prefill or empty generation')
    first_token_scores(response, list(cell['target_token_ids'].values()))
    acks = validate_acknowledgements(control_root, control)
    atomic_json(Path(control['output_root']) / 'acks.json', acks)
    from filler.addition.one_fact import parse_answer
    text = response['text']
    return {'cell': cell, 'response': str(response_path), 'response_sha256': file_digest(response_path),
            'control_sha256': file_digest(response_path.parent / 'control.json'),
            'acks_sha256': file_digest(response_path.parent / 'acks.json'),
            'correct': parse_answer(text) == cell['target'], 'response_text': text,
            'capture': {'cell_id': cell['cell_id'], 'ranks': {str(a['rank']): a['capture'] for a in acks}}}


def validate_records(plan, records):
    """Metadata checks reject missing/duplicate/mismatched capture and response links."""
    from filler.addition.one_fact import parse_answer
    from filler.dsv4.campaign_hook import validate_ack_records
    planned = {c['cell_id']: c for c in plan['cells']}
    seen, paths, requests, runtimes = set(), set(), set(), set()
    for record in records:
        cell = record['cell']
        cid = cell['cell_id']
        if cid in seen or cid not in planned or cell != planned[cid] or record['capture']['cell_id'] != cid:
            raise ValueError('missing, duplicate or mismatched capture cell')
        seen.add(cid)
        response_path = Path(record['response'])
        for name, sha in [('response.json', record['response_sha256']),
                          ('control.json', record['control_sha256']), ('acks.json', record['acks_sha256'])]:
            if file_digest(response_path.parent / name) != sha:
                raise ValueError('response/control/ack checksum mismatch')
        control, acks = read(response_path.parent / 'control.json'), read(response_path.parent / 'acks.json')
        validate_ack_records(acks, control)
        expected_positions = [p['absolute_position'] for p in cell['positions']]
        if (control['cell_id'] != cid or control['config_hash'] != plan['config_hash']
                or control['input_ids_hash'] != digest(cell['input_ids'])
                or control['num_tokens'] != len(cell['input_ids']) or control['positions'] != expected_positions
                or control['layers'] or control['capture_all'] or control['clean_capture'] is not None
                or control['donor_capture'] is not None or control['recomputation'] != 'full_downstream'
                or Path(control['output_root']).resolve() != response_path.parent.resolve()):
            raise ValueError('capture control does not describe this unpatched prompt')
        key = control['runtime_id'], control['request_id']
        if key in requests:
            raise ValueError('duplicate capture request')
        requests.add(key)
        runtimes.add(control['runtime_id'])
        for rank, ack in enumerate(acks):
            ref = record['capture']['ranks'][str(rank)]
            if ref != ack['capture'] or Path(ref['path']).resolve() != (response_path.parent / f'rank{rank}.pt').resolve():
                raise ValueError('capture/ack/path mismatch')
            if ref['path'] in paths:
                raise ValueError('capture path reused')
            paths.add(ref['path'])
        response = read(response_path)
        if (record['response_text'] != response['text'] or record['correct'] != (parse_answer(response['text']) == cell['target'])
                or response['meta_info'].get('cached_tokens') != 0 or not response['output_ids']):
            raise ValueError('native outcome mismatch')
    if seen != set(planned) or len(runtimes) != 1:
        raise ValueError('missing captures or mixed capture runtimes')


def load_capture(record, rank=0):
    import torch
    ref = record['capture']['ranks'][str(rank)]
    if file_digest(Path(ref['path'])) != ref['sha256']:
        raise ValueError('capture tensor checksum mismatch')
    saved = torch.load(ref['path'], map_location='cpu', weights_only=True)
    control = read(Path(record['response']).parent / 'control.json')
    meta = saved['metadata']
    if any(meta.get(k) != control[k] for k in ('cell_id', 'request_id', 'runtime_id', 'config_hash', 'num_tokens')):
        raise ValueError('capture tensor metadata mismatch')
    positions = [p['absolute_position'] for p in record['cell']['positions']]
    if (meta['positions'] != positions or meta['rank'] != rank or meta['num_layers'] != 43
            or meta['fused_mhc_post_pre'] is not False or set(saved['states']) != set(range(43))):
        raise ValueError('missing, duplicate or mismatched tensor positions/layers')
    for state in saved['states'].values():
        if state.shape != (len(positions), 4, 4096) or state.dtype != torch.bfloat16 or not torch.isfinite(state).all():
            raise ValueError('invalid captured residual shape/dtype/values')
    return saved


def stable_top10(logits):
    """Vocabulary-wide top 10, exact boundary ties ordered by ascending ID.

    A small stable sort suffices after explicitly recovering boundary ties; no
    epsilon perturbation of logits or nondeterministic topk tie survives.
    """
    import torch
    if logits.shape[-1] < 10 or not torch.isfinite(logits).all():
        raise ValueError('expected finite vocabulary logits of width at least ten')
    vals, ids = torch.topk(logits, 10, dim=-1)
    boundary = vals[..., -1:]
    vocab = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
    tied = torch.topk(torch.where(logits == boundary, vocab, logits.shape[-1]), 10, largest=False).values
    candidates = torch.cat((ids, tied.clamp_max(logits.shape[-1]-1)), -1)
    scores = torch.cat((torch.where(vals > boundary, vals, -torch.inf),
                        torch.where(tied < logits.shape[-1], boundary, -torch.inf)), -1)
    order = candidates.argsort(dim=-1, stable=True)
    candidates, scores = candidates.gather(-1, order), scores.gather(-1, order)
    order = scores.argsort(dim=-1, descending=True, stable=True)[..., :10]
    return candidates.gather(-1, order)


def score_logits(logits, target_ids):
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    ids = stable_top10(logits)
    targets = torch.as_tensor(target_ids, device=logits.device)
    tv = logits.index_select(-1, targets)
    ranks = []
    vocab = torch.arange(logits.shape[-1], device=logits.device)
    for i, token in enumerate(target_ids):
        value = tv[..., i:i+1]
        ranks.append(1 + ((logits > value) | ((logits == value) & (vocab < token))).sum(-1))
    return {'top_ids': ids, 'top_logprob': lp.gather(-1, ids),
            'target_logprob': lp.index_select(-1, targets), 'target_logits': tv,
            'target_rank': torch.stack(ranks, -1)}


def score_group_mass(logits, group_ids):
    """Normalize over the entire vocabulary, including groups absent from top ten."""
    import torch
    if not group_ids or any(ids.numel() == 0 for ids in group_ids) or not torch.isfinite(logits).all():
        raise ValueError('group mass requires finite logits and nonempty vocabulary groups')
    logz = torch.logsumexp(logits.float(), -1)
    return torch.stack([torch.logsumexp(logits.float().index_select(-1, ids), -1) - logz
                        for ids in group_ids], -1)


def native_check(record, saved, weights):
    import torch
    from filler.dsv4.lens import project_sglang_logits
    state = saved['states'][42][-1]
    errors = []
    for rank in range(1, 4):
        other = load_capture(record, rank)
        # Replicated residual streams must match on every selected layer/position.
        if any(not torch.equal(saved['states'][l], other['states'][l]) for l in range(43)):
            raise ValueError('TP rank residual disagreement')
        errors.append(0)
    response = read(record['response'])
    control = read(Path(record['response']).parent / 'control.json')
    hidden_error = None
    if control['validate_returned_hidden']:
        returned = torch.as_tensor(response['meta_info']['hidden_states'][0])
        if returned.ndim == 2:
            returned = returned[-1]
        hidden_error = float((returned.flatten().float() - state.flatten().float()).abs().max())
    logits = project_sglang_logits(state.to(weights.lm_head_weight.device), weights).float()
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite native readout')
    lp = torch.log_softmax(logits, -1)
    scores = first_token_scores(response, list(record['cell']['target_token_ids'].values()))
    scores.update({int(v[1]): float(v[0]) for v in response['meta_info']['output_top_logprobs'][0]})
    error = max(abs(float(lp[t]) - value) for t, value in scores.items())
    report = {'argmax_equal': int(lp.argmax()) == response['output_ids'][0],
              'hidden_max_abs_error': hidden_error, 'max_logprob_error': error,
              'all_selected_residuals_equal_across_ranks': True}
    if not report['argmax_equal'] or hidden_error not in (None, 0) or error > .15:
        raise ValueError(f'native final-readout validation failed: {report}')
    return {**report, 'passed': True}


def reference_checks(lens, weights, saved):
    import torch
    from filler.dsv4.jlens import published_readout_fixture, transport, unembed_transported
    sys.path.insert(0, str(ROOT / 'ports/jacobian-lens-open-frontier'))
    import jlens
    if Path(jlens.__file__).resolve().parent != ROOT / 'ports/jacobian-lens-open-frontier/jlens':
        raise ValueError('unexpected reference package')
    fixture = published_readout_fixture(weights)
    checks = []
    for layer in LAYERS:
        state = saved['states'][layer][[0, -1]].to(weights.lm_head_weight.device)
        mean = sum(state[:, stream, :].float() for stream in range(4)) / 4
        expected = torch.einsum('ij,bj->bi', lens.jacobians[layer], mean)
        actual = transport(state, lens, layer)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        a = unembed_transported(actual, weights)
        b = fixture.unembed(expected, collapse=False).float()
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        checks.append({'layer': layer, 'passed': True, 'transport_error': float((actual-expected).abs().max()),
                       'readout_error': float((a-b).abs().max())})
    return checks


def score_worker(output, rank, *, mass=False):
    import hashlib
    import importlib.metadata
    import numpy as np
    import torch
    from filler.dsv4.jlens import load_workspace_jlens, project_jlens_logits
    from filler.dsv4.lens import load_checkpoint_readout
    from filler.dsv4.workspace_jlens_artifact import verify_artifact
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    plan = read(output / 'preflight.json')
    verify_plan(plan)
    records = read(output / 'captures.json')
    validate_records(plan, records)
    if not mass:
        verify_artifact(DEFAULT_PATH)
    lens = load_workspace_jlens(DEFAULT_PATH)
    lens = replace(lens, jacobians={l: t.to(device=device, dtype=torch.float32) for l, t in lens.jacobians.items()})
    weights = load_checkpoint_readout(CHECKPOINT, device=device)
    if weights.lm_head_weight.dtype != torch.bfloat16 or len(weights.lm_head_weight) != tokenizer().get_vocab_size():
        raise ValueError('head precision/vocabulary mismatch')
    if not mass:
        atomic_json(output / f'worker_provenance{rank}.json', {
            'rank': rank, 'device': device, 'torch': str(torch.__version__), 'python': sys.version,
            'transformers': importlib.metadata.version('transformers'),
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'allow_tf32': False,
            'readout_tensor_sha256': {name: hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
                for name, value in vars(weights).items() if isinstance(value, torch.Tensor)}})
    groups = read(output / 'selected_groups.json') if mass else []
    vocab = read(output / 'vocabulary.json')
    if digest(vocab) != plan['vocabulary_sha256']:
        raise ValueError('vocabulary mapping differs from preflight')
    group_map = np.asarray(vocab['token_group'])
    group_ids = [torch.tensor(np.flatnonzero(group_map == g['group_id']), device=device) for g in groups]
    reports = []
    with torch.inference_mode():
        for i, record in enumerate(records):
            if (i // 2) % 4 != rank:
                continue
            if time.time() > read(output / 'runtime.json')['deadline']:
                raise RuntimeError('allocation deadline reached during scoring')
            saved = load_capture(record)
            report = {'cell_id': record['cell']['cell_id']}
            if not mass:
                report['native'] = native_check(record, saved, weights)
                # Reference at every layer, on both prompt conditions, on each GPU.
                if len(reports) < 2:
                    report['reference'] = reference_checks(lens, weights, saved)
            values = {}
            for layer in LAYERS:
                states = saved['states'][layer].to(device)
                logits = project_jlens_logits(states, lens, layer, weights)
                if mass:
                    metrics = {'group_logmass': score_group_mass(logits, group_ids)}
                else:
                    metrics = score_logits(logits, [record['cell']['target_token_ids'][k] for k in ('A', 'X', 'A+X')])
                for name, value in metrics.items():
                    values.setdefault(name, []).append(value.cpu().numpy())
            arrays = {name: np.stack(v).astype(np.int32 if name in ('top_ids', 'target_rank') else np.float32)
                      for name, v in values.items()}
            path = output / ('mass' if mass else 'scores') / f"{record['cell']['cell_id']}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **arrays, config_hash=plan['config_hash'], cell_id=record['cell']['cell_id'],
                                capture_sha256=record['capture']['ranks']['0']['sha256'],
                                vocabulary_sha256=plan['vocabulary_sha256'],
                                selected_groups_sha256=digest(groups) if mass else '')
            reports.append({**report, 'path': str(path), 'sha256': file_digest(path)})
            print(f"{'Mass' if mass else 'Score'} GPU{rank} {i+1}/{len(records)} {record['cell']['cell_id']}", flush=True)
    atomic_json(output / f"{'mass' if mass else 'score'}_rank{rank}.json", reports)


def workers(output, mode, deadline):
    children, logs = [], []
    try:
        for rank in range(4):
            log = (output / f'{mode}_gpu{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-m', 'scripts.dsv4.five_shot_recurrence', 'run',
                '--root', str(output), '--worker', mode, '--rank', str(rank)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in children):
            if time.time() > deadline or any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(f'{mode} worker failed or deadline reached; inspect GPU logs')
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError(f'{mode} worker failed')
    finally:
        for child in children:
            stop_process(child)
        for log in logs:
            log.close()


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run(root, port):
    from filler.dsv4.campaign_hook import campaign_hook_spec
    from filler.dsv4.workspace_jlens_artifact import verify_artifact
    from scripts.dsv4.one_fact_patching import launch_command, server_environment, get_json
    from filler.dsv4.recurrence_analysis import select_groups
    from filler.dsv4.recurrence_export import export
    plan = read(root / 'preflight.json')
    verify_plan(plan)
    info, devices, deadline = allocation()
    # Check artifact bytes and identity on the allocated node before any model load.
    verify_artifact(DEFAULT_PATH)
    from filler.dsv4.jlens import load_workspace_jlens
    load_workspace_jlens(DEFAULT_PATH)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    runtime_id = f"{os.environ['SLURM_JOB_ID']}-{uuid.uuid4().hex[:8]}"
    output = root / 'runtimes' / runtime_id
    output.mkdir(parents=True, exist_ok=False)
    control_root = output / 'control'
    command = launch_command(control_root, port)
    atomic_json(output / 'preflight.json', plan)
    atomic_json(output / 'vocabulary.json', vocabulary(tokenizer()))
    atomic_json(output / 'runtime.json', {'runtime_id': runtime_id, 'job_id': os.environ['SLURM_JOB_ID'],
        'allocation': info, 'gpus': devices, 'deadline': deadline, 'hostname': socket.gethostname(),
        'command': command, 'config_hash': plan['config_hash'], 'started_at': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'square_sha256': SHA256})
    def interrupted(*_):
        raise KeyboardInterrupt('allocation interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    records = []
    try:
        with (output / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=ROOT, env=server_environment(), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            try:
                startup_deadline = min(deadline, time.time() + 2400)
                while True:
                    if server.poll() is not None or time.time() > startup_deadline:
                        raise RuntimeError('model startup failed or timed out; see server.log')
                    try:
                        server_info = get_json(f'http://127.0.0.1:{port}/server_info')
                        break
                    except (OSError, ValueError):
                        print('Waiting for instrumented model startup', flush=True)
                        time.sleep(15)
                required = {'disable_radix_cache': True, 'chunked_prefill_size': -1, 'disable_cuda_graph': True,
                    'disable_piecewise_cuda_graph': True, 'disable_overlap_schedule': True,
                    'max_running_requests': 1, 'enable_return_hidden_states': True}
                if (any(server_info.get(k) != v for k, v in required.items())
                        or server_info.get('forward_hooks') != campaign_hook_spec(control_root)
                        or Path(server_info['model_path']).resolve() != CHECKPOINT):
                    raise ValueError('server instrumentation/config mismatch')
                atomic_json(output / 'server_config.json', {k: server_info.get(k) for k in (
                    *required, 'version', 'forward_hooks', 'model_path', 'dtype', 'quantization')})
                for i, cell in enumerate(plan['cells']):
                    if time.time() > deadline:
                        raise RuntimeError('allocation deadline reached during capture')
                    request_id = f'capture-{i:04d}'
                    control = {'request_id': request_id, 'runtime_id': runtime_id, 'config_hash': plan['config_hash'],
                        'cell_id': cell['cell_id'], 'input_ids_hash': digest(cell['input_ids']), 'num_tokens': len(cell['input_ids']),
                        'layers': [], 'positions': [p['absolute_position'] for p in cell['positions']],
                        'recomputation': 'full_downstream', 'capture_all': False, 'clean_capture': None, 'donor_capture': None,
                        'validate_returned_hidden': i < 2,
                        'output_root': str(output / 'passes' / request_id)}
                    records.append(request(control_root, control, cell, port))
                    atomic_json(output / 'captures.json', records)
                    print(f'Captured {i+1}/{len(plan["cells"])}', flush=True)
            finally:
                stop_process(server)
        validate_records(plan, records)
        atomic_json(output / 'CAPTURED.json', {'config_hash': plan['config_hash'], 'count': len(records),
                    'captures_sha256': file_digest(output / 'captures.json')})
        workers(output, 'score', deadline)
        select_groups(output)
        workers(output, 'mass', deadline)
        export(output)
        atomic_json(root / 'COMPLETE.json', {**read(output / 'COMPLETE.json'), 'runtime': str(output)})
    except BaseException as error:
        atomic_json(output / 'FAILED.json', {'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        atomic_json(output / 'ended.json', {'ended_at': datetime.now(timezone.utc).isoformat()})

"""Metadata adapter for validated unpatched full-prompt answer-token captures."""
from __future__ import annotations

import json
from pathlib import Path

from filler.dsv4.lens_positions import validate_positions
from filler.dsv4.patching import digest, file_digest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/deepseek-v4-flash/answer-token-heatmap'
PUBLIC_IDENTITY = ('capture_format', 'runtime_id', 'request_id', 'capture_config_hash',
                   'input_ids_sha256', 'num_tokens', 'position_token_id', 'position_token')


def collect_full_prompt_examples(run, checkpoint, *, tokenizer=None, examples_per_length=96):
    """Inspect manifests/controls/acks and file stats, never activation tensor bytes."""
    from filler.dsv4.campaign_hook import validate_ack_records
    run, checkpoint = Path(run).resolve(), Path(checkpoint).resolve()
    hashes = {}

    def read(path):
        path = path.resolve()
        hashes[str(path)] = file_digest(path)
        return json.loads(path.read_text())

    complete = read(run / 'COMPLETE.json')
    if complete.get('status') != 'complete' or complete.get('filler_lengths') != [0, 20]:
        raise ValueError('a completed k=0/k=20 answer-token capture run is required')
    runtime = Path(complete['rows_path']).parent
    if read(runtime / 'COMPLETE.json') != complete or (runtime / 'FAILED.json').exists():
        raise ValueError('completed runtime pointer mismatch or failed capture run')
    if not Path(complete['rows_path']).is_file():
        raise FileNotFoundError(complete['rows_path'])
    validations = {str(k): read(runtime / f'validation_k{k}.json') for k in (0, 20)}
    if any(v.get('passed') is not True for v in validations.values()):
        raise ValueError('saved native final-readout validation did not pass')
    plan_path = run / 'preflight.json' if (run / 'preflight.json').exists() else runtime.parents[1] / 'preflight.json'
    plan = read(plan_path)
    if digest({k: v for k, v in plan.items() if k != 'config_hash'}) != complete['config_hash']:
        raise ValueError('capture plan checksum mismatch')
    if plan.get('config_hash') != complete['config_hash']:
        raise ValueError('capture configuration mismatch')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        path = checkpoint / name
        expected = plan['source_hashes'][str(path.relative_to(ROOT))]
        if file_digest(path) != expected:
            raise ValueError('capture checkpoint/tokenizer configuration differs')
        hashes[str(path)] = expected
    shard = checkpoint / 'model-00045-of-00048.safetensors'
    stat = shard.stat()
    if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != plan['checkpoint_files'][shard.name]:
        raise ValueError('capture vocabulary-head shard differs')
    if tokenizer is None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    records = read(runtime / 'captures.json')
    planned = {(c['filler_length'], c['cell_id']): c for c in plan['cells']}
    seen, paths, examples, refs, cohort = set(), set(), [], {}, {}
    for record in sorted(records, key=lambda r: (r['cell']['filler_length'], r['cell']['cell_id'])):
        cell, capture = record['cell'], record['capture']
        key = (cell['filler_length'], cell['cell_id'])
        if key in seen or key not in planned or cell != planned[key] or capture['cell_id'] != cell['cell_id']:
            raise ValueError('duplicate or incompatible full-prompt example')
        seen.add(key)
        validate_positions(cell, decode)
        targets = {name: {'token_id': token} for name, token in cell['target_token_ids'].items()}
        for name, value in zip(('A', 'X', 'A+X'), (cell['left_value'], cell['right_value'], cell['target'])):
            if tokenizer.encode(str(value), add_special_tokens=False).ids != [targets[name]['token_id']]:
                raise ValueError('full-prompt target tokenization mismatch')
        path = Path(capture['ranks']['0']['path']).resolve()
        if str(path) in paths:
            raise ValueError('capture file reused by different examples')
        paths.add(str(path))
        control = read(path.parent / 'control.json')
        acks = read(path.parent / 'acks.json')
        validate_ack_records(acks, control)
        if (control['layers'] or control['clean_capture'] is not None or control['donor_capture'] is not None
                or control['recomputation'] != 'full_downstream' or control['cell_id'] != cell['cell_id']
                or control['num_tokens'] != len(cell['input_ids'])
                or control['input_ids_hash'] != digest(cell['input_ids'])):
            raise ValueError('expected an unpatched matching full-prompt capture')
        for rank in range(4):
            ref = capture['ranks'][str(rank)]
            if acks[rank]['capture'] != ref:
                raise ValueError('capture/acknowledgement checksum mismatch')
            refs[str(Path(ref['path']).resolve())] = ref['sha256']
        response_path = Path(record['response']).resolve()
        if response_path != path.parent / 'response.json':
            raise ValueError('response does not belong to this capture request')
        response = read(response_path)
        correct = response['output_ids'][0] == targets['A+X']['token_id']
        cohort.setdefault(cell['filler_length'], [0, 0])[int(correct)] += 1
        expected_meta = {name: acks[0][name] for name in (
            'cell_id', 'rank', 'runtime_id', 'request_id', 'config_hash', 'num_tokens', 'positions')}
        for pos in cell['positions']:
            examples.append({**{name: cell[name] for name in (
                'panel_id', 'cell_id', 'split', 'row', 'col', 'left_value', 'right_value', 'target', 'filler_length')},
                'clean_correct': correct, 'position_label': pos['label'],
                'absolute_position': pos['absolute_position'], 'targets': targets,
                'pass_id': control['runtime_id'] + '/' + control['request_id'],
                'capture_path': str(path), 'capture_format': 'full-prompt',
                'runtime_id': control['runtime_id'], 'request_id': control['request_id'],
                'capture_config_hash': control['config_hash'], 'input_ids_sha256': control['input_ids_hash'],
                'num_tokens': len(cell['input_ids']), 'position_token_id': pos['token_id'],
                'position_token': pos['token'], '_capture_metadata': expected_meta,
                '_capture_expected_sha256': capture['ranks']['0']['sha256']})
    if seen != set(planned) or len(seen) != 2 * examples_per_length or len(examples) != 28 * examples_per_length:
        raise ValueError('expected 192 examples and 2,688 complete prompt positions')
    for k, (wrong, correct) in cohort.items():
        if (wrong + correct != examples_per_length or complete['cohorts'][f'k{k}_correct'] != correct
                or complete['cohorts'][f'k{k}_wrong'] != wrong):
            raise ValueError('saved native cohort counts differ')
    details = {'runtime': str(runtime), 'source_complete': complete, 'saved_native_validation': validations,
               'capture_files_sha256': refs, 'cohort_source': 'native response paired with each full-prompt capture',
               'positions_per_example': {'0': 4, '20': 24}, 'complete_answer_prefix': True}
    return examples, hashes, details


def selected_states(item, example, weights, layers):
    """Select exact token offsets from full or selected-position capture tensors."""
    meta = item['metadata']
    if any(meta.get(k) != value for k, value in example['_capture_metadata'].items()):
        raise ValueError('full-prompt capture metadata mismatch')
    positions = meta['positions']
    if len(set(positions)) != len(positions) or example['absolute_position'] not in positions:
        raise ValueError('missing or duplicate full-prompt capture position')
    offset = positions.index(example['absolute_position'])
    result = {}
    for layer in layers:
        state = item['states'].get(layer)
        if (state is None or tuple(state.shape) != (len(positions), weights.hc_mult, weights.hidden_size)
                or state.dtype != weights.lm_head_weight.dtype):
            raise ValueError(f'missing or incompatible L{layer} full-prompt residual')
        result[layer] = state[offset]
    return result

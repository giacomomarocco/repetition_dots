"""Pinned square J-Lens recurrence on the exact saved five-shot addition prompts.

prepare is metadata/tokenization only. run and export require an approved compute
allocation. Completion is published only after capture, score and notebook gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from filler.addition.one_fact import parse_answer
from filler.dsv4.lens_positions import prompt_position_labels, validate_positions
from filler.dsv4.patching import atomic_json, digest, file_digest
from filler.dsv4.workspace_jlens_artifact import DEFAULT_PATH, REVISION, SHA256, SIZE

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16'
SOURCE = ROOT / 'runs/deepseek-v4-flash/one-fact-addition-5shot-full'
DEFAULT_ROOT = ROOT / 'runs/deepseek-v4-flash/five-shot-square-recurrence'
NOTEBOOK = ROOT / 'notebooks/five_shot_square_recurrence.ipynb'
LAYERS = tuple(range(42))
N_FACTS = 262
READOUTS = 330120


def read(path):
    return json.loads(Path(path).read_text())


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(CHECKPOINT / 'tokenizer.json'))


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def normalize(text):
    return text.strip().casefold()


def vocabulary(tok):
    """Map every vocabulary ID, including variants never seen in a top-ten list."""
    tokens = [decode(tok, [i]) for i in range(tok.get_vocab_size())]
    names = sorted(set(map(normalize, tokens)))
    index = {name: i for i, name in enumerate(names)}
    special = {normalize(v.content) for v in tok.get_added_tokens_decoder().values() if v.special}
    def kind(word):
        if word in special:
            return 'special'
        if any(c.isalpha() for c in word):
            return 'textual'
        if any(c.isdigit() for c in word):
            return 'number'
        return 'punctuation' if word else 'whitespace'
    return {'tokens': tokens, 'groups': names, 'token_group': [index[normalize(t)] for t in tokens],
            'group_kind': [kind(name) for name in names]}


def make_cells(prompts, historical, tok, *, n_facts=N_FACTS):
    selected = [r for r in prompts if r['k'] in (0, 20)]
    history = {r['prompt_id']: r for r in historical if r['k'] in (0, 20)}
    if len(history) != len(selected):
        raise ValueError('missing or duplicate historical outcomes')
    cells, seen = [], set()
    for r in sorted(selected, key=lambda x: (x['pair_id'], x['k'])):
        k, text = r['k'], r['rendered_prompt']
        key = r['pair_id'], k
        if key in seen or text.count('<｜User｜>') != 6 or text.count('</think>') != 6:
            raise ValueError('duplicate prompt or missing five demonstrations')
        seen.add(key)
        old = history[r['prompt_id']]
        if any(old[name] != r[name] for name in r):
            raise ValueError('historical response/prompt mismatch')
        encoded = tok.encode(text, add_special_tokens=False)
        ids = encoded.ids
        labels = prompt_position_labels(k, suffix='five_shot')
        start = len(ids) - len(labels)
        positions = [{'label': label, 'absolute_position': p, 'token_id': ids[p],
                      'token': decode(tok, [ids[p]])} for p, label in enumerate(labels, start)]
        # Validate boundaries against saved text, including punctuation/newline merges.
        target_start = text.rindex('<｜User｜>')
        question_end = text.rindex(r['question']) + len(r['question'])
        q0, q1 = encoded.offsets[start]
        if not (target_start < q0 < question_end <= q1 and text[q1:encoded.offsets[start+1][0]].strip() == ''):
            raise ValueError('last question token does not end the target question')
        targets = {}
        for name, value in zip(('A', 'X', 'A+X'), (r['answer_value'], r['addend'], r['target'])):
            target_ids = tok.encode(str(value), add_special_tokens=False).ids
            if len(target_ids) != 1:
                raise ValueError(f'{name} is not a single token')
            targets[name] = target_ids[0]
        demos_start = text.index('<｜User｜>')
        demos = [token for token, (a, b) in zip(ids, encoded.offsets) if demos_start <= a and b <= target_start]
        target = [token for token, (a, b) in zip(ids, encoded.offsets) if target_start <= a and b <= question_end + 1]
        cell = {**r, 'cell_id': r['prompt_id'], 'filler_length': k, 'position_suffix': 'five_shot',
                'input_ids': ids, 'positions': positions, 'target_token_ids': targets,
                'category': r['fact_id'].split(':')[0].removesuffix('.json'),
                'demonstration_token_ids': sorted(set(demos)), 'question_token_ids': sorted(set(target)),
                'historical_correct': old['correct'], 'historical_response': old['response']}
        validate_positions(cell, lambda v: decode(tok, v))
        if (parse_answer(old['response']) == r['target']) != old['correct']:
            raise ValueError('historical strict-answer label mismatch')
        cells.append(cell)
    pairs = {c['pair_id'] for c in cells}
    if len(pairs) != n_facts or seen != {(p, k) for p in pairs for k in (0, 20)}:
        raise ValueError('missing paired fact conditions')
    if len({c['fact_id'] for c in cells}) != n_facts or len({c['prompt_id'] for c in cells}) != 2*n_facts:
        raise ValueError('facts/prompt IDs are not unique')
    for a, b in zip(cells[::2], cells[1::2]):
        if any(a[key] != b[key] for key in ('pair_id', 'fact_id', 'question', 'answer_value', 'addend', 'target')):
            raise ValueError('paired fact metadata mismatch')
        # All six user turns change filler length; everything else must be exact.
        if b['rendered_prompt'].replace('. ' * 19 + '.\nAnswer:', 'Answer:') != a['rendered_prompt']:
            raise ValueError('paired prompts differ beyond complete filler condition')
    return cells


def source_paths():
    names = ['filler/dsv4/five_shot_recurrence.py', 'filler/dsv4/recurrence_runtime.py',
             'filler/dsv4/recurrence_analysis.py', 'filler/dsv4/recurrence_export.py',
             'filler/dsv4/lens_positions.py', 'filler/dsv4/jlens.py', 'filler/dsv4/lens.py',
             'filler/dsv4/workspace_jlens_artifact.py',
             'filler/dsv4/factorial.py', 'filler/dsv4/campaign_hook.py', 'filler/dsv4/patching.py',
             'filler/addition/one_fact.py', 'scripts/dsv4/one_fact_patching.py',
             'scripts/dsv4/run_activation_patching.py', 'run_deepseek_v4_a100.sh',
             'run_five_shot_recurrence_allocation.sh', 'scripts/dsv4/five_shot_recurrence.py',
             'tests/test_five_shot_recurrence.py', str(NOTEBOOK.relative_to(ROOT)),
             'ports/sglang/python/sglang/srt/models/deepseek_v4.py',
             'ports/sglang/python/sglang/srt/layers/mhc_head.py',
             'ports/sglang/python/sglang/srt/layers/layernorm.py',
             'ports/sglang/python/sglang/srt/layers/sampler.py',
             'ports/deepseek-v4-a100-sglang/dsv4_a100_patch/patch.py']
    return [ROOT / name for name in names] + [SOURCE / n for n in ('prompts.json', 'results.json', 'run_config.json')] + [CHECKPOINT / n for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')]


def prepare(root):
    config = read(SOURCE / 'run_config.json')
    if config['decoding'] != {'temperature': 0, 'max_new_tokens': 8} or len(config['demonstrations']) != 5:
        raise ValueError('unexpected saved demonstration/decoding protocol')
    tok = tokenizer()
    cells = make_cells(read(SOURCE / 'prompts.json'), read(SOURCE / 'results.json'), tok)
    from filler.dsv4.jlens import load_workspace_jlens
    lens = load_workspace_jlens(DEFAULT_PATH, validate_values=False)
    if DEFAULT_PATH.stat().st_size != SIZE or lens.source_layers != LAYERS:
        raise ValueError('square artifact metadata differs')
    reference = ROOT / 'ports/jacobian-lens-open-frontier'
    ref = read(reference / 'SOURCE.json')
    from scripts.dsv4.prepare_jlens import CODE_REVISION
    if ref['revision'] != CODE_REVISION:
        raise ValueError('reference revision differs')
    paths = source_paths() + [reference / 'SOURCE.json'] + [reference / name for name in ref['files']]
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in paths}
    for name, expected in ref['files'].items():
        if hashes[str((reference / name).relative_to(ROOT))] != expected:
            raise ValueError('reference code checksum mismatch')
    shards = {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
              for p in sorted(CHECKPOINT.glob('*.safetensors'))}
    if len(shards) != 48:
        raise ValueError('expected 48 model shards')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'cells': cells,
            'source_hashes': hashes, 'checkpoint_files': shards, 'run_config': config,
            'vocabulary_sha256': digest(vocabulary(tok)),
            'square': {'path': str(DEFAULT_PATH), 'revision': REVISION, 'sha256': SHA256,
                       'size': SIZE, 'mtime_ns': DEFAULT_PATH.stat().st_mtime_ns},
            'layers': list(LAYERS), 'positions': sum(len(c['positions']) for c in cells),
            'readouts': sum(len(c['positions']) for c in cells)*42,
            'bootstrap': {'resamples': 2000, 'seed': 42, 'unit': 'paired fact'},
            'precision': 'FP32 stream mean/transport; BF16 HF norm/head; FP32 log_softmax',
            'ties': 'descending logit, ascending token ID; ordinal target ranks',
            'cohorts': 'strict integer parse of each capture request response',
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    if plan['readouts'] != READOUTS:
        raise ValueError('expected 330120 readouts')
    plan['config_hash'] = digest(plan)
    atomic_json(root / 'preflight.json', plan)
    print(json.dumps({k: plan[k] for k in ('config_hash', 'positions', 'readouts')}, indent=2))
    return plan


def verify_plan(plan):
    if digest({k: v for k, v in plan.items() if k != 'config_hash'}) != plan['config_hash']:
        raise ValueError('plan checksum mismatch')
    for name, sha in plan['source_hashes'].items():
        if file_digest(ROOT / name) != sha:
            raise ValueError(f'input changed since prepare: {name}')
    for name, expected in plan['checkpoint_files'].items():
        stat = (CHECKPOINT / name).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('checkpoint changed since prepare')
    stat = DEFAULT_PATH.stat()
    if stat.st_size != plan['square']['size'] or stat.st_mtime_ns != plan['square']['mtime_ns']:
        raise ValueError('square artifact changed since prepare')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'export'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=30002)
    parser.add_argument('--worker', choices=('score', 'mass'), help=argparse.SUPPRESS)
    parser.add_argument('--rank', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
        return
    from filler.dsv4.recurrence_runtime import require_compute, run, score_worker
    require_compute()
    if args.worker:
        score_worker(root, args.rank, mass=args.worker == 'mass')
    elif args.action == 'run':
        run(root, args.port)
    else:
        from filler.dsv4.recurrence_export import export
        export(root)


if __name__ == '__main__':
    main()

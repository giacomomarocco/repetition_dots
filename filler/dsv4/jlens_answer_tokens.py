"""Compare published J-Lenses on complete saved Answer/colon/space captures."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from filler.dsv4.jlens import (load_jlens, load_workspace_jlens, SOURCE_LAYERS,
                              WORKSPACE_SOURCE_LAYERS)
from filler.dsv4.jlens_full_prompt import collect_full_prompt_examples, selected_states
from filler.dsv4.jlens_top_object import ROOT, ROWS_NAME, collect_examples, write_scores_many
from filler.dsv4.jlens_validation import (digest, write_json, source_provenance, read_examples,
                                        validate_paris)
from filler.dsv4.jlens_comparison import (index_rows, read_jsonl, validate_token_metrics,
    check_baseline_provenance, compare_baseline, LIMITATION)
from filler.dsv4.lens import load_checkpoint_readout
from filler.dsv4.top_object_heatmap import canonical_numeric_token_ids
from filler.dsv4.workspace_jlens_artifact import verify_artifact, REPOSITORY, REVISION


def preflight(args):
    if args.output_dir.exists():
        raise FileExistsError('choose a fresh output directory')
    examples, hashes, details = collect_full_prompt_examples(args.answer_token_run, args.checkpoint)
    load_workspace_jlens(args.square_lens, validate_values=False)
    load_jlens(args.rectangular_lens, validate_values=False)
    from scripts.dsv4.prepare_jlens import LENS_SIZE
    from filler.dsv4.workspace_jlens_artifact import SIZE
    if args.square_lens.stat().st_size != SIZE or args.rectangular_lens.stat().st_size != LENS_SIZE:
        raise ValueError('artifact size mismatch')
    for path in (args.pilot_manifest, args.native_response, args.baseline_dir / ROWS_NAME,
                 args.baseline_dir / 'COMPLETE.json', args.baseline_dir / 'provenance.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = json.loads((args.baseline_dir / 'COMPLETE.json').read_text())
    if baseline.get('passed') is not True or baseline.get('rows') != 86688:
        raise ValueError('completed rectangular regression baseline required')
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if (config['num_hidden_layers'], config['hidden_size'], config['hc_mult']) != (43, 4096, 4):
        raise ValueError('incompatible model configuration')
    plan = {'capture_format': 'full-prompt', 'capture_files': 192, 'capture_positions': len(examples),
            'square_rows': len(examples) * 42, 'rectangular_rows': len(examples) * 21,
            'matched_pairs': len(examples) * 21, 'filler_lengths': [0, 20],
            'positions_per_example': {'0': 4, '20': 24}, 'examples_by_filler_length': {'0': 96, '20': 96},
            'answer_positions': ['answer_word', 'answer_colon', 'answer_prompt'],
            'square_layers': list(WORKSPACE_SOURCE_LAYERS), 'comparison_layers': list(SOURCE_LAYERS),
            'complete_answer_prefix': True, 'metadata_only': True,
            'batch_size': args.batch_size, 'threads': args.threads, 'device': 'cpu',
            'output_dir': str(args.output_dir)}
    return examples, hashes, details, plan


def legacy_probe(args, lens, weights, numeric_ids, output, provenance):
    """Keep a fixed old-capture regression; new runtime activations need not match old ones."""
    examples, hashes = collect_examples(args.grid_root, args.capture_root, [0, 5, 10, 20])
    baseline_provenance = json.loads((args.baseline_dir / 'provenance.json').read_text())
    check_baseline_provenance(baseline_provenance, {**provenance, 'manifest_sha256': hashes})
    probe = examples[:args.batch_size]
    path = output / 'legacy_rectangular_probe.jsonl'
    capture_hashes = {}
    write_scores_many(probe, {'rectangular': lens}, weights, numeric_ids,
                      {'rectangular': path}, batch_size=args.batch_size, capture_hashes=capture_hashes)
    current = index_rows(read_jsonl(path), probe, SOURCE_LAYERS, 'jlens', metrics=False, hashes=capture_hashes)
    baseline = index_rows(read_jsonl(args.baseline_dir / ROWS_NAME), examples, SOURCE_LAYERS, 'jlens', metrics=False)
    check = compare_baseline(current, {k: baseline[k] for k in current})
    check.update(scope='fixed historical capture regression; new full-prompt rows are a different capture source',
                 baseline_total_rows=len(baseline), probe_capture_positions=len(probe))
    write_json(output / 'baseline_validation.json', check)
    if not check['passed']:
        raise ValueError('historical rectangular regression probe failed')
    for name in ('COMPLETE.json', 'provenance.json', ROWS_NAME):
        path = args.baseline_dir / name
        hashes[str(path)] = digest(path)
    return check, {**hashes, **capture_hashes}


@torch.inference_mode()
def run(args):
    examples, metadata_hashes, details, plan = preflight(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight:
        return
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('run through srun on an approved CPU allocation; use --preflight on login')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from filler.dsv4.compare_jlenses import validate_pilot, export_results, execute_notebook
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = {**plan, 'metadata_only': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'arguments': {**{k: str(v) for k, v in vars(args).items()}, 'device': 'cpu'},
        'command': sys.argv, 'python': sys.version, 'torch': torch.__version__,
        'job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
        'allocation': {k: os.environ.get(k) for k in ('SLURM_JOB_ACCOUNT', 'SLURM_JOB_QOS', 'SLURM_CPUS_PER_TASK')},
        'capture_source': details, 'manifest_sha256': metadata_hashes,
        'transport_dtype': 'float32', 'readout_dtype': 'bfloat16', 'norm_rounding': 'published HF',
        'argmax_ties': 'lowest token ID', 'rank': '1 + count(logits > target_logit)',
        'cohorts': details['cohort_source'], 'limitation': LIMITATION,
        'coverage_description': 'Complete answer-prefix coverage for k=0 and k=20: last question, every filler, '
            'Answer, colon and trailing space. All positions within an example use the same saved full-prompt '
            'capture and its native correctness label. k=5 and k=10 remain in the separate historical artifact; '
            'they are not included in this complete-prefix run.'}
    write_json(output / 'STARTED.json', provenance)
    try:
        provenance.update(source_provenance(args.rectangular_lens))
        provenance['square_artifact'] = {'repository': REPOSITORY, 'revision': REVISION,
                                         'sha256': verify_artifact(args.square_lens)}
        lenses = {'square': load_workspace_jlens(args.square_lens), 'rectangular': load_jlens(args.rectangular_lens)}
        provenance['square_fit'] = lenses['square'].provenance
        # Preserve the exact code used, including the notebook, independently of later edits.
        snapshot = output / 'source_snapshot'
        source_hashes = {}
        for name in ('filler/dsv4/jlens.py', 'filler/dsv4/lens.py', 'filler/dsv4/jlens_top_object.py',
                     'filler/dsv4/jlens_comparison.py', 'filler/dsv4/jlens_full_prompt.py',
                     'filler/dsv4/jlens_answer_tokens.py', 'filler/dsv4/compare_jlenses.py',
                     'filler/dsv4/jlens_validation.py', 'filler/dsv4/lens_positions.py',
                     'filler/dsv4/top_object_heatmap.py', 'notebooks/one_fact_jlens_comparison.ipynb',
                     'run_jlens_comparison_allocation.sh'):
            path = ROOT / name
            raw = path.read_bytes()
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        weights = load_checkpoint_readout(args.checkpoint, device='cpu')
        if weights.lm_head_weight.dtype != torch.bfloat16:
            raise ValueError('expected validated BF16 vocabulary head')
        from transformers import AutoTokenizer
        import transformers
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        vocab_size = len(weights.lm_head_weight)
        if len(tokenizer) != vocab_size:
            raise ValueError('vocabulary-size mismatch')
        numeric_ids = canonical_numeric_token_ids(tokenizer)
        provenance.update(vocab_size=vocab_size, numeric_token_ids=numeric_ids, transformers=transformers.__version__)
        provenance['readout_tensor_sha256'] = {name: hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
            for name, t in vars(weights).items() if isinstance(t, torch.Tensor)}
        source_hashes[str(args.checkpoint / 'model-00045-of-00048.safetensors')] = digest(args.checkpoint / 'model-00045-of-00048.safetensors')
        # The completed GPU-native validation is reused only with all underlying capture bytes intact.
        for path, expected in details['capture_files_sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'saved capture checksum mismatch: {path}')
        old_rows_path = Path(details['source_complete']['rows_path'])
        if digest(old_rows_path) != details['source_complete']['rows_sha256']:
            raise ValueError('completed native lens rows checksum mismatch')
        source_hashes[str(old_rows_path)] = details['source_complete']['rows_sha256']
        pilot_examples, pilot_captures, native, pilot_hashes = read_examples(args.capture_root, args.pilot_manifest, args.native_response)
        native_check = validate_paris(pilot_captures[0][42], native, weights)
        write_json(output / 'native_validation.json', native_check)
        if not native_check['passed']:
            raise ValueError('native Paris final-readout gate failed')
        checks = validate_pilot(lenses, weights, pilot_captures, output)
        del pilot_captures
        new_pilot = []
        for k in (0, 20):
            cell_id = next(e['cell_id'] for e in examples if e['filler_length'] == k)
            chosen = [e for e in examples if e['filler_length'] == k and e['cell_id'] == cell_id
                      and e['position_label'] in ('last_question', 'answer_word', 'answer_colon', 'answer_prompt')]
            item = torch.load(chosen[0]['capture_path'], map_location='cpu', weights_only=True)
            new_pilot.extend(selected_states(item, e, weights, range(42)) for e in chosen)
        (output / 'full_prompt_pilot').mkdir()
        full_checks = validate_pilot(lenses, weights, new_pilot, output / 'full_prompt_pilot')
        del new_pilot, item
        baseline_check, baseline_hashes = legacy_probe(args, lenses['rectangular'], weights, numeric_ids, output, provenance)
        provenance['baseline_description'] = (f"Both rectangular argmax scopes reproduce {baseline_check['rows']:,} historical "
            'probe rows exactly. The expanded rows use a different full-prompt capture source and are validated '
            'against independent transport/HF readout references, not required to reproduce old activations.')
        hashes = {**metadata_hashes, **source_hashes, **pilot_hashes, **baseline_hashes}
        provenance['input_and_source_sha256'] = hashes
        write_json(output / 'provenance.json', provenance)
        write_json(output / 'examples.json', [{k: v for k, v in e.items() if not k.startswith('_')} for e in examples])
        paths = {name: output / f'{name}_rows.jsonl' for name in lenses}
        capture_hashes = {}
        counts = write_scores_many(examples, lenses, weights, numeric_ids, paths,
            batch_size=args.batch_size, target_metrics=True, capture_hashes=capture_hashes)
        write_json(output / 'capture_sha256.json', capture_hashes)
        if counts != {'square': plan['square_rows'], 'rectangular': plan['rectangular_rows']}:
            raise ValueError('incomplete full-prompt scoring grid')
        square = index_rows(read_jsonl(paths['square']), examples, WORKSPACE_SOURCE_LAYERS, 'jlens_workspace_mean', hashes=capture_hashes)
        rectangular = index_rows(read_jsonl(paths['rectangular']), examples, SOURCE_LAYERS, 'jlens', hashes=capture_hashes)
        for rows in (square, rectangular):
            validate_token_metrics(rows, vocab_size=vocab_size, numeric_ids=numeric_ids)
        for path, expected in {**hashes, **details['capture_files_sha256']}.items():
            if digest(Path(path)) != expected:
                raise ValueError(f'input changed during comparison: {path}')
        verify_artifact(args.square_lens)
        source_provenance(args.rectangular_lens)
        del lenses, weights
        exports = export_results(output, square, rectangular, tokenizer, provenance)
        validation = {'passed': True, 'complete_answer_prefix': True, 'filler_lengths': [0, 20],
            'captures': len(capture_hashes), 'capture_positions': len(examples), 'rows': counts,
            'matched_pairs': len(rectangular), 'inputs_unchanged': True, 'native': native_check,
            'saved_native_validation': details['saved_native_validation'], 'pilot': checks,
            'full_prompt_pilot': full_checks, 'baseline_probe': baseline_check, **exports}
        write_json(output / 'validation.json', validation)
        execute_notebook(output)
        validation['executed_notebook'] = 'one_fact_jlens_comparison.executed.ipynb'
        write_json(output / 'COMPLETE.json', {**validation, 'elapsed_seconds': time.monotonic() - started})
        print(f'Complete: {output / "REPORT.md"}', flush=True)
    except Exception as error:
        write_json(output / 'FAILED.json', {'type': type(error).__name__, 'error': str(error),
                                          'elapsed_seconds': time.monotonic() - started})
        raise

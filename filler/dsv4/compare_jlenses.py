"""Validate and score both published J-Lenses on the same saved activation grid."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from filler.dsv4.jlens import (SOURCE_LAYERS, WORKSPACE_SOURCE_LAYERS, load_jlens,
    load_workspace_jlens, published_readout_fixture, transport, unembed_transported)
from filler.dsv4.jlens_top_object import (ROOT, SAVED, OUTPUT, ROWS_NAME, collect_examples,
    write_scores_many)
from filler.dsv4.jlens_validation import (digest, write_json, source_provenance,
    read_examples, validate_paris)
from filler.dsv4.lens import load_checkpoint_readout
from filler.dsv4.top_object_heatmap import canonical_numeric_token_ids
from filler.dsv4.workspace_jlens_artifact import DEFAULT_PATH, verify_artifact, REPOSITORY, REVISION
from filler.dsv4.jlens_comparison import (OBJECTS, METRICS, COHORTS, LIMITATION, read_jsonl,
    index_rows, check_baseline_provenance, compare_baseline, pair_rows, summarize,
    square_summary, write_tables, plot_comparison, validate_token_metrics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-grid', action='store_true',
                        help='explicitly reproduce the old grid, which omits Answer/colon')
    parser.add_argument('--answer-token-run', type=Path, default=ROOT / 'runs/deepseek-v4-flash/answer-token-heatmap',
                        help='completed full-prompt captures (default input)')
    parser.add_argument('--grid-root', type=Path, default=SAVED / 'filler-grid-expanded')
    parser.add_argument('--capture-root', type=Path, default=SAVED / 'captures')
    parser.add_argument('--pilot-manifest', type=Path, default=SAVED / 'filler-pilot-v2/manifest.json')
    parser.add_argument('--native-response', type=Path, default=SAVED / 'native_response.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16')
    parser.add_argument('--square-lens', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--rectangular-lens', type=Path, default=ROOT / 'model/jacobian-lens-deepseek-v4-flash-0731/lens.pt')
    parser.add_argument('--baseline-dir', type=Path, default=OUTPUT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1:
        parser.error('threads and batch size must be positive')
    return args


def preflight(args):
    """Read small manifests and mmap lens metadata, never tensor values or hashes."""
    if args.output_dir.exists():
        raise FileExistsError('choose a fresh --output-dir')
    examples, hashes = collect_examples(args.grid_root, args.capture_root, [0, 5, 10, 20])
    counts = {str(k): len({e['cell_id'] for e in examples if e['filler_length'] == k}) for k in (0, 5, 10, 20)}
    if len(examples) != 4128 or set(counts.values()) != {96}:
        raise ValueError('expected all 4,128 saved captures and 96 examples per filler length')
    for path in (args.square_lens, args.rectangular_lens, args.pilot_manifest, args.native_response,
                 args.checkpoint / 'model-00045-of-00048.safetensors',
                 args.checkpoint / 'config.json', args.checkpoint / 'tokenizer.json',
                 args.baseline_dir / ROWS_NAME, args.baseline_dir / 'provenance.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    from scripts.dsv4.prepare_jlens import LENS_SIZE
    from filler.dsv4.workspace_jlens_artifact import SIZE
    if args.square_lens.stat().st_size != SIZE or args.rectangular_lens.stat().st_size != LENS_SIZE:
        raise ValueError('artifact size differs from release')
    load_workspace_jlens(args.square_lens, validate_values=False)
    load_jlens(args.rectangular_lens, validate_values=False)
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if (config['num_hidden_layers'], config['hidden_size'], config['hc_mult']) != (43, 4096, 4):
        raise ValueError('unexpected checkpoint configuration')
    baseline = json.loads((args.baseline_dir / 'COMPLETE.json').read_text())
    if baseline.get('passed') is not True or baseline.get('rows') != 86688 or baseline.get('captures') != 4128:
        raise ValueError('rectangular baseline is incomplete')
    old = json.loads((args.baseline_dir / 'provenance.json').read_text())
    if old['manifest_sha256'] != hashes:
        raise ValueError('incompatible baseline manifest provenance')
    plan = {'captures': len(examples), 'square_rows': len(examples) * 42,
            'rectangular_rows': len(examples) * 21, 'matched_pairs': len(examples) * 21,
            'square_layers': list(WORKSPACE_SOURCE_LAYERS), 'comparison_layers': list(SOURCE_LAYERS),
            'examples_by_filler_length': counts, 'output_dir': str(args.output_dir),
            'threads': args.threads, 'batch_size': args.batch_size, 'device': 'cpu',
            'metadata_only': True}
    return examples, hashes, plan


@torch.inference_mode()
def validate_pilot(lenses, weights, captures, output):
    """Independent transport and actual pinned HF adapter at every artifact layer."""
    from jlens.lens import JacobianLens
    reference_head = published_readout_fixture(weights)
    checks = []
    for name, lens in lenses.items():
        for layer in lens.source_layers:
            states = torch.stack([capture[layer] for capture in captures])
            if name == 'square':
                # Explicit stream sum and einsum are independent of production mean/F.linear.
                mean = sum(states[:, stream, :].float() for stream in range(4)) / 4
                expected = torch.einsum('ij,bj->bi', lens.jacobians[layer].float(), mean)
            else:
                reference = JacobianLens({layer: lens.jacobians[layer]}, n_prompts=lens.n_prompts,
                                         d_model=lens.d_model, d_source=lens.d_source)
                expected = reference.transport(states.float(), layer)
            actual = transport(states, lens, layer)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            expected_logits = reference_head.unembed(expected, collapse=False).float()
            actual_logits = unembed_transported(actual, weights)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)
            checks.append({'lens': name, 'layer': layer, 'positions': len(captures), 'passed': True,
                           'transport_max_abs_error': float((actual - expected).abs().max()),
                           'readout_max_abs_error': float((actual_logits - expected_logits).abs().max())})
            write_json(output / 'pilot_checks.json', checks)
            print(f'Pilot {name} L{layer}: transport and HF readout passed', flush=True)
    return checks


def export_results(output, square, rectangular, tokenizer, provenance):
    import matplotlib.pyplot as plt
    pairs = list(pair_rows(square, rectangular))
    token_text = {}
    for row in pairs:
        for name in ('square_top_token_id', 'rectangular_top_token_id',
                     'square_top_numeric_token_id', 'rectangular_top_numeric_token_id', 'target_token_id'):
            token = row[name]
            if token not in token_text:
                token_text[token] = tokenizer.decode([token], skip_special_tokens=False)
            row[name.removesuffix('_id') + '_text'] = token_text[token]
    summary = summarize(pairs)
    square_stats = square_summary(square)
    write_tables(output / 'paired', pairs)
    write_tables(output / 'summary', summary)
    write_tables(output / 'square_summary', square_stats)
    lengths = sorted({r['filler_length'] for r in summary})
    default_length = 20 if 20 in lengths else lengths[-1]
    cohort_counts = {f'{k}/{c}': next((r['n'] for r in summary if r['filler_length'] == k and r['cohort'] == c), 0)
                     for k in lengths for c in COHORTS}
    for k in lengths:
        if cohort_counts[f'{k}/all'] != cohort_counts[f'{k}/correct'] + cohort_counts[f'{k}/wrong']:
            raise ValueError('cohort counts do not partition the saved examples')
    figure_count = 0
    for k in lengths:
        for cohort in COHORTS:
            for metric in METRICS:
                for square_only, stats, prefix in ((False, summary, 'comparison'), (True, square_stats, 'square')):
                    fig = plot_comparison(stats, k, cohort, metric, square_only=square_only)
                    stem = output / f'{prefix}_k{k}_{cohort}_{metric}'
                    fig.savefig(stem.with_suffix('.png'), dpi=130)
                    fig.savefig(stem.with_suffix('.pdf'))
                    plt.close(fig)
                    figure_count += 1
            print(f'Exported filler={k}, cohort={cohort}', flush=True)
    # Summarize all matched positions/layers with equal pair weighting.
    overview = []
    for target in OBJECTS:
        rows = [r for r in pairs if r['filler_length'] == default_length and r['target_name'] == target]
        n = len(rows)
        values = {name: sum(r[name] for r in rows) / n
                  for name in ('square_top1', 'rectangular_top1', 'difference_top1',
                               'square_mrr', 'rectangular_mrr', 'difference_mrr',
                               'square_logprob', 'rectangular_logprob', 'difference_logprob', 'full_agreement')}
        overview.append({'target': target, 'pairs': n, **values})
    write_tables(output / f'overview_k{default_length}_all', overview)
    lines = ['# Published J-Lens comparison', '', f"Started: {provenance['started_at']}", '',
             f'Scored {len(square) // 42:,} shared prompt positions: {len(square):,} square rows and {len(rectangular):,} rectangular rows. '
             f'Direct comparisons use {len(rectangular):,} matched position/layer pairs in layers 19–39. '
             'Square-only exports retain all layers 0–41.', '',
             'Both release checksums, all matrix values, the square layer-41 identity anchor, '
             'the saved native final-readout gate and both pilot transport/HF references passed. ',
             provenance.get('baseline_description', 'Both rectangular argmax scopes exactly reproduce the completed baseline.'), '',
             provenance.get('coverage_description', 'Historical grid: Answer/colon positions are absent.'), '',
             'Cohorts use the original saved model correctness labels. Every heatmap cell uses '
             'the number of examples in that cohort as denominator; empty cohorts are explicitly blank. '
             'Rank = 1 + count(logits > target_logit). A tied rank-one target need not be the '
             'lowest-ID argmax. Log-probability is over the complete vocabulary. '
             'Differences are square minus rectangular; positive values mean greater target readability.', '',
             f'## Filler length {default_length}, all examples', '',
             'Means below weight every matched position/layer pair equally.', '',
             '| Target | Square top-1 | Rectangular top-1 | Difference | Square MRR | Rectangular MRR |',
             '|---|---:|---:|---:|---:|---:|']
    for row in overview:
        lines.append(f"| {row['target']} | {row['square_top1']:.6f} | {row['rectangular_top1']:.6f} | "
                     f"{row['difference_top1']:+.6f} | {row['square_mrr']:.6f} | {row['rectangular_mrr']:.6f} |")
    lines += ['', f'![Default comparison](comparison_k{default_length}_all_top1.png)', '', LIMITATION, '',
              '`paired.csv` / `paired.json`: exact ranks, full-vocabulary log-probabilities, decoded '
              'winning tokens and paired metrics for every target. `summary.csv` / `summary.json`: '
              'per-cell rates, MRR, log-probabilities, agreement rates and counts where only one '
              'lens selects the target in each argmax scope. `square_summary.*`: all square layers.', '',
              '`provenance.json`, `validation.json`, `baseline_validation.json`, `pilot_checks.json` '
              'and `capture_sha256.json` preserve input integrity, software, precision and validation. '
              'The executed notebook reads these results without loading a model or scoring.', '',
              'Baseline provenance did not include capture or head tensor hashes. Current inputs are '
              'hashed and rechecked; exact baseline argmax reproduction supplies the retrospective '
              'compatibility check, not proof that historical tensor bytes were identical.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return {'paired_target_rows': len(pairs), 'figures': figure_count, 'cohort_counts': cohort_counts}


def execute_notebook(output):
    """Execute trusted local notebook cells in-process, retaining figures/tables.

    Match the existing project exporter without installing a Jupyter stack on
    the shared system. Widgets remain rerunnable; static defaults are embedded.
    """
    import base64
    import contextlib
    import io
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import pandas as pd
    source = ROOT / 'notebooks/one_fact_jlens_comparison.ipynb'
    notebook = json.loads(source.read_text())
    namespace = {}
    outputs = []

    def capture_display(value):
        if isinstance(value, Figure):
            buffer = io.BytesIO()
            value.savefig(buffer, format='png', dpi=120)
            data = {'image/png': base64.b64encode(buffer.getvalue()).decode()}
        elif isinstance(value, pd.DataFrame):
            data = {'text/html': value.to_html(index=False), 'text/plain': value.to_string(index=False)}
        else:
            data = {'text/plain': repr(value)}
        outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': data})

    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        outputs = []
        code = ''.join(cell['source']).replace('RUN = DEFAULT_RUN', f'RUN = Path({str(output.resolve())!r})')
        # Capture explicit display calls just as the existing exporter captures plt.show.
        executable = code.replace('from IPython.display import display', '')
        executable = executable.replace('load_completed(RUN)', 'load_completed(RUN, allow_validated=True)')
        namespace['display'] = capture_display
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(executable, f'{source}:cell{index}', 'exec'), namespace)
        cell['source'] = code.splitlines(keepends=True)
        cell['execution_count'] = count
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': stdout.getvalue()}]
                           if stdout.getvalue() else []) + outputs
    plt.close('all')
    write_json(output / 'one_fact_jlens_comparison.executed.ipynb', notebook)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if not args.legacy_grid:
        from filler.dsv4.jlens_answer_tokens import run
        return run(args)
    examples, manifests, plan = preflight(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight:
        return
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('run through srun on an approved compute allocation; use --preflight on login')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = {**plan, 'metadata_only': False, 'started_at': datetime.now(timezone.utc).isoformat(),
                  'arguments': {**{k: str(v) for k, v in vars(args).items()}, 'device': 'cpu'},
                  'command': sys.argv, 'torch': torch.__version__, 'python': sys.version,
                  'node': socket.gethostname(), 'job_id': os.environ['SLURM_JOB_ID'],
                  'allocation': {k: os.environ.get(k) for k in ('SLURM_JOB_ACCOUNT', 'SLURM_JOB_QOS', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK')},
                  'manifest_sha256': manifests, 'transport_dtype': 'float32',
                  'readout_dtype': 'bfloat16', 'norm_rounding': 'published HF',
                  'rank': '1 + count(logits > target_logit)', 'argmax_ties': 'lowest token ID',
                  'cohorts': 'original saved clean_correct labels', 'limitation': LIMITATION}
    write_json(output / 'STARTED.json', provenance)
    try:
        provenance.update(source_provenance(args.rectangular_lens))
        provenance['square_artifact'] = {'repository': REPOSITORY, 'revision': REVISION,
                                         'sha256': verify_artifact(args.square_lens)}
        lenses = {'square': load_workspace_jlens(args.square_lens), 'rectangular': load_jlens(args.rectangular_lens)}
        provenance['square_fit'] = lenses['square'].provenance
        pilot_examples, captures, native, input_hashes = read_examples(args.capture_root, args.pilot_manifest, args.native_response)
        weights = load_checkpoint_readout(args.checkpoint, device='cpu')
        if weights.lm_head_weight.dtype != torch.bfloat16:
            raise ValueError('expected BF16 vocabulary head')
        from transformers import AutoTokenizer
        import transformers
        provenance['transformers'] = transformers.__version__
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        if len(weights.lm_head_weight) != json.loads((args.checkpoint / 'config.json').read_text())['vocab_size']:
            raise ValueError('vocabulary head/config size mismatch')
        vocab_size = len(weights.lm_head_weight)
        if len(tokenizer) != vocab_size:
            raise ValueError('tokenizer/head vocabulary size mismatch')
        provenance['vocab_size'] = vocab_size
        numeric_ids = canonical_numeric_token_ids(tokenizer)
        provenance['numeric_token_ids'] = numeric_ids
        provenance['readout_tensor_sha256'] = {name: hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
                                               for name, t in vars(weights).items() if isinstance(t, torch.Tensor)}
        for relative in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model-00045-of-00048.safetensors'):
            path = args.checkpoint / relative
            input_hashes[str(path)] = digest(path)
        for relative in ('filler/dsv4/jlens.py', 'filler/dsv4/jlens_top_object.py',
                         'filler/dsv4/jlens_comparison.py', 'filler/dsv4/compare_jlenses.py',
                         'filler/dsv4/jlens_validation.py', 'filler/dsv4/lens.py',
                         'filler/dsv4/top_object_heatmap.py', 'filler/dsv4/workspace_jlens_artifact.py',
                         'notebooks/one_fact_jlens_comparison.ipynb', 'run_jlens_comparison_allocation.sh'):
            path = ROOT / relative
            input_hashes[str(path)] = digest(path)
        for relative in (ROWS_NAME, 'COMPLETE.json', 'provenance.json'):
            path = args.baseline_dir / relative
            input_hashes[str(path)] = digest(path)
        provenance['input_and_source_sha256'] = input_hashes
        for example in examples:
            for name, value in zip(OBJECTS, (example['left_value'], example['right_value'], example['target'])):
                if tokenizer.encode(str(value), add_special_tokens=False) != [example['targets'][name]['token_id']]:
                    raise ValueError('saved target tokenization mismatch')
        for name, value in (('A', 57), ('X', 11), ('A+X', 68)):
            if tokenizer.encode(str(value), add_special_tokens=False) != [pilot_examples[1]['target_token_ids'][name]]:
                raise ValueError('pilot target tokenization mismatch')
        baseline_provenance = json.loads((args.baseline_dir / 'provenance.json').read_text())
        check_baseline_provenance(baseline_provenance, provenance)
        write_json(output / 'provenance.json', provenance)
        native_check = validate_paris(captures[0][42], native, weights)
        write_json(output / 'native_validation.json', native_check)
        if not native_check['passed']:
            raise AssertionError('native Paris final-readout gate failed')
        checks = validate_pilot(lenses, weights, captures, output)
        del captures
        paths = {name: output / f'{name}_rows.jsonl' for name in lenses}
        capture_hashes = {}
        counts = write_scores_many(examples, lenses, weights, numeric_ids, paths,
                                   batch_size=args.batch_size, target_metrics=True, capture_hashes=capture_hashes)
        write_json(output / 'capture_sha256.json', capture_hashes)
        if counts != {'square': 173376, 'rectangular': 86688}:
            raise ValueError('incomplete scoring grid')
        square = index_rows(read_jsonl(paths['square']), examples, WORKSPACE_SOURCE_LAYERS, 'jlens_workspace_mean', hashes=capture_hashes)
        rectangular = index_rows(read_jsonl(paths['rectangular']), examples, SOURCE_LAYERS, 'jlens', hashes=capture_hashes)
        for rows in (square, rectangular):
            validate_token_metrics(rows, vocab_size=vocab_size, numeric_ids=numeric_ids)
        baseline = index_rows(read_jsonl(args.baseline_dir / ROWS_NAME), examples, SOURCE_LAYERS, 'jlens', metrics=False)
        baseline_check = compare_baseline(rectangular, baseline)
        write_json(output / 'baseline_validation.json', baseline_check)
        if not baseline_check['passed']:
            raise AssertionError('rectangular argmax differs from baseline; inspect baseline_validation.json before accepting')
        del baseline, lenses, weights
        # Rehash inputs before exports, including the exact capture bytes scored.
        for path, expected in {**input_hashes, **manifests, **capture_hashes}.items():
            if digest(Path(path)) != expected:
                raise ValueError(f'input changed during computation: {path}')
        source_provenance(args.rectangular_lens)
        verify_artifact(args.square_lens)
        exports = export_results(output, square, rectangular, tokenizer, provenance)
        validation = {'passed': True, 'native': native_check, 'pilot': checks,
                      'baseline_argmax_equal': True, 'inputs_unchanged': True,
                      'captures': len(capture_hashes), 'rows': counts, 'matched_pairs': len(rectangular), **exports}
        write_json(output / 'validation.json', validation)
        # The internal notebook executor can read validated results before the
        # completion marker. Publish COMPLETE only after every export succeeds.
        execute_notebook(output)
        validation['executed_notebook'] = 'one_fact_jlens_comparison.executed.ipynb'
        write_json(output / 'COMPLETE.json', {**validation, 'elapsed_seconds': time.monotonic() - started})
        print(f'Complete: {output / "REPORT.md"}', flush=True)
    except Exception as error:
        write_json(output / 'FAILED.json', {'type': type(error).__name__, 'error': str(error),
                                          'elapsed_seconds': time.monotonic() - started})
        raise

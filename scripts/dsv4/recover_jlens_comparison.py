"""Finish a validated, fully scored comparison without repeating vocabulary projections."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from filler.dsv4.compare_jlenses import export_results, execute_notebook
from filler.dsv4.jlens_comparison import (read_jsonl, index_rows, check_baseline_provenance,
                                        compare_baseline, validate_token_metrics)
from filler.dsv4.jlens_top_object import collect_examples, ROWS_NAME, SOURCE_LAYERS, WORKSPACE_SOURCE_LAYERS
from filler.dsv4.jlens_validation import digest, write_json, source_provenance
from filler.dsv4.workspace_jlens_artifact import verify_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scored-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('recovery requires an approved compute allocation')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / 'STARTED.json', {'source': str(args.scored_dir), 'job_id': os.environ['SLURM_JOB_ID']})
    try:
        source = args.scored_dir.resolve()
        provenance = json.loads((source / 'provenance.json').read_text())
        original = provenance['arguments']
        examples, manifests = collect_examples(Path(original['grid_root']), Path(original['capture_root']), [0, 5, 10, 20])
        if manifests != provenance['manifest_sha256'] or len(examples) != 4128:
            raise ValueError('saved manifest provenance mismatch')
        capture_hashes = json.loads((source / 'capture_sha256.json').read_text())
        if set(capture_hashes) != {e['capture_path'] for e in examples}:
            raise ValueError('capture hash coverage mismatch')
        native = json.loads((source / 'native_validation.json').read_text())
        checks = json.loads((source / 'pilot_checks.json').read_text())
        if not native['passed'] or {(r['lens'], r['layer']) for r in checks if r['passed']} != {
                *(('square', l) for l in WORKSPACE_SOURCE_LAYERS),
                *(('rectangular', l) for l in SOURCE_LAYERS)}:
            raise ValueError('original real-artifact gates incomplete')
        for path, expected in {**provenance['input_and_source_sha256'], **manifests, **capture_hashes}.items():
            if digest(Path(path)) != expected:
                raise ValueError(f'input/source changed after scoring: {path}')
        source_provenance(Path(original['rectangular_lens']))
        verify_artifact(Path(original['square_lens']))
        print('Recovery: original artifact, source and capture integrity passed', flush=True)
        # Link only immutable finished scores; no data is copied or overwritten.
        for name in ('square_rows.jsonl', 'rectangular_rows.jsonl', 'capture_sha256.json',
                     'native_validation.json', 'pilot_checks.json'):
            (output / name).hardlink_to(source / name)
        square = index_rows(read_jsonl(output / 'square_rows.jsonl'), examples,
                            WORKSPACE_SOURCE_LAYERS, 'jlens_workspace_mean', hashes=capture_hashes)
        rectangular = index_rows(read_jsonl(output / 'rectangular_rows.jsonl'), examples,
                                 SOURCE_LAYERS, 'jlens', hashes=capture_hashes)
        config = json.loads((Path(original['checkpoint']) / 'config.json').read_text())
        vocab_size = config['vocab_size']
        numeric_ids = set(provenance['numeric_token_ids'])
        for rows in (square, rectangular):
            validate_token_metrics(rows, vocab_size=vocab_size, numeric_ids=numeric_ids)
        baseline_dir = Path(original['baseline_dir'])
        check_baseline_provenance(json.loads((baseline_dir / 'provenance.json').read_text()), provenance)
        baseline = index_rows(read_jsonl(baseline_dir / ROWS_NAME), examples, SOURCE_LAYERS, 'jlens', metrics=False)
        baseline_check = compare_baseline(rectangular, baseline)
        write_json(output / 'baseline_validation.json', baseline_check)
        if not baseline_check['passed']:
            raise ValueError('rectangular baseline mismatch; see diagnostics')
        del baseline
        print('Recovery: all 86,688 rectangular rows exactly match both baseline argmax scopes', flush=True)
        provenance['recovery'] = {'scored_dir': str(source), 'job_id': os.environ['SLURM_JOB_ID'],
                                  'reason': 'cache expensive tokenizer vocabulary-size lookup during row validation',
                                  'source_sha256': {str(Path(__file__).resolve()): digest(Path(__file__))},
                                  'original_provenance_sha256': digest(source / 'provenance.json')}
        write_json(output / 'provenance.json', provenance)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(original['checkpoint'], local_files_only=True, trust_remote_code=False)
        if len(tokenizer) != vocab_size:
            raise ValueError('tokenizer/config size mismatch')
        exports = export_results(output, square, rectangular, tokenizer, provenance)
        validation = {'passed': True, 'native': native, 'pilot': checks,
                      'baseline_argmax_equal': True, 'inputs_unchanged': True,
                      'captures': len(examples), 'rows': {'square': len(square), 'rectangular': len(rectangular)},
                      'matched_pairs': len(rectangular), **exports}
        write_json(output / 'validation.json', validation)
        execute_notebook(output)
        validation['executed_notebook'] = 'one_fact_jlens_comparison.executed.ipynb'
        write_json(output / 'COMPLETE.json', {**validation, 'elapsed_seconds': time.monotonic() - started})
        print(f'Complete recovered comparison: {output / "REPORT.md"}', flush=True)
    except Exception as error:
        write_json(output / 'FAILED.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()

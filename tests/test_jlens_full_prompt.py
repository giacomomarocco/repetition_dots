"""Small regressions for complete-prefix capture identity, scoring and rendering."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from filler.dsv4.lens_positions import prompt_position_labels, validate_positions
from filler.dsv4.jlens_full_prompt import collect_full_prompt_examples
from filler.dsv4.jlens_top_object import write_scores_many
from filler.dsv4.jlens import JLens
from filler.dsv4.jlens_comparison import index_rows, pair_rows, summarize, plot_comparison, square_summary
from filler.dsv4.patching import digest, file_digest
from test_jlens_top_object import readout


class Tokenizer:
    def decode(self, ids, **kwargs):
        return {99: 'prefix', 0: '</think>', 1: '.', 2: 'Answer', 3: ':', 4: ' '}[ids[0]]

    def encode(self, value, **kwargs):
        return SimpleNamespace(ids=[int(value) + 9])


def metadata_fixture(tmp_path, monkeypatch):
    from filler.dsv4 import jlens_full_prompt as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (checkpoint / name).write_text('{}')
    shard = checkpoint / 'model-00045-of-00048.safetensors'
    shard.touch()
    root = tmp_path / 'run'
    runtime = root / 'runtimes/test'
    runtime.mkdir(parents=True)
    records = []
    for k in (0, 20):
        ids = [99, 99, 0, *([1] * k), 2, 3, 4]
        positions = [{'label': label, 'absolute_position': i+2, 'token_id': ids[i+2],
                      'token': Tokenizer().decode([ids[i+2]])}
                     for i, label in enumerate(prompt_position_labels(k))]
        cell = {'cell_id': 'p:0', 'panel_id': 'p', 'split': 'test', 'row': 0, 'col': 0,
                'left_value': 1, 'right_value': 2, 'target': 3, 'filler_length': k,
                'input_ids': ids, 'positions': positions, 'target_token_ids': {'A': 10, 'X': 11, 'A+X': 12}}
        folder = runtime / f'k{k}'
        folder.mkdir()
        control = {'request_id': f'k{k}', 'runtime_id': 'test', 'config_hash': 'config',
                   'num_tokens': len(ids), 'cell_id': cell['cell_id'], 'layers': [],
                   'positions': [p['absolute_position'] for p in positions], 'capture_all': k==20,
                   'clean_capture': None, 'donor_capture': None, 'recomputation': 'full_downstream',
                   'input_ids_hash': digest(ids)}
        refs, acks = {}, []
        for rank in range(4):
            path = folder / f'rank{rank}.pt'
            path.write_bytes(b'not loaded by metadata preflight')
            refs[str(rank)] = {'path': str(path), 'sha256': file_digest(path)}
            audits = [{'layer': l, 'patched_positions': [], 'restored_count': 0,
                       'replacement_exact': True, 'restoration_exact': True} for l in range(43)]
            acks.append({**control, 'rank': rank, 'patch_positions': control['positions'],
                         'positions': list(range(len(ids))) if k==20 else control['positions'],
                         'audits': audits, 'fused_mhc_post_pre': False, 'capture': refs[str(rank)]})
        (folder / 'control.json').write_text(json.dumps(control))
        (folder / 'acks.json').write_text(json.dumps(acks))
        (folder / 'response.json').write_text(json.dumps({'output_ids': [12 if k==0 else 10]}))
        records.append({'cell': cell, 'capture': {'cell_id': cell['cell_id'], 'ranks': refs},
                        'response': str(folder / 'response.json')})
    plan = {'cells': [r['cell'] for r in records],
            'source_hashes': {str(p.relative_to(tmp_path)): file_digest(p) for p in checkpoint.glob('*.json')},
            'checkpoint_files': {shard.name: {'size': shard.stat().st_size, 'mtime_ns': shard.stat().st_mtime_ns}}}
    plan['config_hash'] = digest(plan)
    (root / 'preflight.json').write_text(json.dumps(plan))
    (runtime / 'captures.json').write_text(json.dumps(records))
    rows = runtime / 'native_rows.jsonl'
    rows.touch()
    complete = {'status': 'complete', 'filler_lengths': [0, 20], 'config_hash': plan['config_hash'],
                'rows_path': str(rows), 'cohorts': {'k0_correct': 1, 'k0_wrong': 0, 'k20_correct': 0, 'k20_wrong': 1}}
    for p in (root, runtime):
        (p / 'COMPLETE.json').write_text(json.dumps(complete))
    for k in (0,20):
        (runtime / f'validation_k{k}.json').write_text('{"passed":true}')
    return root, checkpoint, runtime, records


def test_metadata_preflight_requires_all_three_positions(tmp_path, monkeypatch):
    root, checkpoint, _, _ = metadata_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('metadata preflight loaded tensors'))
    examples, hashes, details = collect_full_prompt_examples(root, checkpoint, tokenizer=Tokenizer(), examples_per_length=1)
    assert len(examples)==28 and len(details['capture_files_sha256'])==8
    assert [e['position_label'] for e in examples[:4]] == prompt_position_labels(0)
    assert all(e['clean_correct'] for e in examples[:4])
    assert not any(e['clean_correct'] for e in examples[4:])
    assert details['complete_answer_prefix'] and hashes


@pytest.mark.parametrize('issue', ['missing', 'token', 'patched', 'input_hash', 'ack', 'response', 'native', 'cohort'])
def test_metadata_rejects_incompatible_captures(tmp_path, monkeypatch, issue):
    root, checkpoint, runtime, records = metadata_fixture(tmp_path, monkeypatch)
    folder = runtime/'k0'
    if issue in ('missing', 'token'):
        cell=deepcopy(records[0]['cell'])
        if issue=='missing': cell['positions'].pop(-2)
        else: cell['positions'][-2]['token_id']=2
        with pytest.raises(ValueError): validate_positions(cell, Tokenizer().decode)
        return
    if issue in ('patched', 'input_hash'):
        path=folder/'control.json'; value=json.loads(path.read_text())
        value['donor_capture' if issue=='patched' else 'input_ids_hash']='wrong'
    elif issue=='ack':
        path=folder/'acks.json'; value=json.loads(path.read_text()); value[0]['capture']['sha256']='wrong'
    elif issue=='response':
        path=runtime/'captures.json'; value=records; value[0]['response']=str(runtime/'k20/response.json')
    elif issue=='native':
        path=runtime/'validation_k0.json'; value={'passed':False}
    else:
        path=folder/'response.json'; value={'output_ids':[10]}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        collect_full_prompt_examples(root,checkpoint,tokenizer=Tokenizer(),examples_per_length=1)


def score_fixture(tmp_path):
    rectangular, weights = readout()
    square = JLens({19:torch.eye(3).half(),39:torch.eye(3).half()},25,3,3,stream_reduction='mean')
    positions = [2,3,4,5]
    state=torch.zeros(4,4,3)
    for i in range(4): state[i,:,i%3]=i+1
    meta={'positions':positions,'rank':0,'cell_id':'p:0','num_tokens':6,
          'runtime_id':'runtime','request_id':'request','config_hash':'config'}
    path=tmp_path/'capture.pt'
    torch.save({'metadata':meta,'states':{19:state,39:state}},path)
    examples=[]
    for i,label in enumerate(prompt_position_labels(0)):
        examples.append({'filler_length':0,'cell_id':'p:0','panel_id':'p','split':'test','row':0,'col':0,
                         'left_value':1,'right_value':2,'target':3,'clean_correct':True,
                         'position_label':label,'absolute_position':positions[i],'pass_id':'runtime/request',
                         'targets':{'A':{'token_id':0},'X':{'token_id':1},'A+X':{'token_id':2}},
                         'capture_format':'full-prompt','runtime_id':'runtime','request_id':'request',
                         'input_ids_sha256':'input-hash','capture_config_hash':'config',
                         'num_tokens':6,'position_token_id':i,'position_token':label,
                         'capture_path':str(path),'_capture_expected_sha256':file_digest(path),
                         '_capture_metadata':meta})
    return examples, {'square':square,'rectangular':rectangular}, weights


def test_full_prompt_scoring_uses_each_token_and_loads_once(tmp_path, monkeypatch):
    examples,lenses,weights=score_fixture(tmp_path)
    paths={n:tmp_path/f'{n}.jsonl' for n in lenses}
    loads=[]; original=torch.load
    def load(*a,**kw): loads.append(1); return original(*a,**kw)
    monkeypatch.setattr(torch,'load',load)
    hashes={}
    counts=write_scores_many(examples,lenses,weights,[0,1,2],paths,batch_size=64,target_metrics=True,capture_hashes=hashes)
    assert counts=={'square':8,'rectangular':8} and len(loads)==1
    indexes={}
    for name,path in paths.items():
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['top_numeric_token_id'] for r in rows if r['layer']==19]==[0,1,2,0]
        assert not any(k.startswith('_') for r in rows for k in r)
        indexes[name]=index_rows(rows,examples,(19,39),'jlens_workspace_mean' if name=='square' else 'jlens',hashes=hashes)
    paired=list(pair_rows(indexes['square'],indexes['rectangular'],layers=(19,39)))
    stats=summarize(paired)
    fig=plot_comparison(stats,0)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()]==['last_question','Answer',':','space (answer_prompt)']
    import matplotlib.pyplot as plt
    plt.close(fig)
    changed=deepcopy(indexes['rectangular']);next(iter(changed.values()))['input_ids_sha256']='wrong'
    with pytest.raises(ValueError,match='identity/provenance'):
        list(pair_rows(indexes['square'],changed,layers=(19,39)))


@pytest.mark.parametrize('issue',['hash','position','runtime','shape'])
def test_full_prompt_scoring_rejects_bad_alignment(tmp_path,issue):
    examples,lenses,weights=score_fixture(tmp_path)
    if issue=='hash': examples[0]['_capture_expected_sha256']='wrong'
    if issue=='position': examples[0]['absolute_position']=99
    if issue=='runtime': examples[0]['_capture_metadata']=dict(examples[0]['_capture_metadata'],runtime_id='wrong')
    if issue=='shape':
        path=Path(examples[0]['capture_path']);item=torch.load(path,weights_only=True)
        item['states'][19]=torch.zeros(3,4,3);torch.save(item,path)
        for e in examples:e['_capture_expected_sha256']=file_digest(path)
    with pytest.raises(ValueError):
        write_scores_many(examples,lenses,weights,[0,1,2],{n:tmp_path/f'{n}.jsonl' for n in lenses},capture_hashes={})


def test_default_cli_routes_to_complete_prefix(tmp_path,monkeypatch):
    from filler.dsv4 import compare_jlenses, jlens_answer_tokens
    calls=[]
    monkeypatch.setattr(jlens_answer_tokens,'run',lambda args:calls.append(args))
    compare_jlenses.main(['--preflight','--output-dir',str(tmp_path/'fresh')])
    assert len(calls)==1 and not calls[0].legacy_grid


def test_full_prefix_compute_guard(tmp_path,monkeypatch):
    from filler.dsv4 import jlens_answer_tokens as runner
    from filler.dsv4.compare_jlenses import parse_args
    monkeypatch.setattr(runner,'preflight',lambda args:([],{},{},{'complete_answer_prefix':True}))
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(runner,'load_checkpoint_readout',lambda *a,**kw:pytest.fail('loaded weights on login'))
    with pytest.raises(RuntimeError,match='approved CPU allocation'):
        runner.run(parse_args(['--output-dir',str(tmp_path/'new')]))


def test_complete_prefix_export_and_executed_notebook(tmp_path, monkeypatch):
    from filler.dsv4 import compare_jlenses as export
    examples,lenses,weights=score_fixture(tmp_path)
    paths={n:tmp_path/f'{n}.jsonl' for n in lenses}
    hashes={}
    write_scores_many(examples,lenses,weights,[0,1,2],paths,target_metrics=True,capture_hashes=hashes)
    indexes={}
    for name,path in paths.items():
        seeds=[json.loads(line) for line in path.read_text().splitlines() if json.loads(line)['layer']==19]
        layers=range(42) if name=='square' else range(19,40)
        rows=[dict(row,layer=layer) for row in seeds for layer in layers]
        indexes[name]=index_rows(rows,examples,tuple(layers),'jlens_workspace_mean' if name=='square' else 'jlens',hashes=hashes)
    output=tmp_path/'exports';output.mkdir()
    # One metric exercises the full artifact path without redundant renders.
    monkeypatch.setattr(export,'METRICS',('top1',))
    decoder=SimpleNamespace(decode=lambda ids,**kw:str(ids[0]))
    stats=export.export_results(output,indexes['square'],indexes['rectangular'],decoder,
        {'started_at':'synthetic','coverage_description':'All three answer-prefix tokens are included.'})
    assert stats['figures']==6 and stats['cohort_counts']=={'0/all':1,'0/correct':1,'0/wrong':0}
    assert (output/'comparison_k0_all_top1.png').is_file()
    assert '168 square rows and 84 rectangular rows' in (output/'REPORT.md').read_text()
    (output/'validation.json').write_text('{"passed":true}')
    export.execute_notebook(output)
    notebook=json.loads((output/'one_fact_jlens_comparison.executed.ipynb').read_text())
    text='\n'.join(o.get('text','') for c in notebook['cells'] if c['cell_type']=='code' for o in c['outputs'])
    assert 'Answer/colon/space coverage: {0: True}' in text
    html='\n'.join(o.get('data',{}).get('text/html','') for c in notebook['cells'] if c['cell_type']=='code' for o in c['outputs'])
    assert 'answer_word' in html and 'answer_colon' in html

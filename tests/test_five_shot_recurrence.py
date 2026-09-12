"""CPU fixtures for the five-shot capture/scoring/export contract; no model loads."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from filler.dsv4.five_shot_recurrence import (
    SOURCE, read, tokenizer, make_cells, vocabulary, normalize, CHECKPOINT)
from filler.dsv4.lens_positions import validate_positions, prompt_position_labels
from filler.dsv4.patching import atomic_json, digest, file_digest
from filler.dsv4.recurrence_runtime import stable_top10, score_logits, first_token_scores, validate_records
from filler.dsv4.recurrence_analysis import (
    grouped_top, summary_arrays, bootstrap_weights, interval, representative, region_indices)
from filler.dsv4.recurrence_export import cell_frequencies


@pytest.fixture(scope='module')
def real_cells():
    return make_cells(read(SOURCE / 'prompts.json'), read(SOURCE / 'results.json'), tokenizer())


def test_actual_524_prompt_coverage(real_cells):
    assert len(real_cells) == 524
    assert sum(len(c['positions']) for c in real_cells)*42 == 330120
    tok = tokenizer()
    for c in real_cells:
        assert c['input_ids'] == tok.encode(c['rendered_prompt'], add_special_tokens=False).ids
        assert len(c['positions']) == c['k']+5
        assert c['rendered_prompt'].endswith('Answer:<｜Assistant｜></think>')
        assert [p['token'] for p in c['positions'][-4:]] == ['Answer', ':', '<｜Assistant｜>', '</think>']
        validate_positions(c, lambda ids: tok.decode(ids, skip_special_tokens=False))
    assert all(real_cells[i]['pair_id'] == real_cells[i+1]['pair_id'] for i in range(0,524,2))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'token', 'suffix', 'absolute'])
def test_bad_position_coverage(real_cells, mutation):
    c = deepcopy(real_cells[0])
    if mutation == 'missing': c['positions'].pop(-3)
    if mutation == 'duplicate': c['positions'][-2] = c['positions'][-1]
    if mutation == 'token': c['positions'][-1]['token_id'] += 1
    if mutation == 'suffix': c['position_suffix'] = 'space'
    if mutation == 'absolute': c['positions'][0]['absolute_position'] -= 1
    tok = tokenizer()
    with pytest.raises(ValueError):
        validate_positions(c, lambda ids: tok.decode(ids, skip_special_tokens=False))


def test_original_space_suffix_still_mandatory():
    texts = ['Q', 'Answer', ':', ' ']
    c = {'input_ids': list(range(4)), 'filler_length': 0,
         'positions': [{'label': label, 'absolute_position': i, 'token_id': i, 'token': texts[i]}
                       for i, label in enumerate(prompt_position_labels(0))]}
    validate_positions(c, lambda ids: texts[ids[0]])
    c['positions'].pop()
    with pytest.raises(ValueError): validate_positions(c, lambda ids: texts[ids[0]])


@pytest.mark.parametrize('seed', range(5))
def test_vocab_top10_ties_and_exact_ranks(seed):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randint(-3,4,(3,4,73),generator=gen).float()
    expected = logits.argsort(dim=-1, descending=True, stable=True)[..., :10]
    assert torch.equal(stable_top10(logits), expected)
    targets = [0,20,72]
    result = score_logits(logits, targets)
    for j, token in enumerate(targets):
        full = logits.argsort(dim=-1, descending=True, stable=True)
        rank = (full == token).int().argmax(-1)+1
        assert torch.equal(result['target_rank'][..., j], rank)
    torch.testing.assert_close(result['top_logprob'], logits.log_softmax(-1).gather(-1, expected))
    assert torch.equal(stable_top10(torch.zeros(2,31)), torch.arange(10).expand(2,-1))
    with pytest.raises(ValueError): stable_top10(torch.full((11,),float('nan')))


def test_eight_token_response_uses_first_scores():
    response = {'output_ids': [11,1], 'meta_info': {'output_token_ids_logprobs': [[[-.1,11,None]], [[-.9,11,None]]]}}
    assert first_token_scores(response, [11,11]) == {11:-.1}
    assert len(response['output_ids']) == 2
    bad=deepcopy(response); bad['meta_info']['output_token_ids_logprobs'].pop()
    with pytest.raises(ValueError): first_token_scores(bad,[11])


def test_grouping_denominators_breadth_and_persistence():
    assert normalize(' WORD\n') == normalize('word')
    # 2 prompts, 2 layers, 1 position; IDs 0 and 1 share one wordpiece.
    top = np.array([[[[0,1]], [[2,3]]], [[[0,2]], [[0,1]]]])
    grouped = grouped_top(top,[0,0,1,2])
    assert grouped[0,0,0].tolist() == [0,-1]
    items, metrics, cells = summary_arrays(grouped,[0])
    assert items.tolist() == [0,1,2] and cells == 2
    np.testing.assert_allclose(metrics['persistence'].toarray(), [[.5,.5,.5],[1,.5,0]])
    np.testing.assert_allclose(metrics['breadth'].toarray(), [[1,1,1],[1,1,0]])
    np.testing.assert_allclose(metrics['top1'].toarray(), [[.5,.5,0],[1,0,0]])
    frequency = cell_frequencies(grouped, np.array([True,True]),3)
    assert frequency['n'] == 2
    lookup=dict(zip(frequency['code'],frequency['top10_count']))
    assert lookup[0] == 2 and lookup[3] == 1
    assert cell_frequencies(grouped,np.zeros(2,bool),3)['n'] == 0


def test_paired_resampling_and_subgroups():
    weights = bootstrap_weights(5)
    assert weights.shape == (2000,5)
    np.testing.assert_array_equal(weights.sum(-1),np.full(2000,5))
    np.testing.assert_array_equal(weights,bootstrap_weights(5))
    a=np.arange(5)/10
    b=a+.2
    np.testing.assert_allclose(interval(b-a,weights),[[.2],[.2]])
    np.testing.assert_allclose(interval(b-a,weights,[True,False,False,False,False]),[[.2],[.2]])
    assert np.isnan(interval(a,weights,np.zeros(5,bool))).all()
    draws=np.random.default_rng(42).integers(0,5,(2000,5))
    np.testing.assert_allclose(interval(a,weights)[:,0],np.quantile(a[draws].mean(-1),[.025,.975]))


def test_representative_matches_brute_force_and_ties():
    rng=np.random.default_rng(1)
    top=np.array([[[rng.choice(30,10,replace=False) for _ in range(2)] for _ in range(3)] for _ in range(5)])
    correct=[False,True,True,False,True]
    pairs=['e','d','c','b','a']
    best,score,_=representative(top,correct,pairs)
    expected=np.array([np.mean([len(set(top[i,l,p]) & set(top[j,l,p]))/10
                               for j in range(5) for l in range(3) for p in range(2)]) for i in range(5)])
    np.testing.assert_allclose(score,expected)
    assert best == min([1,2,4],key=lambda i:(-expected[i],pairs[i]))
    same=np.tile(np.arange(10),(3,2,2,1))
    assert representative(same,[False,True,True],['a','c','b'])[0] == 2
    with pytest.raises(ValueError): representative(same,[False]*3,['a','c','b'])


def artifact_fixture(tmp_path, real_cells, *, pairs=2):
    """Synthetic numeric artifacts with real prompt/capture identity and 42 layers."""
    cells=deepcopy(real_cells[:pairs*2])
    plan={'cells':cells, 'readouts':sum(len(c['positions']) for c in cells)*42,
          'source_hashes':{}, 'checkpoint_files':{}, 'bootstrap':{'resamples':2000,'seed':42,'unit':'paired fact'}}
    plan['config_hash']=digest(plan)
    records=[]
    vocab={'tokens':[f'word{i:02d}' for i in range(40)], 'groups':[f'word{i:02d}' for i in range(40)],
           'token_group':list(range(40)), 'group_kind':['textual']*40}
    # Retain real numeric IDs in a synthetic vocabulary so exact target validation runs.
    size=max(max(c['input_ids']) for c in cells)+1
    for key in ('tokens','groups'): vocab[key] += [f'number{i}' for i in range(40,size)]
    vocab['token_group']=list(range(size)); vocab['group_kind']+=['number']*(size-40)
    plan['vocabulary_sha256']=digest(vocab)
    plan['config_hash']=digest({k:v for k,v in plan.items() if k!='config_hash'})
    atomic_json(tmp_path/'vocabulary.json',vocab)
    atomic_json(tmp_path/'preflight.json',plan)
    atomic_json(tmp_path/'runtime.json',{'config_hash':plan['config_hash'],'synthetic':True})
    reports={rank:[] for rank in range(4)}
    massreports={rank:[] for rank in range(4)}
    rng=np.random.default_rng(15)
    for i,c in enumerate(cells):
        folder=tmp_path/'passes'/str(i); folder.mkdir(parents=True)
        positions=[p['absolute_position'] for p in c['positions']]
        control={'request_id':str(i),'runtime_id':'fixture','config_hash':plan['config_hash'],
                 'cell_id':c['cell_id'],'input_ids_hash':digest(c['input_ids']),'num_tokens':len(c['input_ids']),
                 'positions':positions,'layers':[],'capture_all':False,'clean_capture':None,'donor_capture':None,
                 'recomputation':'full_downstream','output_root':str(folder),'validate_returned_hidden':i<2}
        refs,acks={},[]
        for rank in range(4):
            path=folder/f'rank{rank}.pt'; path.write_bytes(b'synthetic capture; GPU tests use real tensors')
            refs[str(rank)]={'path':str(path),'sha256':file_digest(path)}
            audits=[{'layer':l,'patched_positions':[],'restored_count':0,'replacement_exact':True,'restoration_exact':True} for l in range(43)]
            acks.append({**control,'rank':rank,'patch_positions':positions,'positions':positions,
                         'fused_mhc_post_pre':False,'audits':audits,'capture':refs[str(rank)]})
        correct=(i%4 != 0)
        response={'text':str(c['target']) if correct else 'not an integer', 'output_ids':[c['target_token_ids']['A+X'],1],
                  'meta_info':{'cached_tokens':0}}
        for name,value in [('control.json',control),('acks.json',acks),('response.json',response)]: atomic_json(folder/name,value)
        record={'cell':c,'response':str(folder/'response.json'),'response_sha256':file_digest(folder/'response.json'),
                'control_sha256':file_digest(folder/'control.json'),'acks_sha256':file_digest(folder/'acks.json'),
                'correct':correct,'response_text':response['text'],'capture':{'cell_id':c['cell_id'],'ranks':refs}}
        records.append(record)
        shape=(42,len(positions))
        ids=np.array([rng.choice(40,10,replace=False) for _ in range(42*len(positions))]).reshape(*shape,10)
        ids.sort(-1)
        lp=np.full(ids.shape,-5,np.float32) # all tie; IDs already ascending
        values={'top_ids':ids.astype(np.int32),'top_logprob':lp,'target_logprob':np.full((*shape,3),-10,np.float32),
                'target_logits':np.full((*shape,3),-10,np.float32),'target_rank':np.full((*shape,3),100,np.int32)}
        for mode,arrays,reportdict in [('scores',values,reports),('mass',{'group_logmass':np.full((*shape,20),-4,np.float32)},massreports)]:
            path=tmp_path/mode/f"{c['cell_id']}.npz"; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**arrays,config_hash=plan['config_hash'],cell_id=c['cell_id'],capture_sha256=refs['0']['sha256'],
                                vocabulary_sha256=plan['vocabulary_sha256'],selected_groups_sha256='')
            reportdict[i%4].append({'cell_id':c['cell_id'],'path':str(path),'sha256':file_digest(path),
                      'native':{'passed':True,'max_logprob_error':0,'hidden_max_abs_error':0 if i<2 else None},
                      'reference':[{'layer':l,'passed':True} for l in range(42)]})
    atomic_json(tmp_path/'captures.json',records)
    atomic_json(tmp_path/'CAPTURED.json',{'config_hash':plan['config_hash'],'count':len(records),
                                        'captures_sha256':file_digest(tmp_path/'captures.json')})
    for rank in range(4):
        atomic_json(tmp_path/f'score_rank{rank}.json',reports[rank])
        atomic_json(tmp_path/f'mass_rank{rank}.json',massreports[rank])
        atomic_json(tmp_path/f'worker_provenance{rank}.json',{'rank':rank,'readout_tensor_sha256':{'fixture':'synthetic'}})
    return plan,records


def finish_mass_metadata(root):
    for rank in range(4):
        reports=read(root/f'mass_rank{rank}.json')
        for report in reports:
            path=Path(report['path'])
            with np.load(path,allow_pickle=False) as saved:
                arrays={k:saved[k] for k in saved.files}
            arrays['selected_groups_sha256']=digest(read(root/'selected_groups.json'))
            np.savez_compressed(path,**arrays)
            report['sha256']=file_digest(path)
        atomic_json(root/f'mass_rank{rank}.json',reports)


@pytest.mark.parametrize('issue',['missing','duplicate','mismatch','response','ack','control','reused_path'])
def test_capture_integrity_rejections(tmp_path,real_cells,issue):
    plan,records=artifact_fixture(tmp_path,real_cells)
    validate_records(plan,records)
    if issue=='missing': records.pop()
    elif issue=='duplicate': records.append(records[0])
    elif issue=='mismatch': records[0]['cell']['input_ids'][0]+=1
    elif issue=='response': records[0]['response']=records[1]['response']
    elif issue=='reused_path': records[0]['capture']['ranks']['0']=records[1]['capture']['ranks']['0']
    else:
        path=Path(records[0]['response']).parent/f'{issue}.json'
        if issue=='ack': path=path.with_name('acks.json')
        path.write_text('{}')
    with pytest.raises((ValueError,KeyError)):
        validate_records(plan,records)


def test_score_coverage_and_mass_selection(tmp_path,real_cells):
    from filler.dsv4.recurrence_analysis import load_scores,select_groups
    artifact_fixture(tmp_path,real_cells)
    plan,data=load_scores(tmp_path)
    assert data[20]['top_ids'].shape==(2,42,25,10)
    selected=select_groups(tmp_path)
    finish_mass_metadata(tmp_path)
    assert len(selected)==20
    _,mass=load_scores(tmp_path,mass=True)
    assert np.all(mass[20]['group_logmass']==-4)
    # Positive exact mass at cells where a selected group is outside the top ten.
    outside=~(data[20]['top_ids']==selected[0]['group_id']).any(-1)
    assert outside.any() and np.all(np.exp(mass[20]['group_logmass'][...,0][outside])>0)
    report=read(tmp_path/'score_rank0.json'); report.append(report[0]); atomic_json(tmp_path/'score_rank0.json',report)
    with pytest.raises(ValueError,match='duplicate'): load_scores(tmp_path)


def test_launcher_instrumentation_and_port(monkeypatch,tmp_path):
    from scripts.dsv4.one_fact_patching import launch_command,server_environment
    from filler.dsv4.campaign_hook import campaign_hook_spec
    monkeypatch.setenv('SGLANG_PORT','34567')
    env=server_environment()
    assert 'SGLANG_PORT' not in env
    cmd=launch_command(tmp_path,34567)
    assert cmd[cmd.index('--port')+1]=='34567'
    assert json.loads(cmd[cmd.index('--forward-hooks')+1])==campaign_hook_spec(tmp_path)
    for flag in ('--disable-radix-cache','--disable-cuda-graph','--enable-return-hidden-states','--disable-overlap-schedule'):
        assert flag in cmd
    assert cmd[cmd.index('--chunked-prefill-size')+1]=='-1'


def test_full_export_and_real_notebook_cells(tmp_path,real_cells,monkeypatch):
    from filler.dsv4 import recurrence_export as module
    from filler.dsv4.recurrence_analysis import select_groups
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    artifact_fixture(tmp_path,real_cells)
    select_groups(tmp_path)
    finish_mass_metadata(tmp_path)
    # Only production tensor/cluster provenance is mocked. All score identity,
    # summaries, filters, bootstrap, selected groups, completion and real notebook
    # cells execute against the small fixture. Avoid 85 large login-node figures.
    monkeypatch.setattr(module,'verify_plan',lambda plan:None)
    monkeypatch.setattr(module,'LAB_LOG',tmp_path/'LAB_LOG.md')
    def little(*a,**kw):
        fig,ax=plt.subplots(figsize=(1,1)); ax.imshow(np.eye(2)); return fig
    monkeypatch.setattr(module,'paired_heatmap',little)
    def save_small(fig,stem):
        fig.set_size_inches(1,1)
        fig.savefig(str(stem)+'.png',dpi=24)
        fig.savefig(str(stem)+'.pdf')
        plt.close(fig)
    monkeypatch.setattr(module,'save_figure',save_small)
    module.export(tmp_path)
    complete=read(tmp_path/'COMPLETE.json')
    assert complete['passed'] and complete['readouts']==2520
    notebook=read(tmp_path/'five_shot_square_recurrence.executed.ipynb')
    assert all(c['execution_count'] for c in notebook['cells'] if c['cell_type']=='code')
    assert all(o['output_type']!='error' for c in notebook['cells'] if c['cell_type']=='code' for o in c['outputs'])
    assert module.load_completed(tmp_path)==tmp_path
    # Publication cannot survive changed exported content.
    (tmp_path/'tables/cohorts.csv').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'): module.load_completed(tmp_path)


@pytest.mark.parametrize('issue',['none','layer','duplicate_position','request','shape','dtype','nonfinite'])
def test_tensor_capture_validation(tmp_path,real_cells,issue):
    from filler.dsv4.recurrence_runtime import load_capture
    plan,records=artifact_fixture(tmp_path,real_cells)
    record=records[0]
    control=read(Path(record['response']).parent/'control.json')
    state=torch.zeros(len(control['positions']),4,4096,dtype=torch.bfloat16)
    saved={'states':{l:state for l in range(43)},'metadata':{
        **{k:control[k] for k in ('cell_id','request_id','runtime_id','config_hash','num_tokens')},
        'positions':control['positions'][:],'rank':0,'num_layers':43,'fused_mhc_post_pre':False}}
    if issue=='layer': saved['states'].pop(0)
    if issue=='duplicate_position': saved['metadata']['positions'][0]=saved['metadata']['positions'][1]
    if issue=='request': saved['metadata']['request_id']='different'
    if issue=='shape': saved['states'][0]=state[:1]
    if issue=='dtype': saved['states'][0]=state.float()
    if issue=='nonfinite': state[0,0,0]=float('nan')
    path=Path(record['capture']['ranks']['0']['path'])
    torch.save(saved,path)
    record['capture']['ranks']['0']['sha256']=file_digest(path)
    if issue=='none': assert len(load_capture(record)['states'])==43
    else:
        with pytest.raises(ValueError): load_capture(record)


@pytest.mark.parametrize('fail_capture',[False,True])
def test_automated_lifecycle_stops_server_before_scoring(tmp_path,real_cells,monkeypatch,fail_capture):
    from unittest.mock import MagicMock
    from filler.dsv4 import recurrence_runtime as runtime
    from filler.dsv4 import recurrence_analysis as analysis
    from filler.dsv4 import recurrence_export as exporter
    from filler.dsv4 import workspace_jlens_artifact as artifact
    from filler.dsv4 import jlens
    from scripts.dsv4 import one_fact_patching as launcher
    import time
    plan,records=artifact_fixture(tmp_path,real_cells)
    events=[]
    monkeypatch.setenv('SLURM_JOB_ID','fixture')
    monkeypatch.setattr(runtime,'verify_plan',lambda _:None)
    monkeypatch.setattr(runtime,'allocation',lambda:({},[],time.time()+3600))
    monkeypatch.setattr(artifact,'verify_artifact',lambda _:None)
    monkeypatch.setattr(jlens,'load_workspace_jlens',lambda _:None)
    monkeypatch.setattr(runtime.socket,'socket',MagicMock())
    monkeypatch.setattr(runtime.signal,'signal',lambda *a:None)
    monkeypatch.setattr(runtime,'tokenizer',lambda:None)
    monkeypatch.setattr(runtime,'vocabulary',lambda _: {})
    process=SimpleNamespace(poll=lambda:None)
    def popen(*a,**kw): events.append('start'); return process
    monkeypatch.setattr(runtime.subprocess,'Popen',popen)
    monkeypatch.setattr(runtime,'stop_process',lambda _:events.append('stop'))
    def info(_):
        from filler.dsv4.campaign_hook import campaign_hook_spec
        control=next((tmp_path/'runtimes').iterdir())/'control'
        return {'disable_radix_cache':True,'chunked_prefill_size':-1,'disable_cuda_graph':True,
                'disable_piecewise_cuda_graph':True,'disable_overlap_schedule':True,'max_running_requests':1,
                'enable_return_hidden_states':True,'forward_hooks':campaign_hook_spec(control),'model_path':str(CHECKPOINT)}
    monkeypatch.setattr(launcher,'get_json',info)
    pending=iter(records)
    def request(*args):
        events.append('capture')
        if fail_capture: raise RuntimeError('synthetic capture failure')
        return next(pending)
    monkeypatch.setattr(runtime,'request',request)
    monkeypatch.setattr(runtime,'workers',lambda _,mode,deadline:events.append(mode))
    monkeypatch.setattr(analysis,'select_groups',lambda _:events.append('select'))
    def export(path): events.append('export'); atomic_json(path/'COMPLETE.json',{'status':'complete','passed':True})
    monkeypatch.setattr(exporter,'export',export)
    if fail_capture:
        with pytest.raises(RuntimeError,match='synthetic capture failure'):
            runtime.run(tmp_path,30002)
        assert events==['start','capture','stop']
        assert not (tmp_path/'COMPLETE.json').exists()
        folder=next((tmp_path/'runtimes').iterdir())
        assert (folder/'FAILED.json').exists() and (folder/'ended.json').exists()
        return
    runtime.run(tmp_path,30002)
    assert events==['start',*(['capture']*4),'stop','score','select','mass','export']
    assert read(tmp_path/'COMPLETE.json')['status']=='complete'


def test_exact_group_mass_includes_all_variants_outside_top_ten():
    from filler.dsv4.recurrence_runtime import score_group_mass
    logits=torch.arange(30).float()[None]
    groups=[torch.tensor([0,1]),torch.tensor([28,29])]
    mass=score_group_mass(logits,groups).exp()
    torch.testing.assert_close(mass,torch.stack([logits.softmax(-1)[:,ids].sum(-1) for ids in groups],-1))
    assert mass[0,0]>0 and not torch.isin(groups[0],stable_top10(logits)).any()
    assert mass[0,1] > logits.softmax(-1)[0,29]
    with pytest.raises(ValueError): score_group_mass(logits,[])

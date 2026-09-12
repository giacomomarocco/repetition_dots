from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest
import torch

from filler.dsv4.five_shot_repeat import FiveShotRepeat, CONDITIONS, mapping, score, control_gate
from filler.dsv4.patching import atomic_json, digest, file_digest, Journal
from filler.dsv4.patching_logits import first_prediction, make_logits_hook, collect_logits
from filler.dsv4.five_shot_recurrence import make_cells, read, SOURCE, tokenizer


def response(text='12', ids=None):
    ids = [12, 1] if ids is None else ids
    return {'text': text, 'output_ids': ids, 'meta_info': {'cached_tokens':0,
        'output_token_ids_logprobs': [[[-.1,12,None]]] * len(ids)}}


def cell(k=20):
    return {'k':k, 'cell_id':f'c{k}', 'prompt_id':f'c{k}', 'pair_id':'p', 'target':12,
            'target_token_ids': {'A':12,'X':12,'A+X':12}, 'input_ids':list(range(30)),
            'positions':[{'label':f'filler_{i}','absolute_position':i+5} for i in range(20)]}


@pytest.mark.parametrize('text,correct', [('12',True), (' 12\n',True), ('12 things',False),
                                          ('12\n13',False), ('',False), ('13',False)])
def test_complete_answer_not_first_token(text, correct):
    result = score(cell(), response(text))
    assert result['correct'] is correct
    assert result['first_token_correct'] is True


def test_first_prediction_and_full_identity():
    raw = response(ids=list(range(8)))
    before = deepcopy(raw)
    assert first_prediction(raw,8)['output_ids'] == [0]
    assert raw == before
    with pytest.raises(ValueError): first_prediction(raw)
    raw['meta_info']['output_token_ids_logprobs'].pop()
    with pytest.raises(ValueError): first_prediction(raw,8)
    a, b = response(), response('12\nextra')
    assert control_gate(a,b,[12],full_response=False)['passed']
    assert not control_gate(a,b,[12],full_response=True)['passed']


def test_real_saved_pairs_and_final_question_mapping():
    prompts, history = read(SOURCE/'prompts.json'), read(SOURCE/'results.json')
    cells = make_cells(prompts, history, tokenizer())
    assert len(cells) == 524
    for c in cells[1::2]:
        for s in range(6):
            dest, src = mapping(c,s)
            pos = {p['label']:p['absolute_position'] for p in c['positions']}
            assert len(dest) == 19-s
            assert src == [pos[f'filler_{s}']]*(19-s)
            assert min(dest) > pos['last_question']
            assert max(dest) < pos['answer_word']
    with pytest.raises(ValueError): make_cells(prompts + [prompts[0]],history,tokenizer())
    changed=deepcopy(prompts)
    selected=next(c for c in changed if c['k']==20)
    selected['rendered_prompt'] += 'x'
    with pytest.raises(ValueError): make_cells(changed,history,tokenizer())


def test_logits_decode_once_and_duplicate_prefill_rejected(tmp_path):
    root = tmp_path/'control'
    control = dict(request_id='r', runtime_id='rt', config_hash='h', cell_id='c',
        num_tokens=3, input_ids_hash=digest([1,2,3]), scored_token_ids=[2], raw_logits=True,
        max_new_tokens=8, output_root=str(tmp_path/'pass'))
    atomic_json(root/'NEXT.json',control)
    atomic_json(root/'acks/r.rank0.json',control)
    hook=make_logits_hook({'control_root':str(root)})
    module=SimpleNamespace(vocab_size=4)
    output=SimpleNamespace(next_token_logits=torch.tensor([[0.,1.,3.,2.]]))
    prefill=(torch.tensor([1,2,3]),None,None,SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:False)))
    decode=(torch.tensor([2]),None,None,SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:True)))
    assert hook(module,prefill,output) is output
    path=tmp_path/'pass/logits.rank0.json'
    saved=path.read_bytes()
    for _ in range(7): assert hook(module,decode,output) is output
    assert path.read_bytes()==saved
    with pytest.raises(RuntimeError): hook(module,decode,output)
    with pytest.raises(RuntimeError): hook(module,prefill,output)
    logits=output.next_token_logits[0]
    lp=logits.log_softmax(0)
    raw={'output_ids':[2,1], 'meta_info': {'output_token_ids_logprobs':[[[float(lp[2]),2,None]]]*2}}
    for rank in range(1,4):
        data=json.loads(saved); data['rank']=rank
        atomic_json(tmp_path/f'pass/logits.rank{rank}.json',data)
    assert collect_logits(control,raw)['raw_logits']['argmax']==2


class FakeCampaign(FiveShotRepeat):
    def __init__(self, *args, fail_at=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls=[]
        self.fail_at=fail_at
    def execute(self, **kwargs):
        self.calls.append(kwargs)
        n=len(self.calls)
        if n==self.fail_at: raise InterruptedError('simulated interruption')
        path=self.runtime_root/f'{n}/response.json'
        atomic_json(path,response())
        return {'runtime_id':self.runtime_id,'request_id':str(n),
                'capture':{'cell_id':kwargs['cell']['cell_id'],'runtime_id':self.runtime_id},
                'response':{'path':str(path),'sha256':file_digest(path)}}


def test_interrupted_block_restarts_fresh_and_references_deduplicate(tmp_path):
    pairs=[{'record_id':f'example|{i}','cells':[dict(cell(0),pair_id=str(i)),dict(cell(),pair_id=str(i))]} for i in range(2)]
    manifest={'config_hash':'h','pairs':pairs}
    def create(runtime, fail_at=None):
        return FakeCampaign(manifest,tmp_path,runtime,None,tmp_path,deadline=time.time()+1000,
                            validation=lambda *a,**kw:{'passed':True},fail_at=fail_at)
    first=create('old',fail_at=23)
    with pytest.raises(InterruptedError): first.run()
    assert len(first.journal.records)==1
    assert (tmp_path/'runtimes/old/21/response.json').is_file()
    second=create('new')
    assert second.run()['examples']==2
    assert len(second.calls)==20  # all 14 conditions plus fresh runtime diagnostics
    assert not second.calls[0].get('donor') and not second.calls[1].get('donor')
    assert [r['runtime_id'] for r in second.journal.records.values()]==['old','new']
    assert all(set(r['results'])==set(CONDITIONS) for r in second.journal.records.values())
    assert len([r['results']['baseline'] for r in second.journal.records.values()])==2
    third=create('unused'); third.run(); assert not third.calls
    with pytest.raises(ValueError): second.journal.append(next(iter(second.journal.records.values())))


def test_http_transport_eight_token_payload(tmp_path, monkeypatch):
    from filler.dsv4.patching_campaign import HTTPTransport
    import filler.dsv4.campaign_hook as hooks
    import scripts.dsv4.run_activation_patching as http
    payloads=[]
    monkeypatch.setattr(http,'flush_cache',lambda url:None)
    def post(url, payload, timeout):
        payloads.append(payload)
        return response()
    monkeypatch.setattr(http,'post_json',post)
    monkeypatch.setattr(hooks,'validate_acknowledgements',lambda *a:[])
    control={'output_root':str(tmp_path/'pass'),'max_new_tokens':8,'capture_all':False}
    result,_=HTTPTransport('http://local',tmp_path/'control').run(control,[1,2,3],[12])
    assert payloads[0]['sampling_params']=={'temperature':0,'max_new_tokens':8}
    assert len(result['output_ids'])==2

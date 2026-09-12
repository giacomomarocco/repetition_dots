from copy import deepcopy
import json
import time
import pytest

from filler.dsv4.five_shot_repeat_plot import load_partial_report
from filler.dsv4.patching import digest
from test_five_shot_repeat import FakeCampaign, cell


def prepared(tmp_path):
    pairs=[{'record_id':f'example|{i}','cells':[
        dict(cell(0),pair_id=str(i),historical_correct=False),
        dict(cell(),pair_id=str(i),historical_correct=True)]} for i in range(2)]
    manifest={'pairs':pairs,'historical':{'0':{'correct':0,'count':2},'20':{'correct':2,'count':2}}}
    manifest['config_hash']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    campaign=FakeCampaign(manifest,tmp_path,'runtime',None,tmp_path,deadline=time.time()+1000,
                          validation=lambda *a,**kw:{'passed':True},fail_at=23)
    with pytest.raises(InterruptedError):campaign.run()
    return campaign


def test_preview_atomic_common_subset_and_live_tail_never_repaired(tmp_path):
    prepared(tmp_path)
    path=tmp_path/'results.jsonl'
    complete=path.read_bytes()
    path.write_bytes(complete+b'{"record":')
    progress=(tmp_path/'progress.json').read_bytes()
    report=load_partial_report(tmp_path)
    assert report['preliminary'] and report['completed_examples']==1
    assert all(p['count']==1 for p in report['points'])
    assert report['historical']=={'0':{'correct':0,'count':1},'20':{'correct':1,'count':1}}
    assert path.read_bytes()==complete+b'{"record":'
    assert (tmp_path/'progress.json').read_bytes()==progress
    path.write_bytes(complete+complete)
    with pytest.raises(ValueError,match='duplicated'):load_partial_report(tmp_path)


@pytest.mark.parametrize('mutation',['score','missing','runtime','gate','checksum'])
def test_bad_checkpoint_rejected(tmp_path,mutation):
    prepared(tmp_path)
    path=tmp_path/'results.jsonl'
    envelope=json.loads(path.read_text().splitlines()[0]); record=envelope['record']
    if mutation=='score':record['scores']['baseline']['correct']=False
    if mutation=='missing':record['results'].pop('identity_0')
    if mutation=='runtime':record['results']['repeat_0']['runtime_id']='other'
    if mutation=='gate':record['runtime_gate']['native']['passed']=False
    envelope['sha256']='bad' if mutation=='checksum' else digest(record)
    path.write_text(json.dumps(envelope)+'\n')
    with pytest.raises(ValueError):load_partial_report(tmp_path)

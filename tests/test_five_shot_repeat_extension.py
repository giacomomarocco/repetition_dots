import json
import time
import pytest
from filler.dsv4.five_shot_repeat_extension import Extension, positions, DEFAULT_ROOT
from filler.dsv4.patching import atomic_json, file_digest

def response():
    return {'text':'12','output_ids':[12,1],'meta_info':{'cached_tokens':0,'output_token_ids_logprobs':[[[-.1,12,None]]]*2}}

def test_all_saved_source6_positions():
    m=json.loads((DEFAULT_ROOT/'manifest.json').read_text())
    assert len(m['pairs'])==262
    for pair in m['pairs']:
        c=pair['cells'][1];p={v['label']:v['absolute_position'] for v in c['positions']}
        dest,src=positions(c)
        assert dest==[p[f'filler_{i}'] for i in range(7,20)]
        assert src==[p['filler_6']]*13
        assert min(dest)>p['last_question'] and max(dest)<p['answer_word']
        assert len(set(p[f'filler_{i}'] for i in range(20))-set(dest))==7

class Fake(Extension):
    def execute(self,**kw):
        self.calls.append(kw)
        if len(self.calls)==self.fail_at:raise InterruptedError()
        path=self.runtime_root/self.attempt/str(len(self.calls))/'response.json'
        atomic_json(path,response())
        return {'request_id':str(len(self.calls)),'runtime_id':self.runtime_id,'response':{'path':str(path),'sha256':file_digest(path)}}

def test_resume_reuses_same_baseline_and_redoes_incomplete_block(tmp_path):
    c={'pair_id':'p','cell_id':'c','target':12,'target_token_ids':{'A+X':12},'positions':[{'label':f'filler_{i}','absolute_position':i+5} for i in range(20)]}
    base={'config_hash':'base','pairs':[{'record_id':str(i),'cells':[None,dict(c,pair_id=str(i))]} for i in range(2)]}
    path=tmp_path/'baseline.json';atomic_json(path,response())
    baseline={'capture':{'runtime_id':'warm'},'response':{'path':str(path),'sha256':file_digest(path)}}
    records={str(i):{'results':{'no_intervention':baseline}} for i in range(2)}
    m={'config_hash':'extension','runtime_id':'warm','runtime_path':str(tmp_path),'deadline':time.time()+5000}
    def make(attempt,fail=None):
        f=Fake(m,base,records,tmp_path/'ext',30002);f.calls=[];f.fail_at=fail;f.attempt=attempt;return f
    a=make('a',5)
    with pytest.raises(InterruptedError):a.run()
    assert len(a.journal.records)==1
    b=make('b');b.run()
    assert len(b.calls)==3 and len(b.journal.records)==2
    assert all(k['donor']==baseline['capture'] for k in a.calls+b.calls)
    assert list(a.calls[0]['layers'])==list(range(43))
    assert a.calls[0]['source_positions']==a.calls[0]['positions']
    assert list(a.calls[1]['layers'])==[42]
    assert a.calls[2]['source_positions']==[11]*13
    assert b.manifest['config_hash']=='base'
    d=make('c');d.run();assert not d.calls

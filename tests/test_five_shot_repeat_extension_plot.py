import json
import pytest
from filler.dsv4.five_shot_repeat_plot import include_extension
from filler.dsv4.patching import digest, file_digest, atomic_json
from filler.addition.accuracy_plot import wilson_interval


def fixture(root):
    pairs=[{'record_id':str(i),'cells':[None,{'pair_id':str(i),'target':12}]} for i in range(262)]
    atomic_json(root/'manifest.json',{'pairs':pairs})
    (root/'results.jsonl').write_text('original references')
    child=root/'source-6-extension'
    m={'parent_config_hash':'parent','runtime_id':'warm','parent_manifest_sha256':file_digest(root/'manifest.json'),'parent_journal_sha256':file_digest(root/'results.jsonl')}
    m['config_hash']=digest(m);atomic_json(child/'manifest.json',m)
    blocks=[];rows=[]
    for i in range(262):
        score={'text':'12' if i<100 else '13','parsed_answer':12 if i<100 else 13,'correct':i<100}
        b={'record_id':str(i),'pair_id':str(i),'config_hash':m['config_hash'],'runtime_id':'warm','identity_gate':{'passed':True},'score':score}
        blocks.append({'record':b,'sha256':digest(b)})
        rows.append({'pair_id':str(i),**score})
    (child/'results.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in blocks))
    atomic_json(child/'analysis/raw_scores.json',rows)
    lo,hi=wilson_interval(100,262)
    point={'condition':'repeat_6','untouched':7,'correct':100,'count':262,'accuracy':100/262,'ci_low':lo,'ci_high':hi}
    done={'status':'complete','config_hash':m['config_hash'],'parent_config_hash':'parent','integrity':{'passed':True}}
    atomic_json(child/'COMPLETE.json',done);atomic_json(child/'analysis/summary.json',{**done,'point':point})
    report={'config_hash':'parent','points':[{'count':262} for _ in range(8)]}
    return report,child


def test_adds_one_point_without_recounting_references(tmp_path):
    report,child=fixture(tmp_path)
    combined=include_extension(tmp_path,report)
    assert combined['points'][:8]==report['points'] and len(combined['points'])==9
    assert combined['points'][-1]['correct']==100 and combined['points'][-1]['untouched']==7
    assert len(report['points'])==8


@pytest.mark.parametrize('mutation',['duplicate','raw_score','count','parent'])
def test_rejects_extension_integrity_failures(tmp_path,mutation):
    report,child=fixture(tmp_path)
    if mutation=='duplicate':
        p=child/'results.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
    if mutation=='raw_score':
        p=child/'analysis/raw_scores.json';rows=json.loads(p.read_text());rows[0]['correct']=False;atomic_json(p,rows)
    if mutation=='count':report['points'][0]['count']=261
    if mutation=='parent':(tmp_path/'results.jsonl').write_text('altered parent')
    with pytest.raises(ValueError):include_extension(tmp_path,report)


def test_unfinished_extension_not_displayed(tmp_path):
    report={'points':[]}
    assert include_extension(tmp_path,report) is report

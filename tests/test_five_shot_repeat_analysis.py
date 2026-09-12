"""Response-count reconciliation independent of large residual serialization."""
import time
import pytest

from filler.dsv4 import five_shot_repeat as module
from filler.dsv4.patching_campaign import read_response
from test_five_shot_repeat import FakeCampaign, cell


def test_complete_blocks_only_reference_counts_and_raw_response_reconciliation(tmp_path, monkeypatch):
    pairs=[{'record_id':f'example|{i}','cells':[dict(cell(0),pair_id=str(i)),dict(cell(),pair_id=str(i))]} for i in range(2)]
    manifest={'config_hash':'h','pairs':pairs,'historical':{'0':{'correct':1,'count':2},'20':{'correct':2,'count':2}}}
    def campaign(runtime,fail_at=None):
        return FakeCampaign(manifest,tmp_path,runtime,None,tmp_path,deadline=time.time()+1000,
                            validation=lambda *a,**kw:{'passed':True},fail_at=fail_at)
    first=campaign('old',23)
    with pytest.raises(InterruptedError):first.run()
    # Only large rank serialization checks are mocked; raw response checksums,
    # complete-answer parsing, gate reconstruction, and counts remain real.
    monkeypatch.setattr(module,'checked_result',lambda result,*a,**kw:read_response(result))
    with pytest.raises(ValueError,match='incomplete'):module.summarize(manifest,first.journal,tmp_path)
    second=campaign('new');second.run()
    report=module.summarize(manifest,second.journal,tmp_path)
    assert len(report['points'])==8
    assert all(p['count']==2 and p['correct']==2 for p in report['points'])
    assert report['integrity']['runtimes']==['new','old']
    assert all(t['True_to_True']==2 for t in report['transitions'].values())
    block=next(iter(second.journal.records.values()))
    block['scores']['baseline']['correct']=False
    with pytest.raises(ValueError,match='reconciliation'):module.summarize(manifest,second.journal,tmp_path)

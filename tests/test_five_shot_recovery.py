"""Recovery verification keeps capture identities immutable and cleanup automatic."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from filler.dsv4.patching import atomic_json,digest,file_digest
from scripts.dsv4 import recover_five_shot_recurrence as recovery


@pytest.mark.parametrize('issue',['none','snapshot','output','capture_hash'])
def test_recovery_verifier_separates_controller_source_from_capture(tmp_path,monkeypatch,issue):
    from filler.dsv4 import five_shot_recurrence,recurrence_runtime,recurrence_export
    source=tmp_path/'snapshot'; source.mkdir()
    output=tmp_path/'capture'; output.mkdir()
    path=source/'controller.py';path.write_text('audited new controller')
    plan={'source_hashes':{'scripts/dsv4/one_fact_patching.py':'old'},'cells':[]}
    plan['config_hash']=digest(plan)
    atomic_json(output/'preflight.json',plan)
    manifest={'capture_runtime':str(output),'capture_config_hash':plan['config_hash'],
              'snapshot_sha256':{'controller.py':file_digest(path)},
              'controller_source_changes':{'scripts/dsv4/one_fact_patching.py':{'capture_sha256':'old','scoring_sha256':'new'}}}
    if issue=='capture_hash':manifest['capture_config_hash']='wrong'
    manifest['config_hash']=digest(manifest)
    atomic_json(source/'RECOVERY_SNAPSHOT.json',manifest)
    if issue=='snapshot':path.write_text('changed')
    if issue=='output':output=tmp_path/'other'
    monkeypatch.setattr(recovery,'ROOT',source)
    observed=[]
    monkeypatch.setattr(five_shot_recurrence,'verify_plan',lambda p:observed.append(deepcopy(p)))
    # Register restoration for the functions replaced by the production installer.
    monkeypatch.setattr(recurrence_runtime,'verify_plan',recurrence_runtime.verify_plan)
    monkeypatch.setattr(recurrence_runtime,'read',recurrence_runtime.read)
    monkeypatch.setattr(recurrence_export,'verify_plan',recurrence_export.verify_plan)
    if issue!='none':
        with pytest.raises(ValueError):recovery.install_verifier(output)
    else:
        _,verify=recovery.install_verifier(output)
        assert observed[0]['source_hashes']['scripts/dsv4/one_fact_patching.py']=='new'
        assert plan['source_hashes']['scripts/dsv4/one_fact_patching.py']=='old'
        assert observed[0]['config_hash']!=plan['config_hash']
        # A later approved job uses its own deadline, without changing the
        # expired capture runtime or its scientific identity.
        atomic_json(output/'runtime.json',{'deadline':1,'job_id':'capture-job'})
        monkeypatch.setenv('SLURM_JOB_ID','new-job')
        atomic_json(output/'recovery_execution.json',{
            'snapshot_config_hash':manifest['config_hash'],'capture_config_hash':plan['config_hash'],
            'job_id':'new-job','deadline':9999999999})
        assert recurrence_runtime.read(output/'runtime.json')['deadline']==9999999999
        assert five_shot_recurrence.read(output/'runtime.json')['deadline']==1
        monkeypatch.setenv('SLURM_JOB_ID','different-job')
        with pytest.raises(ValueError,match='different execution'):
            recurrence_runtime.read(output/'runtime.json')
        changed=deepcopy(plan);changed['cells']=[{'different':'prompt'}]
        with pytest.raises(ValueError):verify(changed)


def test_recovery_stops_every_worker_on_failure(tmp_path,monkeypatch):
    from filler.dsv4 import recurrence_runtime
    children=[SimpleNamespace(poll=lambda:1),*[SimpleNamespace(poll=lambda:None) for _ in range(3)]]
    pending=iter(children);stopped=[]
    monkeypatch.setattr(recovery.subprocess,'Popen',lambda *a,**kw:next(pending))
    monkeypatch.setattr(recurrence_runtime,'stop_process',lambda child:stopped.append(child))
    with pytest.raises(RuntimeError,match='failed'):
        recovery.run_workers(tmp_path,'score',time.time()+600)
    assert len(stopped)==4

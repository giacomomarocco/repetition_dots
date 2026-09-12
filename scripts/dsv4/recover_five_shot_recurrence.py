"""Scoring-only recovery of a captured five-shot run using a frozen source copy.

The capture plan remains unchanged. Exactly one audited, controller-only source
change is recorded separately; all numerical/capture sources retain their pins.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def install_verifier(output):
    from filler.dsv4.five_shot_recurrence import read, verify_plan
    from filler.dsv4.patching import digest, file_digest
    from filler.dsv4 import recurrence_runtime, recurrence_export
    recovery = read(ROOT / 'RECOVERY_SNAPSHOT.json')
    if digest({k:v for k,v in recovery.items() if k != 'config_hash'}) != recovery['config_hash']:
        raise ValueError('recovery snapshot manifest checksum mismatch')
    if str(output.resolve()) != recovery['capture_runtime']:
        raise ValueError('recovery belongs to a different capture runtime')
    for name, expected in recovery['snapshot_sha256'].items():
        if file_digest(ROOT / name) != expected:
            raise ValueError(f'frozen recovery source changed: {name}')
    def verify(plan):
        if plan['config_hash'] != recovery['capture_config_hash'] or digest({k:v for k,v in plan.items() if k!='config_hash'}) != plan['config_hash']:
            raise ValueError('original capture plan changed')
        # Preserve and independently verify the original capture identity. This
        # adjusted copy exists only to check current frozen software and weights.
        adapted = {**plan, 'source_hashes': dict(plan['source_hashes'])}
        for name, change in recovery['controller_source_changes'].items():
            if adapted['source_hashes'][name] != change['capture_sha256']:
                raise ValueError('unexpected historical controller source')
            adapted['source_hashes'][name] = change['scoring_sha256']
        adapted['config_hash'] = digest({k:v for k,v in adapted.items() if k!='config_hash'})
        verify_plan(adapted)
    recurrence_runtime.verify_plan = verify
    recurrence_export.verify_plan = verify
    def runtime_read(path):
        value = read(path)
        if Path(path).resolve() == (output / 'runtime.json').resolve():
            execution = read(output / 'recovery_execution.json')
            if (execution['snapshot_config_hash'] != recovery['config_hash']
                    or execution['capture_config_hash'] != recovery['capture_config_hash']
                    or execution['job_id'] != os.environ.get('SLURM_JOB_ID')):
                raise ValueError('recovery deadline belongs to a different execution')
            # The captured runtime stays immutable on disk. Only the scoring
            # controller's walltime check uses the newly approved allocation.
            value = {**value, 'deadline': execution['deadline']}
        return value
    recurrence_runtime.read = runtime_read
    verify(read(output / 'preflight.json'))
    return recovery, verify


def run_workers(output, phase, deadline):
    from filler.dsv4.recurrence_runtime import stop_process
    children, logs = [], []
    try:
        for rank in range(4):
            log = (output / f'recovery_{phase}_gpu{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-m', 'scripts.dsv4.recover_five_shot_recurrence',
                '--output', str(output), '--phase', phase, '--rank', str(rank)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in children):
            if time.time() > deadline or any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(f'recovery {phase} failed or reached deadline; inspect worker logs')
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError(f'recovery {phase} failed')
    finally:
        for process in children:
            stop_process(process)
        for log in logs:
            log.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--phase',choices=('score','mass'))
    parser.add_argument('--rank',type=int,choices=range(4),default=0)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    recovery, verify = install_verifier(output)
    from filler.dsv4.five_shot_recurrence import read
    from filler.dsv4.patching import atomic_json, file_digest
    from filler.dsv4.recurrence_runtime import validate_records, allocation, require_compute, score_worker
    from filler.dsv4.recurrence_analysis import select_groups
    from filler.dsv4.recurrence_export import export
    plan=read(output/'preflight.json')
    validate_records(plan,read(output/'captures.json'))
    if args.preflight:
        print(json.dumps({'passed':True,'snapshot':str(ROOT),'capture_config_hash':plan['config_hash'],
                          'prompts':len(plan['cells']),'readouts':plan['readouts']},indent=2))
        return
    require_compute()
    if args.phase:
        score_worker(output,args.rank,mass=args.phase=='mass')
        return
    info,gpus,deadline=allocation()
    if deadline-time.time()<600:
        raise RuntimeError('insufficient time remaining in the recovery allocation')
    metadata={'snapshot':str(ROOT),'snapshot_config_hash':recovery['config_hash'],
              'capture_config_hash':plan['config_hash'],'job_id':os.environ['SLURM_JOB_ID'],
              'allocation':info,'gpus':gpus,'deadline':deadline,
              'started_at':datetime.now(timezone.utc).isoformat(),'model_load':False,
              'snapshot_manifest':recovery}
    atomic_json(output/'recovery_execution.json',metadata)
    def stop(*_): raise KeyboardInterrupt('recovery interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        run_workers(output,'score',deadline)
        print('Scoring and native/reference validation complete',flush=True)
        select_groups(output)
        run_workers(output,'mass',deadline)
        print('Exact group-mass rescoring complete',flush=True)
        export(output)
        verify(plan)
        metadata.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'recovery_execution.json',metadata)
        complete=read(output/'COMPLETE.json')
        complete['export_sha256']['recovery_execution.json']=file_digest(output/'recovery_execution.json')
        complete['recovery_job_id']=os.environ['SLURM_JOB_ID']
        atomic_json(output/'COMPLETE.json',complete)
        atomic_json(output.parents[1]/'COMPLETE.json',{**complete,'runtime':str(output)})
        print('Recovery complete: '+str(output/'REPORT.md'),flush=True)
    except BaseException as exc:
        metadata.update(status='failed',error=str(exc),ended_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'recovery_execution.json',metadata)
        raise


if __name__=='__main__':
    main()

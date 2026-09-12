"""One extra repeat point using the still-warm five-shot campaign and its baselines."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time

from filler.dsv4.five_shot_repeat import DEFAULT_ROOT, CONDITIONS, control_gate, score, lab
from filler.dsv4.patching import Journal, atomic_json, digest, file_digest
from filler.dsv4.patching_campaign import Campaign, HTTPTransport, read_response, WalltimeReached
from filler.dsv4.patching_logits import validate_logits

EXTENSION_ROOT=DEFAULT_ROOT/'source-6-extension'


def read(path):
    return json.loads(Path(path).read_text())


def base_records(root):
    manifest=read(root/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='config_hash'})!=manifest['config_hash']:
        raise ValueError('parent manifest checksum mismatch')
    records={}
    for line in (root/'results.jsonl').read_text().splitlines():
        envelope=json.loads(line); b=envelope['record']
        if envelope['sha256']!=digest(b) or b['config_hash']!=manifest['config_hash'] or b['record_id'] in records:
            raise ValueError('parent checkpoint integrity failed')
        if set(b['results'])!=set(CONDITIONS) or not b['runtime_gate']['native']['passed']:
            raise ValueError('incomplete parent block or missing native validation')
        records[b['record_id']]=b
    if set(records)!={p['record_id'] for p in manifest['pairs']} or len(records)!=262:
        raise ValueError('requires all 262 completed parent blocks')
    return manifest,records


def positions(cell):
    p={v['label']:v['absolute_position'] for v in cell['positions']}
    dest=[p[f'filler_{i}'] for i in range(7,20)]
    return dest,[p['filler_6']]*13


def prepare(root, parent=DEFAULT_ROOT, *, controller_pid=187390):
    if (root/'results.jsonl').exists():
        raise ValueError('extension already has results')
    base,records=base_records(parent)
    runtimes={b['runtime_id'] for b in records.values()}
    if len(runtimes)!=1:
        raise ValueError('warm reuse requires one parent runtime')
    runtime=runtimes.pop()
    runtime_path=parent/'runtimes'/runtime
    info=read(runtime_path/'runtime.json')
    if (parent/'COMPLETE.json').exists():
        raise ValueError('parent has finished; warm server may already be released')
    source=Path(__file__).resolve()
    manifest={'parent_root':str(parent),'parent_config_hash':base['config_hash'],
        'parent_manifest_sha256':file_digest(parent/'manifest.json'),
        'parent_journal_sha256':file_digest(parent/'results.jsonl'),
        'runtime_id':runtime,'runtime_path':str(runtime_path),'job_id':info['job_id'],
        'hostname':info['hostname'],'deadline':info['deadline'], 'controller_pid':controller_pid,
        'source_index':6,'untouched':7,'replacements':13,'layers':list(range(43)),
        'tp_size':4,'max_new_tokens':8,'source_hashes':{str(source):file_digest(source)},
        'semantics':'reuse exact same-runtime clean k20 residuals and reference responses; '
                    'one identity and one repeat request per example, plus one initial layer42 control; '
                    'all four ranks, final-question fillers only; checkpoint complete two-request extension blocks'}
    manifest['config_hash']=digest(manifest)
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'manifest.json',manifest)
    (root/'source_snapshot.py').write_bytes(source.read_bytes())
    lab(root,'Prepared source index 6: x=7 untouched fillers, 13 replacements, all 262 examples. '
        'Reuse the already measured parent baselines and same-runtime clean captures. '
        '525 new generation requests including identity controls and one layer-42 diagnostic. '
        'The parent controller is paused only while the warm-server extension runs and is resumed in finally.')
    return manifest


def verify(manifest):
    if digest({k:v for k,v in manifest.items() if k!='config_hash'})!=manifest['config_hash']:
        raise ValueError('extension manifest checksum mismatch')
    parent=Path(manifest['parent_root'])
    if file_digest(parent/'manifest.json')!=manifest['parent_manifest_sha256'] or file_digest(parent/'results.jsonl')!=manifest['parent_journal_sha256']:
        raise ValueError('parent inputs changed')
    for name,sha in manifest['source_hashes'].items():
        if file_digest(Path(name))!=sha:
            raise ValueError('extension code changed after preparation')
    return base_records(parent)


class Extension(Campaign):
    def __init__(self, manifest, base, records, root, port):
        self.extension_manifest,self.base_records=manifest,records
        self.manifest=base  # Hook donor metadata uses the original capture namespace.
        self.root,self.runtime_id=root,manifest['runtime_id']
        self.runtime_root=root/'runtimes'/self.runtime_id
        self.transport=HTTPTransport(f'http://127.0.0.1:{port}',Path(manifest['runtime_path'])/'control')
        self.journal=Journal(root,manifest['config_hash'])
        self.deadline=manifest['deadline']-1200  # Reserve time to resume the parent audit.
        self.durations=[]; self.current=None; self.stop_requested=False

    def check_time(self):
        if self.stop_requested or time.time()>self.deadline:
            raise WalltimeReached('extension checkpointed with parent-audit time reserved')

    def run(self):
        diagnostic=None
        for pair in self.manifest['pairs']:
            rid=pair['record_id']
            if rid in self.journal.records: continue
            cell=pair['cells'][1]; parent=self.base_records[rid]
            baseline=parent['results']['no_intervention']; clean=read_response(baseline)
            ids=list(dict.fromkeys(cell['target_token_ids'].values()))
            dest,source=positions(cell); capture=baseline['capture']
            identity=self.execute(cell=cell,token_ids=ids,mode='full_downstream',layers=range(43),
                                  positions=dest,source_positions=dest,donor=capture)
            gate=control_gate(clean,read_response(identity),ids,full_response=True)
            atomic_json(self.runtime_root/'checks'/f"{identity['request_id']}.json",gate)
            if not gate['passed']: raise ValueError('source-6 identity failed')
            if diagnostic is None:
                result=self.execute(cell=cell,token_ids=ids,mode='full_downstream',layers=[42],
                                    positions=dest,source_positions=source,donor=capture)
                check=control_gate(clean,read_response(result),ids,full_response=True)
                diagnostic={'result':result,'gate':check,'parent_record_id':rid}
                atomic_json(self.root/'layer42.json',diagnostic)
                if not check['passed']:raise ValueError('source-6 layer42 invariance failed')
            trial=self.execute(cell=cell,token_ids=ids,mode='full_downstream',layers=range(43),
                               positions=dest,source_positions=source,donor=capture)
            self.journal.append({'record_id':rid,'kind':'source6_block','pair_id':cell['pair_id'],
                'runtime_id':self.runtime_id,'baseline':baseline,'identity':identity,'trial':trial,'identity_gate':gate,
                'score':score(cell,read_response(trial))})
            print(f"Source 6: {len(self.journal.records)}/262 examples complete",flush=True)
        atomic_json(self.root/'REQUESTS_COMPLETE.json',{'config_hash':self.extension_manifest['config_hash'],
                                                     'examples':len(self.journal.records)})


def checked(result,cell,manifest,baseline,*,identity=False,diagnostic=False):
    from filler.dsv4.campaign_hook import validate_ack_records
    for p,h in [('control_path','control_sha256'),('ack_path','ack_sha256')]:
        if file_digest(Path(result[p]))!=result[h]:raise ValueError('extension artifact changed')
    control=read(result['control_path']); dest,source=positions(cell)
    if (control['config_hash']!=manifest['parent_config_hash'] or control['runtime_id']!=manifest['runtime_id']
            or control['cell_id']!=cell['cell_id'] or control['input_ids_hash']!=digest(cell['input_ids'])
            or control['request_id']!=result['request_id'] or control['num_tokens']!=len(cell['input_ids'])
            or control['layers']!=([42] if diagnostic else list(range(43)))
            or control['positions']!=dest or control['source_positions']!=(dest if identity else source)
            or control['donor_capture']!=baseline['capture'] or control['clean_capture'] is not None
            or control['capture_all'] or control['recomputation']!='full_downstream' or control['max_new_tokens']!=8):
        raise ValueError('extension control mapping mismatch')
    acks=read(result['ack_path']);validate_ack_records(acks,control)
    if result['capture']['ranks']!={str(a['rank']):a['capture'] for a in acks}:
        raise ValueError('extension rank references mismatch')
    for a in acks:
        if file_digest(Path(a['capture']['path']))!=a['capture']['sha256']:
            raise ValueError('extension residual checksum mismatch')
    response=read_response(result)
    if response['meta_info'].get('cached_tokens')!=0:raise ValueError('cached extension prefill')
    validate_logits(response,control)
    return response


def analyze(root):
    from filler.addition.accuracy_plot import wilson_interval
    manifest=read(root/'manifest.json');base,parents=verify(manifest)
    journal=Journal(root,manifest['config_hash'])
    if set(journal.records)!=set(parents):raise ValueError('incomplete extension')
    rows=[]
    for pair in base['pairs']:
        rid=pair['record_id'];b=journal.records[rid];cell=pair['cells'][1]
        baseline=parents[rid]['results']['no_intervention']
        if b['baseline']!=baseline or b['pair_id']!=cell['pair_id']:
            raise ValueError('extension baseline alignment mismatch')
        clean=read_response(baseline);ids=list(dict.fromkeys(cell['target_token_ids'].values()))
        identity=checked(b['identity'],cell,manifest,baseline,identity=True)
        gate=control_gate(clean,identity,ids,full_response=True)
        if not gate['passed'] or gate!=b['identity_gate']:raise ValueError('extension identity reconciliation failed')
        trial=checked(b['trial'],cell,manifest,baseline)
        scored=score(cell,trial)
        if scored!=b['score']:raise ValueError('extension raw-response reconciliation failed')
        rows.append({'pair_id':cell['pair_id'],**scored})
    diag=read(root/'layer42.json');pair=next(p for p in base['pairs'] if p['record_id']==diag['parent_record_id'])
    cell=pair['cells'][1];baseline=parents[pair['record_id']]['results']['no_intervention']
    response=checked(diag['result'],cell,manifest,baseline,diagnostic=True)
    gate=control_gate(read_response(baseline),response,list(dict.fromkeys(cell['target_token_ids'].values())),full_response=True)
    if not gate['passed'] or gate!=diag['gate']:raise ValueError('extension layer42 reconciliation failed')
    correct=sum(r['correct'] for r in rows);lo,hi=wilson_interval(correct,262)
    point={'condition':'repeat_6','untouched':7,'correct':correct,'count':262,'accuracy':correct/262,'ci_low':lo,'ci_high':hi}
    atomic_json(root/'analysis/raw_scores.json',rows)
    atomic_json(root/'analysis/summary.json',{'point':point,'config_hash':manifest['config_hash'],
        'parent_config_hash':manifest['parent_config_hash'],'integrity':{'passed':True,'examples':262,'identity_controls':262,'layer42_controls':1}})
    atomic_json(root/'COMPLETE.json',{'status':'complete','config_hash':manifest['config_hash'],
        'parent_config_hash':manifest['parent_config_hash'],'integrity':{'passed':True,'examples':262}})
    lab(root,'Completed source-6 extension and independent raw-capture/score reconciliation: '+json.dumps(point))
    return point


def run(root,port):
    from scripts.dsv4.one_fact_patching import get_json
    manifest=read(root/'manifest.json');base,records=verify(manifest)
    if os.environ.get('SLURM_JOB_ID')!=manifest['job_id'] or socket.gethostname()!=manifest['hostname']:
        raise ValueError('requires the original still-warm allocation/node')
    pid=manifest['controller_pid']
    if b'scripts.dsv4.five_shot_repeat' not in Path(f'/proc/{pid}/cmdline').read_bytes():
        raise ValueError('parent controller identity mismatch')
    info=get_json(f'http://127.0.0.1:{port}/server_info')
    expected=read(Path(manifest['runtime_path'])/'runtime.json')['server']
    if any(info.get(k)!=expected.get(k) for k in ('forward_hooks','model_path','tp_size')):
        raise ValueError('warm server identity differs')
    campaign=Extension(manifest,base,records,root,port)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,campaign.request_stop)
    paused=False
    try:
        os.kill(pid,signal.SIGSTOP);paused=True
        print('Parent audit paused; running source 6 on the warm server.',flush=True)
        campaign.run()
        print(json.dumps(analyze(root),indent=2),flush=True)
    finally:
        if paused:os.kill(pid,signal.SIGCONT)
        print('Parent audit resumed.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--root',type=Path,default=EXTENSION_ROOT)
    p.add_argument('--port',type=int,default=30002)
    args=p.parse_args(argv)
    if args.action=='prepare':prepare(args.root)
    elif args.action=='run':run(args.root,args.port)
    else:print(json.dumps(analyze(args.root),indent=2))

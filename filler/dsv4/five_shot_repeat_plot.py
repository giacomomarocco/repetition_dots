"""Notebook presentation for the completed five-shot repeat campaign."""
from __future__ import annotations

import json
from pathlib import Path

from filler.addition.accuracy_plot import wilson_interval


def load_partial_report(root):
    """Read immutable complete journal envelopes; never repair a live journal.

    This is a checkpoint preview. Final rank-capture reconciliation remains a
    separate completion gate and is not claimed by this reader.
    """
    from filler.addition.one_fact import parse_answer
    from filler.dsv4.patching import digest
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='config_hash'}) != manifest['config_hash']:
        raise ValueError('manifest checksum mismatch')
    expected={p['record_id']:p for p in manifest['pairs']}
    path=root/'results.jsonl'
    data=path.read_bytes() if path.exists() else b''
    blocks={}
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break  # The controller may currently be appending this final line.
        envelope=json.loads(line)
        block=envelope['record']
        rid=block['record_id']
        if (envelope['sha256'] != digest(block) or block['config_hash'] != manifest['config_hash']
                or rid not in expected or rid in blocks or block['kind'] != 'example_block'):
            raise ValueError('invalid, unexpected or duplicated checkpoint block')
        conditions=['baseline','no_intervention']+[f'identity_{s}' for s in range(6)]+[f'repeat_{s}' for s in range(6)]
        if set(block['results']) != set(conditions) or set(block['scores']) != set(conditions):
            raise ValueError('incomplete checkpoint condition set')
        pair=expected[rid]
        if block['pair_id'] != pair['cells'][0]['pair_id']:
            raise ValueError('checkpoint pair mismatch')
        gate=block['runtime_gate']
        if (not gate['native']['passed'] or len(gate['diagnostics']) != 6
                or not all(d['gate']['passed'] for d in gate['diagnostics'])
                or set(block['gates']) != {f'identity_{s}' for s in range(6)}
                or not all(g['passed'] for g in block['gates'].values())):
            raise ValueError('checkpoint validation gates missing or failed')
        for name in conditions:
            cell=pair['cells'][0 if name=='baseline' else 1]
            result,score=block['results'][name],block['scores'][name]
            parsed=parse_answer(score['text'])
            if (result['runtime_id'] != block['runtime_id']
                    or result['capture']['cell_id'] != cell['cell_id']
                    or score['parsed_answer'] != parsed or score['correct'] != (parsed==cell['target'])
                    or not 1 <= len(score['output_ids']) <= 8
                    or score['first_token_correct'] != (score['output_ids'][0]==cell['target_token_ids']['A+X'])):
                raise ValueError('checkpoint scoring or paired baseline mismatch')
        blocks[rid]=block
    if not blocks:
        raise ValueError('no complete example blocks available yet')
    points=[]
    for name in ['baseline','no_intervention']+[f'repeat_{s}' for s in range(6)]:
        n=len(blocks)
        correct=sum(b['scores'][name]['correct'] for b in blocks.values())
        lo,hi=wilson_interval(correct,n)
        points.append({'condition':name,'untouched':int(name[-1])+1 if name.startswith('repeat') else None,
                       'correct':correct,'count':n,'accuracy':correct/n,'ci_low':lo,'ci_high':hi})
    return {'config_hash':manifest['config_hash'],'preliminary':True,'completed_examples':len(blocks),
            'total_examples':len(expected),'points':points,'included_record_ids':sorted(blocks),
            'historical':{str(k):{'correct':sum(expected[rid]['cells'][i]['historical_correct'] for rid in blocks),
                                  'count':len(blocks)} for i,k in enumerate((0,20))},
            'historical_full_dataset':manifest['historical'],
            'validation_status':'complete checkpoint blocks; final all-capture reconciliation pending'}


def load_report(root, *, allow_partial=False):
    root = Path(root)
    if allow_partial and not (root/'COMPLETE.json').exists():
        return include_extension(root, load_partial_report(root))
    report = json.loads((root/'analysis/summary.json').read_text())
    complete = json.loads((root/'COMPLETE.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    rows = json.loads((root/'analysis/raw_scores.json').read_text())
    expected_pairs = {p['cells'][0]['pair_id'] for p in manifest['pairs']}
    if (complete['status'] != 'complete' or not complete['integrity']['passed']
            or report['config_hash'] != complete['config_hash']
            or report['config_hash'] != manifest['config_hash'] or len(expected_pairs) != 262):
        raise ValueError('requires the complete 262-example campaign')
    conditions = ['baseline','no_intervention'] + [f'repeat_{s}' for s in range(6)]
    if [p['condition'] for p in report['points']] != conditions:
        raise ValueError('missing, duplicate or reordered plot conditions')
    for point in report['points']:
        selected = [r for r in rows if r['condition'] == point['condition']]
        if len(selected) != 262 or {r['pair_id'] for r in selected} != expected_pairs:
            raise ValueError('incomplete or duplicated reference/condition counts')
        correct = sum(r['correct'] for r in selected)
        lo, hi = wilson_interval(correct, 262)
        if (point['correct'] != correct or point['count'] != 262 or point['accuracy'] != correct/262
                or point['ci_low'] != lo or point['ci_high'] != hi):
            raise ValueError('plotted count/interval reconciliation failed')
    if [p['untouched'] for p in report['points'][2:]] != list(range(1,7)):
        raise ValueError('incorrect untouched-filler axis')
    return include_extension(root, report)


def include_extension(root, report):
    """Add x=7 only after all 262 extension examples pass their capture audit."""
    from filler.dsv4.patching import digest, file_digest
    from filler.addition.one_fact import parse_answer
    child=Path(root)/'source-6-extension'
    if not (child/'COMPLETE.json').exists():
        return report
    def read(name): return json.loads((child/name).read_text())
    m,done,summary=read('manifest.json'),read('COMPLETE.json'),read('analysis/summary.json')
    if (digest({k:v for k,v in m.items() if k!='config_hash'})!=m['config_hash']
            or any(v['config_hash']!=m['config_hash'] or v['parent_config_hash']!=report['config_hash'] for v in (done,summary))
            or not done['integrity']['passed'] or not summary['integrity']['passed']
            or done['status']!='complete'
            or file_digest(Path(root)/'results.jsonl')!=m['parent_journal_sha256']
            or file_digest(Path(root)/'manifest.json')!=m['parent_manifest_sha256']):
        raise ValueError('extension completion or parent provenance mismatch')
    base=json.loads((Path(root)/'manifest.json').read_text())
    expected={p['record_id']:p['cells'][1] for p in base['pairs']}
    blocks={}
    for line in (child/'results.jsonl').read_text().splitlines():
        e=json.loads(line);b=e['record'];rid=b['record_id']
        if (e['sha256']!=digest(b) or b['config_hash']!=m['config_hash'] or rid in blocks or rid not in expected
                or b['runtime_id']!=m['runtime_id'] or b['pair_id']!=expected[rid]['pair_id']
                or not b['identity_gate']['passed']):
            raise ValueError('extension checkpoint integrity failed')
        scored=b['score'];parsed=parse_answer(scored['text'])
        if scored['parsed_answer']!=parsed or scored['correct']!=(parsed==expected[rid]['target']):
            raise ValueError('extension checkpoint scoring mismatch')
        blocks[rid]=b
    rows=read('analysis/raw_scores.json')
    expected_rows=[{'pair_id':p['cells'][1]['pair_id'],**blocks[p['record_id']]['score']} for p in base['pairs'] if p['record_id'] in blocks]
    if (len(blocks)!=262 or set(blocks)!=set(expected) or rows!=expected_rows
            or any(p['count']!=262 for p in report['points'])):
        raise ValueError('extension requires the same complete 262-example cohort')
    correct=sum(r['correct'] for r in rows);lo,hi=wilson_interval(correct,262)
    point={'condition':'repeat_6','untouched':7,'correct':correct,'count':262,'accuracy':correct/262,'ci_low':lo,'ci_high':hi}
    if summary['point']!=point:raise ValueError('extension plotted counts mismatch')
    return {**report,'points':[*report['points'],point],'extension':{
        'root':str(child),'config_hash':m['config_hash'],'parent_config_hash':m['parent_config_hash'],
        'validation_status':'all 262 extension examples and controls reconciled with raw responses and rank captures'}}


def plot_report(report):
    """Use the caller's rc settings, including fonts and TeX rendering."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8,5))
    points = report['points']
    for p, label, color, hatch, style in zip(points[:2],
            ['Baseline (k=0)','No intervention'], ['0.78','0.55'], [None,'///'], ['--',':']):
        ax.axhspan(p['ci_low'],p['ci_high'],facecolor=color,edgecolor='0.4',alpha=.45,
                   hatch=hatch,label=label,zorder=0,linewidth=.5)
        ax.axhline(p['accuracy'],color='0.35',linestyle=style,linewidth=1,zorder=1)
    selected = points[2:]
    ax.errorbar([p['untouched'] for p in selected], [p['accuracy'] for p in selected],
        yerr=[[p['accuracy']-p['ci_low'] for p in selected],
              [p['ci_high']-p['accuracy'] for p in selected]],
        marker='o',linestyle='none',capsize=4,linewidth=2,label='Repeat intervention',zorder=3)
    low,high=min(p['ci_low'] for p in points),max(p['ci_high'] for p in points)
    margin=max(.025,(high-low)*.12)
    title=f"One-fact addition — 5-shot repeat patching (n = {points[0]['count']})"
    max_x=max(p['untouched'] for p in selected)
    ax.set(xlabel='Number of fillers left untouched',ylabel='Exact-answer accuracy',
           title=title,
           xlim=(.5,max_x+.5),ylim=(low-margin,high+margin),xticks=range(1,max_x+1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False,fontsize=11)
    fig.tight_layout()
    return fig,ax


def plot_five_shot_repeat(root, *, allow_partial=False):
    report=load_report(root, allow_partial=allow_partial)
    fig,ax=plot_report(report)
    return fig,ax,report

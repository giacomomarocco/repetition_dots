"""Execute the patching section in the notebook's actual selected kernel."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK=ROOT/'notebooks/addition_accuracy.ipynb'
RUN=ROOT/'runs/deepseek-v4-flash/one-fact-patching-five-shot-repeat'


def execute(*, fixture=False):
    from jupyter_client import KernelManager
    def code_cell(source):
        return {'cell_type':'code','metadata':{},'source':source,'outputs':[],'execution_count':None}
    original=json.loads(NOTEBOOK.read_text())
    selected=[c for c in original['cells'] if c.get('id')=='five-shot-repeat-plot']
    if len(selected)!=1:
        raise ValueError('expected one dedicated patching cell')
    output_root=RUN/'notebook-validation'
    output_root.mkdir(parents=True,exist_ok=True)
    cells=[code_cell('''import sys, json, matplotlib
from matplotlib.dviread import find_tex_file
print(json.dumps({'python':sys.executable,'matplotlib':matplotlib.__version__,
                  'rc_file':matplotlib.matplotlib_fname(),
                  'font_family':list(matplotlib.rcParams['font.family']),
                  'font_serif':list(matplotlib.rcParams['font.serif']),
                  'font_size':matplotlib.rcParams['font.size'],
                  'text_usetex':matplotlib.rcParams['text.usetex'],
                  'cmr10_tfm':str(find_tex_file('cmr10.tfm'))}))
'''),code_cell(''.join(original['cells'][1]['source']))]
    plot_cell=code_cell(''.join(selected[0]['source']))
    if fixture:
        from filler.addition.accuracy_plot import wilson_interval
        fixture_root=Path(tempfile.mkdtemp(prefix='five-shot-repeat-plot-'))
        (fixture_root/'analysis').mkdir()
        manifest=json.loads((RUN/'manifest.json').read_text())
        conditions=['baseline','no_intervention']+[f'repeat_{s}' for s in range(6)]
        points,rows=[],[]
        for name,correct in zip(conditions,[137,182,112,122,132,142,152,162]):
            lo,hi=wilson_interval(correct,262)
            points.append({'condition':name,'untouched':int(name[-1])+1 if name.startswith('repeat') else None,
                           'correct':correct,'count':262,'accuracy':correct/262,'ci_low':lo,'ci_high':hi})
            rows.extend({'pair_id':p['cells'][0]['pair_id'],'condition':name,'correct':i<correct}
                        for i,p in enumerate(manifest['pairs']))
        data={'config_hash':manifest['config_hash'],'points':points,'historical':manifest['historical']}
        for name,value in [('manifest.json',manifest),('COMPLETE.json',{'status':'complete',
            'config_hash':manifest['config_hash'],'integrity':{'passed':True}}),
            ('analysis/summary.json',data),('analysis/raw_scores.json',rows)]:
            (fixture_root/name).write_text(json.dumps(value))
        plot_cell['source']=plot_cell['source'].replace("repeat_root = PROJECT_ROOT / 'runs/deepseek-v4-flash/one-fact-patching-five-shot-repeat'",
                                                  f'repeat_root = Path({str(fixture_root)!r})')
        plot_cell['source']=plot_cell['source'].replace('display(repeat_fig)',
                          "repeat_ax.set_title('Rendering fixture — synthetic patching counts')\n    display(repeat_fig)")
    cells.append(plot_cell)
    cells.append(code_cell('''
repeat_fig.canvas.draw()
renderer=repeat_fig.canvas.get_renderer()
width,height=repeat_fig.canvas.get_width_height()
lo,hi=repeat_ax.get_ylim()
visible_ylabels=[label for y,label in zip(repeat_ax.get_yticks(),repeat_ax.get_yticklabels()) if lo <= y <= hi]
for artist in [repeat_ax.xaxis.label,repeat_ax.yaxis.label,repeat_ax.title,repeat_ax.get_legend(),
               *repeat_ax.get_xticklabels(),*visible_ylabels]:
    box=artist.get_window_extent(renderer)
    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width+1 and box.y1 <= height+1, (artist,box,width,height)
lo,hi=repeat_ax.get_ylim()
assert all(lo < p['ci_low'] <= p['ci_high'] < hi for p in repeat_report['points'])
print('All interval bounds and labels fit the rendered canvas.')
'''))
    nb={'nbformat':4,'nbformat_minor':5,'metadata':original['metadata'],'cells':cells}
    kernel=original['metadata']['kernelspec']['name']
    km=KernelManager(kernel_name=kernel)
    km.start_kernel(cwd=str(ROOT))
    client=km.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        for cell in cells:
            msg_id=client.execute(cell['source'],store_history=True)
            while True:
                msg=client.get_iopub_msg(timeout=180)
                if msg.get('parent_header',{}).get('msg_id') != msg_id:
                    continue
                kind,content=msg['header']['msg_type'],msg['content']
                if kind=='error':
                    raise RuntimeError('Kernel execution failed: '+ '\n'.join(content['traceback']))
                if kind=='execute_input':
                    cell['execution_count']=content['execution_count']
                elif kind=='stream':
                    cell['outputs'].append({'output_type':'stream','name':content['name'],'text':content['text']})
                elif kind in ('display_data','execute_result'):
                    out={'output_type':kind,'data':content['data'],'metadata':content['metadata']}
                    if kind=='execute_result': out['execution_count']=content['execution_count']
                    cell['outputs'].append(out)
                elif kind=='status' and content['execution_state']=='idle':
                    break
    finally:
        client.stop_channels()
        km.shutdown_kernel(now=True)
    images=[o['data']['image/png'] for c in nb['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    if not images:
        raise ValueError('selected notebook kernel emitted no inline PNG')
    status='complete' if (RUN/'COMPLETE.json').exists() else 'checkpoint_preview'
    stem='fixture' if fixture else ('actual' if status=='complete' else 'preview')
    for i,encoded in enumerate(images):
        (output_root/f'{stem}-{i}.png').write_bytes(base64.b64decode(encoded))
    (output_root/f'{stem}.ipynb').write_text(json.dumps(nb,indent=1))
    report={'passed':True,'fixture':fixture,'data_status':status,'kernel':kernel,
            'inline_images':len(images),'environment_output':nb['cells'][0]['outputs'],
            'figure_checks':nb['cells'][-1]['outputs']}
    (output_root/f'{stem}.json').write_text(json.dumps(report,indent=2))
    if not fixture:
        # Preserve edits made in the user's notebook while the kernel ran.
        latest=json.loads(NOTEBOOK.read_text())
        current=[c for c in latest['cells'] if c.get('id')=='five-shot-repeat-plot']
        if len(current)!=1 or current[0]['source']!=selected[0]['source']:
            raise RuntimeError('notebook plot cell changed during validation; rendered artifacts retained')
        current[0]['outputs']=plot_cell['outputs']
        current[0]['execution_count']=plot_cell['execution_count']
        NOTEBOOK.write_text(json.dumps(latest,indent=1,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',action='store_true')
    execute(fixture=parser.parse_args().fixture)

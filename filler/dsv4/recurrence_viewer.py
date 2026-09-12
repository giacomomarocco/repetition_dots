"""Small artifact-only browser viewer with Unicode text and exact top-ten details."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from filler.dsv4.patching import file_digest
from filler.dsv4.five_shot_recurrence import read


def write_viewer(output):
    output=Path(output).resolve()
    complete=read(output/'COMPLETE.json')
    if not complete.get('passed') or complete.get('status')!='complete':
        raise ValueError('a completed analysis is required')
    def checked(name):
        if file_digest(output/name)!=complete['export_sha256'][name]:
            raise ValueError(f'artifact checksum mismatch: {name}')
        return output/name
    representative=read(checked('representative.json'))
    vocab=read(checked('vocabulary.json'))
    groups=read(checked('selected_groups.json'))
    with checked('tables/top20_textual.csv').open() as f:
        leaderboard=list(csv.DictReader(f))
    blocks={}
    used=set()
    for k in (0,20):
        record=representative[f'k{k}'];cell=record['cell']
        with np.load(checked(f"scores/{cell['cell_id']}.npz"),allow_pickle=False) as scores:
            values={name:scores[name].tolist() for name in ('top_ids','top_logprob','target_logprob','target_rank')}
            used.update(map(int,scores['top_ids'].ravel()))
        with np.load(checked(f"mass/{cell['cell_id']}.npz"),allow_pickle=False) as mass:
            values['group_logmass']=mass['group_logmass'].tolist()
        blocks[k]={**values,'positions':cell['positions'],'prompt':cell['rendered_prompt'],
                   'A':cell['answer_value'],'X':cell['addend'],'sum':cell['target'],
                   'response':record['response_text'],'correct':record['correct'],
                   'demonstrations':cell['demonstration_token_ids'],'question':cell['question_token_ids']}
    tokens={t:{'text':vocab['tokens'][t],'group':vocab['groups'][vocab['token_group'][t]]} for t in used}
    data={'pair_id':representative['pair_id'],'overlap':representative['mean_top10_overlap'],
          'blocks':blocks,'tokens':tokens,'groups':groups,'leaderboard':leaderboard}
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    path=output/'recurrence_viewer.html'
    path.write_text(TEMPLATE.replace('/*DATA*/',payload))
    return path


TEMPLATE=r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five-shot square J-Lens recurrence</title>
<style>
body{font:15px system-ui,"Noto Sans CJK SC",sans-serif;color:#17243b;background:#f5f7fa;margin:24px;line-height:1.45}
main{max-width:1800px;margin:auto}h1{font-size:27px}h2{font-size:20px}section{background:white;border:1px solid #dce3ec;border-radius:8px;padding:18px;margin:18px 0}
a{color:#155ca7}label{margin-right:20px}select{font:inherit;padding:7px;max-width:100%}.scroll{overflow:auto}svg{display:block}
table{border-collapse:collapse;width:100%}th,td{padding:7px 10px;border-bottom:1px solid #e4e9ef;text-align:left}th{background:#eef3f8;position:sticky;top:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:12px;font-size:13px}.muted{color:#4d6075}.stats{font-weight:600}
.cell{cursor:pointer}.cell:hover rect{stroke:#e15100;stroke-width:2.5}.details{max-height:460px;overflow:auto}.mono{font-family:ui-monospace,monospace}
</style><main>
<h1>Square J-Lens recurrence in five-shot addition</h1>
<p>262 facts, k=0 and k=20; 42 square-lens layers; 330,120 position/layer readouts. All native readout and reference checks passed.</p>
<p>These are recurring lens readouts, not generated sentences or causal evidence. The condition changes filler in all five demonstrations and the target.</p>
<p><a href="REPORT.md">Full report</a> · <a href="five_shot_square_recurrence.executed.ipynb">Executed notebook</a> · <a href="tables/recurrence.csv.gz">All recurrence summaries</a> · <a href="tables/paired_differences.csv">Paired differences</a> · <a href="tables/outcomes.csv">Native outcomes</a></p>
<section><h2>Recurring filler words/wordpieces</h2>
<p>Ranked by mean within-prompt top-10 persistence over k20 filler cells. Breadth is the fraction of prompts with at least one occurrence. Variants are stripped and case-folded, with duplicates removed within each top-10 list. Intervals use 2,000 paired-fact bootstrap resamples, seed 42.</p>
<div class="scroll"><table id="leaderboard"></table></div></section>
<section><h2>Typical successful trajectory</h2><p id="selection"></p>
<label>Color metric <select id="metric"><option value="top">Top-token probability</option><optgroup id="words" label="Exact recurring-group probability mass"></optgroup><optgroup label="Exact numeric-token probability"><option value="target:0">A</option><option value="target:1">X</option><option value="target:2">A+X</option></optgroup></select></label>
<p class="muted">Color runs from 0 (pale) to 1 (blue). Labels show the top token; click any cell for exact top-ten details. Shared positions align vertically. Gray columns have no filler at k=0. All group probabilities include vocabulary variants outside the top ten.</p>
<div id="heatmaps"></div></section>
<section><h2 id="detailtitle">Cell details</h2><div class="details"><table id="details"></table></div></section>
<section><h2>Complete paired prompts and native responses</h2><div id="prompts"></div></section>
<p class="muted">This viewer renders Unicode text in your browser. Static PNG/PDF figures are also available; the compute node's default font omits some CJK glyphs. Exact UTF-8 spellings, token IDs, scores, and annotations are preserved here and in the CSV/NumPy artifacts.</p>
</main><script>
'use strict';
const D=/*DATA*/;
const el=(tag,text)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x};
const pct=x=>(100*Number(x)).toFixed(2)+'%';
function table(id,head,rows){const t=document.getElementById(id);t.replaceChildren();const h=el('tr');head.forEach(s=>h.append(el('th',s)));t.append(h);for(const row of rows){const tr=el('tr');for(const v of row)tr.append(v instanceof Node?v:el('td',v));t.append(tr)}}
const ranks=D.leaderboard.map((r,i)=>{const links=el('td');for(const metric of ['top10','probability']){const a=el('a',metric+' PNG');a.href='figures/word_'+String(i).padStart(2,'0')+'_'+metric+'.png';links.append(a,document.createTextNode(' '))}return [String(i+1),r.text,pct(r.persistence),pct(r.persistence_low)+'–'+pct(r.persistence_high),pct(r.breadth),pct(r.in_demonstrations_fraction),pct(r.in_question_fraction),links]});
table('leaderboard',['Rank','Word/wordpiece','Persistence','95% interval','Breadth','In demonstrations','In target question','Figures'],ranks);
document.getElementById('selection').textContent=D.pair_id+' · '+pct(D.overlap)+' mean top-ten overlap. Selected among k20-correct prompts by maximum average corresponding-cell overlap with every k20 prompt; ties use pair_id.';
D.groups.forEach((g,i)=>{const option=el('option',g.wordpiece);option.value='group:'+i;document.getElementById('words').append(option)});
function detail(k,l,p){const b=D.blocks[k],position=b.positions[p];document.getElementById('detailtitle').textContent='k='+k+', layer '+l+', '+position.label+' (absolute token '+position.absolute_position+', '+JSON.stringify(position.token)+')';const rows=b.top_ids[l][p].map((id,j)=>[String(j+1),String(id),JSON.stringify(D.tokens[id].text),D.tokens[id].group,b.top_logprob[l][p][j].toFixed(7),pct(Math.exp(b.top_logprob[l][p][j])),b.demonstrations.includes(id)?'yes':'no',b.question.includes(id)?'yes':'no']);table('details',['Rank','Token ID','Exact decoded text','Word/wordpiece','Full-vocab log P','Probability','In demos','In question'],rows)}
const NS='http://www.w3.org/2000/svg';
function svgEl(tag,attrs,text){const x=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,v);if(text!==undefined)x.textContent=text;return x}
function render(){const container=document.getElementById('heatmaps');container.replaceChildren();const metric=document.getElementById('metric').value;for(const k of [0,20]){const b=D.blocks[k];container.append(el('h3','k='+k+' · A='+b.A+', X='+b.X+', expected='+b.sum+' · response '+JSON.stringify(b.response)+' · '+(b.correct?'correct':'wrong')));const scroll=el('div');scroll.className='scroll';const svg=svgEl('svg',{width:1910,height:1160,viewBox:'0 0 1910 1160',role:'img','aria-label':'Layer by position heatmap for k='+k});const positions=D.blocks[20].positions;for(let c=0;c<25;c++){svg.append(svgEl('text',{x:68+c*73,y:14,'font-size':10,transform:'rotate(55 '+(68+c*73)+' 14)'},positions[c].label))}for(let l=0;l<42;l++){const y=90+(41-l)*25;svg.append(svgEl('text',{x:12,y:y+16,'font-size':12},'L'+l));for(let c=0;c<25;c++){const p=b.positions.findIndex(x=>x.label===positions[c].label);if(p<0){svg.append(svgEl('rect',{x:52+c*73,y,width:72,height:24,fill:'#eef0f3'}));continue}let logp=b.top_logprob[l][p][0];if(metric.startsWith('group:'))logp=b.group_logmass[l][p][Number(metric.split(':')[1])];if(metric.startsWith('target:'))logp=b.target_logprob[l][p][Number(metric.split(':')[1])];const prob=Math.exp(logp),g=svgEl('g',{class:'cell',tabindex:0});g.append(svgEl('rect',{x:52+c*73,y,width:72,height:24,fill:'rgb('+(245-215*prob)+','+(248-150*prob)+','+(252-70*prob)+')'}));let text=JSON.stringify(D.tokens[b.top_ids[l][p][0]].text);if(text.length>12)text=text.slice(0,11)+'…';g.append(svgEl('text',{x:88+c*73,y:y+16,'text-anchor':'middle','font-size':9,fill:prob>.6?'white':'#18253a'},text));g.append(svgEl('title',{},'k='+k+' L'+l+' '+positions[c].label+'; selected probability '+prob.toPrecision(6)+'\n'+b.top_ids[l][p].map((id,j)=>(j+1)+'. '+JSON.stringify(D.tokens[id].text)+' [ID '+id+'] logP='+b.top_logprob[l][p][j]).join('\n')));g.addEventListener('click',()=>detail(k,l,p));g.addEventListener('keydown',e=>{if(e.key==='Enter')detail(k,l,p)});svg.append(g)}}scroll.append(svg);container.append(scroll)}}
document.getElementById('metric').addEventListener('change',render);render();detail(20,20,1);
for(const k of [0,20]){const b=D.blocks[k],c=document.getElementById('prompts');c.append(el('h3','k='+k+' · native response '+JSON.stringify(b.response)));c.append(el('pre',b.prompt))}
</script></html>'''

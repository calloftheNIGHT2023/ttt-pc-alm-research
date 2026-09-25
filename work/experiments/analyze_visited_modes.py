import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();proto=json.loads((a.results/'protocol.json').read_text());rows=json.loads((a.results/'modes.json').read_text())
assert len(rows)==len(proto['seeds'])*len(proto['configs'])*len(proto['stages'])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
assert all(hashlib.sha256((a.results/r['trace_file']).read_bytes()).hexdigest()==r['trace_sha256'] for r in rows)
summary=[]
for c in proto['configs']:
    for n in proto['stages']:
        group=[r for r in rows if r['method']==c['name'] and r['n_context']==n];gains=[r['extra_absolute_prior_mass'] for r in group]
        summary.append(dict(method=c['name'],n_context=n,task_count=len(group),tasks_with_positive_extra=sum(g>0 for g in gains),
                            positive_extra_modes=sum(r['positive_extra_patterns'] for r in group),positive_retained_modes=sum(r['positive_retained_patterns'] for r in group),
                            unique_visited_patterns=sum(r['unique_visited_patterns'] for r in group),
                            mean_retained_fraction_of_observed_union=float(np.mean([r['retained_fraction_of_observed_union'] for r in group if r['retained_fraction_of_observed_union'] is not None])),
                            max_extra_absolute_prior_mass=max(gains),trace_array_bytes_total=sum(r['trace_array_bytes'] for r in group)))
out=a.results.parent/'visited_modes_analysis';out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(dict(summary=summary,all_paths_and_original_geometry_match=True,source_and_trace_hashes_match=True,
    scope='no queries, no risk improvement claim; relative coverage only within visited union'),indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');xx=np.arange(len(proto['configs']));width=.35
for i,n in enumerate(proto['stages']):
    group=[r for r in summary if r['n_context']==n]
    axes[0].bar(xx+(i-.5)*width,[r['tasks_with_positive_extra'] for r in group],width=width,label=f'n={n}')
    axes[1].bar(xx+(i-.5)*width,[1-r['mean_retained_fraction_of_observed_union'] for r in group],width=width,label=f'n={n}')
for ax in axes:ax.set_xticks(xx,[c['name'].replace('_','\n') for c in proto['configs']],fontsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
axes[0].set(ylabel='Tasks out of 16',title='Positive-volume visited modes not retained',ylim=(0,17))
axes[1].set(ylabel='Mean additional fraction within observed union',title='Not coverage of the unknown full posterior',ylim=(0,1))
fig.savefig(out/'visited_modes.png',dpi=160);plt.close(fig)
print(json.dumps(dict(summary=summary),indent=2))

import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streaming_branch_projection as base
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--points',type=Path,required=True);a=p.parse_args()
proto=json.loads((a.results/'protocol.json').read_text());rows=json.loads((a.results/'regions.json').read_text());previous=json.loads((a.points/'modes.json').read_text());old={(r['method'],r['seed'],r['n_context']):r for r in previous}
assert len(rows)==len(proto['seeds'])*len(proto['configs'])*len(proto['stages'])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
pooled={};mode_sets={};maxgap=0.
for r in rows:
    key=(r['method'],r['seed'],r['n_context']);prev=old[key]
    with np.load(a.points/prev['trace_file']) as data:x=data['x']
    modes={base.pattern(x,np.asarray(z['representative'])).astype(np.uint8).tobytes().hex():z['volume'] for z in prev['positive_retained']}
    modes.update({z['pattern']:z['volume'] for z in r['positive_extra']});mode_sets[key]=modes
    pool=pooled.setdefault((r['seed'],r['n_context']),{})
    for k,v in modes.items():
        if k in pool:maxgap=max(maxgap,abs(v-pool[k]))
        pool[k]=v
summary=[]
for c in proto['configs']:
    for n in proto['stages']:
        group=[r for r in rows if r['method']==c['name'] and r['n_context']==n]
        cover=[]
        for r in group:
            total=sum(pooled[(r['seed'],n)].values());mass=sum(mode_sets[(c['name'],r['seed'],n)].values())
            cover.append(mass/total if total else np.nan)
        summary.append(dict(method=c['name'],n_context=n,tasks=len(group),tasks_with_extra=sum(r['positive_extra_patterns']>0 for r in group),
                            positive_extra_patterns=sum(r['positive_extra_patterns'] for r in group),positive_retained_patterns=sum(r['positive_retained_patterns'] for r in group),
                            unique_visited_patterns=sum(r['unique_visited_patterns'] for r in group),geometry_extra_calls=sum(r['extra_geometry_calls'] for r in group),
                            mean_original_fraction_of_own_union=float(np.mean([r['retained_fraction_of_observed_union'] for r in group if r['retained_fraction_of_observed_union'] is not None])),
                            mean_full_visited_fraction_of_pooled_union=float(np.nanmean(cover)),
                            max_compact_bank_bytes=max(r['compact_bank_and_signature_bytes'] for r in group)))
out=a.results.parent/'visited_regions_analysis';out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(dict(summary=summary,source_hashes_match=True,shared_mode_volume_max_difference=maxgap,
    scope='all visited regions; no query prediction and no full-posterior coverage claim'),indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');xx=np.arange(len(proto['configs']));width=.35
for i,n in enumerate(proto['stages']):
    group=[r for r in summary if r['n_context']==n]
    axes[0].bar(xx+(i-.5)*width,[r['tasks_with_extra'] for r in group],width,label=f'n={n}')
    axes[1].bar(xx+(i-.5)*width,[r['mean_full_visited_fraction_of_pooled_union'] for r in group],width,label=f'n={n}')
for ax in axes:ax.set_xticks(xx,[c['name'].replace('_','\n') for c in proto['configs']],fontsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
axes[0].set(ylabel='Tasks out of 16',ylim=(0,17),title='Visited cells contain new feasible mass')
axes[1].set(ylabel='Mean volume fraction',ylim=(0,1.03),title='Coverage within four-method visited union only')
fig.savefig(out/'visited_regions.png',dpi=160);plt.close(fig);print(json.dumps(dict(summary=summary,shared_mode_volume_max_difference=maxgap),indent=2))

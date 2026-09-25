import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--candidate',default='input_exact240');args=p.parse_args()
proto=json.loads((args.results/'protocol.json').read_text());rows=json.loads((args.results/'episodes.json').read_text())
assert len(rows)==proto['count']*len(proto['configs'])*len(proto['stages'])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
seeds=list(range(proto['seed0'],proto['seed0']+proto['count']));names=[c['name'] for c in proto['configs']];summary=[];arrays={}
for name in names:
    group=[r for r in rows if r['method']==name]
    mse=np.array([np.mean([r['query_mse'] for r in group if r['seed']==seed]) for seed in seeds]);seconds=np.array([sum(r['adaptation_seconds']+r['read_queries_seconds'] for r in group if r['seed']==seed) for seed in seeds]);arrays[name]=(mse,seconds)
    summary.append({'method':name,'mean_trajectory_mse':float(mse.mean()),'median_full_seconds':float(np.median(seconds)),
        'stage_mse':[float(np.mean([r['query_mse'] for r in group if r['n_context']==n])) for n in proto['stages']],
        'stage_feasible':[sum(r['support_feasible'] for r in group if r['n_context']==n) for n in proto['stages']],
        'per_stream_mse':mse.tolist(),'per_stream_seconds':seconds.tolist()})
candidate=args.candidate;cm,ct=arrays[candidate];rng=np.random.default_rng(661455);idx=rng.integers(0,len(seeds),(20000,len(seeds)));comparisons=[]
for name in names:
    if name==candidate:continue
    mm,tt=arrays[name];dm=cm-mm;dt=ct-tt
    comparisons.append({'baseline':name,'candidate_minus_baseline_mean_mse':float(dm.mean()),'paired_descriptive_95_mse_interval':np.quantile(dm[idx].mean(axis=1),[.025,.975]).tolist(),
        'candidate_better_streams':int(np.sum(dm<0)),'candidate_minus_baseline_median_seconds':float(np.median(dt)),
        'paired_descriptive_95_median_time_interval':np.quantile(np.median(dt[idx],axis=1),[.025,.975]).tolist()})
out=args.results.parent/(args.results.name+'_analysis');out.mkdir(exist_ok=True)
result={'summary':summary,'comparisons':comparisons,'source_hashes_match':True,'scope':'development resource frontier; intervals descriptive and no noninferiority claim'}
(out/'frontier.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
table=['|method|trajectory MSE|stage MSE|stage feasible|full seconds|','|---|---:|---|---|---:|']
for r in summary:table.append(f'|{r["method"]}|{r["mean_trajectory_mse"]:.8g}|'+ ' / '.join(f'{v:.6g}' for v in r['stage_mse'])+f'|{r["stage_feasible"]}|{r["median_full_seconds"]:.4f}|')
(out/'frontier_table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
for i,r in enumerate(summary):
    marker='*' if r['method']==candidate else ('s' if r['method'].startswith('input') else 'o')
    ax.scatter(r['median_full_seconds'],r['mean_trajectory_mse'],label=r['method'],marker=marker,s=100 if marker=='*' else 55)
ax.set(xscale='log',yscale='log',xlabel='Full online fit + query read seconds (log)',ylabel='Mean trajectory query MSE (log)',
    title=f'{candidate} vs stronger BP budgets: {len(seeds)} development streams')
ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8);ax.grid(alpha=.15);fig.savefig(out/'input_frontier.png',dpi=160);plt.close(fig)
print(json.dumps({'summary':summary,'comparisons':comparisons}),flush=True)

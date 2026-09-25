import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--meta',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);a=p.parse_args()
proto=json.loads((a.meta/'protocol.json').read_text());rows=json.loads((a.meta/'episodes.json').read_text());training=json.loads((a.meta/'training.json').read_text());details=json.loads((a.meta/'training_summary.json').read_text())
assert len(rows)==len(proto['configs'])*len(proto['evaluation_seeds'])*4
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
assert all(hashlib.sha256((a.meta/(r['method']+'.pt')).read_bytes()).hexdigest()==r['checkpoint_sha256'] for r in details)
controls=json.loads((a.controls/'episodes.json').read_text());names=['alm64_wide','alm64_priorbox','context_alm20','context_adam60','context_adam240']
allrows=rows+[r for r in controls if r['method'] in names];names +=[r['name'] for r in proto['configs']]
seeds=proto['evaluation_seeds'];summary=[];arrays={}
for name in names:
    group=[r for r in allrows if r['method']==name]
    ms=np.array([np.mean([r['query_mse'] for r in group if r['seed']==s]) for s in seeds]);arrays[name]=ms
    seconds=[sum(r['adaptation_seconds']+r['read_queries_seconds'] for r in group if r['seed']==s) for s in seeds]
    record=dict(method=name,trajectory_mse=float(ms.mean()),median_full_seconds=float(np.median(seconds)),
                stage_mse=[float(np.mean([r['query_mse'] for r in group if r['n_context']==n])) for n in [4,8,16,24]],
                stage_feasible=[sum(r['support_feasible'] for r in group if r['n_context']==n) for n in [4,8,16,24]])
    if name not in ['alm64_wide','alm64_priorbox','context_alm20','context_adam60','context_adam240']:
        record.update(next(r for r in details if r['method']==name));record['fast_state_bytes']=max(r['fast_state_bytes'] for r in group);record['shared_model_bytes']=group[0]['shared_model_bytes']
    summary.append(record)
rng=np.random.default_rng(661455);idx=rng.integers(0,len(seeds),(20000,len(seeds)));comparisons=[]
for candidate in ['alm64_wide','alm64_priorbox','context_alm20']:
    for baseline in names:
        if baseline==candidate:continue
        diff=arrays[candidate]-arrays[baseline]
        comparisons.append(dict(candidate=candidate,baseline=baseline,mean_difference=float(diff.mean()),
                                descriptive_95_interval=np.quantile(diff[idx].mean(1),[.025,.975]).tolist(),better_streams=int(np.sum(diff<0))))
out=a.meta.parent/'meta_analysis';out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(dict(summary=summary,comparisons=comparisons,source_and_checkpoint_hashes_match=True,
    scope='old development streams; finite outer budget; CPU timings measured in separate sequential runs, not randomized cross-family speed evidence'),indent=2),encoding='utf-8')
table=['|method|trajectory MSE|stages|full CPU seconds|outer training seconds|','|---|---:|---|---:|---:|']
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.8g}|'+ ' / '.join(f'{v:.7g}' for v in r['stage_mse'])+f'|{r["median_full_seconds"]:.5f}|{r.get("outer_training_seconds",0):.2f}|')
(out/'table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for c in proto['configs']:
    name=c['name'];group=[r for r in training if r['method']==name]
    if len(group)>1:axes[0].plot([r['step'] for r in group],[r['validation_raw_query_mse'] for r in group],marker='.',label=name)
for r in summary:
    if r['method'] in ['alm64_wide','context_alm20','context_adam60']+[c['name'] for c in proto['configs']]:
        axes[1].plot([4,8,16,24],r['stage_mse'],marker='.',label=r['method'])
axes[0].set(xlabel='Outer training step',ylabel='Validation raw query MSE',title='Equal 32,000-task outer budget')
axes[1].set(xlabel='Observed contexts',ylabel='Clipped query MSE (log)',yscale='log',title='16 historically observed development streams')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(out/'matched_meta.png',dpi=160);plt.close(fig)
print(json.dumps(dict(summary=summary),indent=2))

import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
protocol=json.loads((args.results/'protocol.json').read_text());data=json.loads((args.results/'trace.json').read_text());rows=data['rows']
assert all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest for name,digest in protocol['source_sha256'].items())
assert len(rows)==len(protocol['seeds'])*protocol['restarts']*sum(sum(i<=steps for i in protocol['checkpoints']) for steps in protocol['methods'].values())
summaries=[];checkpoints=[]


def summarize(group):
    values=np.array([r['line_losses'] for r in group]);old=values[:,0];full=values[:,-1];partial=np.min(values[:,1:-1],axis=1)
    active=old>1e-12;worse=(full>old+1e-12)&active;rescue=worse&(partial<old-1e-12)
    best=np.argmin(values,axis=1);q=[r['conditional_after']-r['conditional_before'] for r in group if r['conditional_before'] is not None]
    pairs={(r['seed'],r['restart']) for r,t in zip(group,rescue) if t};seeds={s for s,_ in pairs}
    return dict(events=len(group),nonzero_loss_events=int(active.sum()),full_step_worse=int(worse.sum()),partial_rescues=int(rescue.sum()),
                rescued_task_restart_pairs=len(pairs),rescued_tasks=len(seeds),
                full_worse_fraction=float(worse.sum()/max(active.sum(),1)),rescue_fraction_of_worse=float(rescue.sum()/max(worse.sum(),1)),
                partial_step_strictly_better_than_full=int(np.sum((partial<full-1e-12)&active)),
                alpha_counts={str(a):int(np.sum(best==i)) for i,a in enumerate(protocol['alphas'])},
                conditional_energy_max_increase=max(q) if q else None,
                mean_free_true_branch_mismatch=float(np.mean([r['free_true_branch_mismatch'] for r in group])) if q else None,
                positive_diagnostic_gradient_dot=int(sum(r['diagnostic_gradient_dot']>1e-12 for r in group)),
                mean_full_loss_change=float(np.mean(full-old)),mean_best_partial_loss_change=float(np.mean(partial-old)),
                mean_grid_loss_change=float(np.mean(np.min(values,axis=1)-old)))


for name in protocol['methods']:
    group=[r for r in rows if r['method']==name];summaries.append(dict(method=name,**summarize(group)))
    for iteration in protocol['checkpoints']:
        sub=[r for r in group if r['iteration']==iteration]
        if sub:checkpoints.append(dict(method=name,iteration=iteration,**summarize(sub)))
out=args.results.parent/'analysis';out.mkdir(exist_ok=True)
result=dict(summary=summaries,checkpoints=checkpoints,source_hashes_match=True,
            all_original_trajectory_parameter_gaps_zero=all(a['original_parameter_gap']==0 for a in data['audits']),
            scope='support-only directions, not held-out quality; Adam64 prefix duplicates Adam240; event counts not independent tasks')
(out/'alignment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for name in ['orthogonal','guard','adam240']:
    group=[r for r in checkpoints if r['method']==name];xx=[r['iteration'] for r in group]
    axes[0].plot(xx,[r['full_worse_fraction'] for r in group],marker='o',label=name)
    axes[1].plot(xx,[r['rescue_fraction_of_worse'] if r['full_step_worse'] else np.nan for r in group],marker='o',label=name)
    for r in group:
        if 0<r['full_step_worse']<=10:
            axes[1].annotate(f"n={r['full_step_worse']}",(r['iteration'],r['rescue_fraction_of_worse']),xytext=(3,8 if r['rescue_fraction_of_worse']<.1 else -13),textcoords='offset points',fontsize=8)
for ax in axes:ax.set(xscale='log',xlabel='Actual update number',ylim=(-.03,1.03));ax.grid(alpha=.2);ax.legend()
axes[0].set(ylabel='Fraction of active proposals',title='Full parameter proposal increases true write loss')
axes[1].set(ylabel='Fraction of increasing full proposals',title='A predeclared shorter step decreases old loss')
fig.suptitle('Support-only diagnostic, 12 tasks x 16 matched starts; no query data')
fig.savefig(out/'forward_alignment.png',dpi=170);plt.close(fig)
print(json.dumps(result),flush=True)

import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
proto=json.loads((args.results/'protocol.json').read_text());data=json.loads((args.results/'trace.json').read_text());rows=data['rows'];events=data['events']
assert len(rows)==4*16*(480+240) and len(events)==4*4*3*16
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
def event_stats(group):
    useful=[r for r in group if r['conditional_energy_gain']>1e-10]
    return {'events':len(group),'strict_local_improvement':len(useful),
        'forward_loss_worse_than_coordinate':sum(r['joint_band_loss']>r['coordinate_band_loss']+1e-12 for r in useful),
        'forward_loss_better_than_coordinate':sum(r['joint_band_loss']<r['coordinate_band_loss']-1e-12 for r in useful),
        'forward_loss_worse_than_before':sum(r['joint_band_loss']>r['before_band_loss']+1e-12 for r in useful),
        'mean_forward_loss_delta_vs_coordinate':float(np.mean([r['joint_band_loss']-r['coordinate_band_loss'] for r in useful])) if useful else None,
        'sum_conditional_energy_gain':float(sum(r['conditional_energy_gain'] for r in group))}
checkpoints=[]
for name in ['split','joint']:
    for iteration in [0,1,4,19,59,60,119,120,179,180,239,479]:
        group=[r for r in rows if r['method']==name and r['iteration']==iteration]
        if not group:continue
        selected=[min([r for r in group if r['seed']==seed],key=lambda r:r['best_max_support_error']) for seed in proto['seeds']]
        checkpoints.append({'method':name,'iteration':iteration,
            'median_current_band_loss':float(np.median([r['band_loss'] for r in group])),
            'median_best_selected_max_error':float(np.median([r['best_max_support_error'] for r in selected])),
            'median_residual':float(np.median([r['primal_residual_rms'] for r in group])),
            'median_stationarity_over_residual':float(np.median([r['local_stationarity_rms']/max(r['primal_residual_rms'],1e-15) for r in group])),
            'median_forward_activity_gap':float(np.median([r['forward_activity_gap_rms'] for r in group])),
            'median_path_bound':float(np.median([r['path_bound_rms'] for r in group])),
            'mean_affine_layer_rms':np.mean([r['affine_rms_by_layer'] for r in group],axis=0).tolist(),
            'mean_nonlinear_layer_rms':np.mean([r['nonlinear_rms_by_layer'] for r in group],axis=0).tolist()})
summary={'source_hashes_match':True,'equivalence':data['equivalence_checks'],'all_events':event_stats(events),
    'by_layer':{str(j):event_stats([r for r in events if r['layer']==j]) for j in range(3)},
    'by_iteration':{str(j):event_stats([r for r in events if r['iteration']==j]) for j in [0,60,120,180]},'checkpoints':checkpoints,
    'scope':'observed-support-only diagnostics, not new performance comparison'}
out=args.results.parent/'analysis';out.mkdir(exist_ok=True);(out/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for name,color in [('split','#007d8a'),('joint','#ad647d')]:
    iterations=range(480 if name=='split' else 240)
    groups=[[r for r in rows if r['method']==name and r['iteration']==it] for it in iterations]
    values=[np.median([r['band_loss'] for r in g]) for g in groups]
    axes[0].semilogy(np.arange(len(values))+1,values,label=name,color=color)
    for key,style in [('forward_activity_gap_rms','-'),('path_bound_rms','--')]:
        values=[np.median([r[key] for r in g]) for g in groups];axes[1].semilogy(np.arange(len(values))+1,values,style,label=f'{name}: '+('actual gap' if style=='-' else 'bound'),color=color)
    values=[np.median([r['local_stationarity_rms']/max(r['primal_residual_rms'],1e-15) for r in g]) for g in groups]
    axes[2].plot(np.arange(len(values))+1,values,label=name,color=color)
for ax in axes:ax.set_xlabel('Local sweeps');ax.grid(alpha=.15);ax.legend(fontsize=8)
axes[0].set(title='Median over all 64 starts',ylabel='Actual forward context band loss')
axes[1].set(title='Free activities are not predictions',ylabel='Forward / activity consistency')
axes[2].set(title='Primal accuracy before dual update',ylabel='Stationarity diagnostic / residual RMS')
fig.savefig(out/'forward_consistency.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'}),flush=True)
print(json.dumps({'selected_checkpoints':[r for r in checkpoints if r['iteration'] in [0,19,59,119,239,479]]}),flush=True)

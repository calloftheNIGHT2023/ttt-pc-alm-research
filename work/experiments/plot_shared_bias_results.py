import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
rows=json.loads((args.results/'episodes.json').read_text());out=args.results.parent/(args.results.name+'_analysis')
summary=json.loads((out/'analysis.json').read_text())['summary']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for row in summary:
    ax.scatter(row['median_total_stream_seconds'],row['trajectory_mse'],s=65,label=row['method'])
ax.set(xscale='log',yscale='log',xlabel='Full online fit + query read seconds (log)',ylabel='Mean unseen-query MSE (log)',
    title='Exact joint block: conditional progress vs complete-task cost')
ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8);ax.grid(alpha=.15);fig.savefig(out/'joint_cost_frontier.png',dpi=160);plt.close(fig)
diagnostics={}
for name in ['joint240','joint240_no_dual']:
    group=[r for r in rows if r['method']==name]
    diagnostics[name]={'exact_block_calls':sum(r['exact_block_calls'] for r in group),
        'conditional_gain_total':sum(r['sum_conditional_gain_over_one_coordinate_pass'] for r in group),
        'max_intervals':max(r['max_sum_envelope_intervals'] for r in group)}
seeds=sorted({r['seed'] for r in rows});main={seed:np.mean([r['query_mse'] for r in rows if r['seed']==seed and r['method']=='joint240']) for seed in seeds}
comparisons={}
for name in ['local240','coordinate100_240','local480','batch_adam240_010']:
    diff=[float(main[seed]-np.mean([r['query_mse'] for r in rows if r['seed']==seed and r['method']==name])) for seed in seeds]
    comparisons[name]={'seed_order':seeds,'paired_trajectory_mse_differences':diff,'joint_better_streams':sum(d<0 for d in diff)}
result={'conditional_audit':diagnostics,'paired_comparisons':comparisons,'independent_streams':4,'scope':'development, no significance claim'}
(out/'joint_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

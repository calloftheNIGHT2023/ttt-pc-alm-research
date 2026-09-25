"""First support-rescuable trace in fixed data order; illustration, not effect estimate."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import forward_line_solver as line
root=Path(__file__).resolve().parents[2];base=line.base;base.BOUND=.2
data=json.loads((root/'results/forward_alignment/trace/trace.json').read_text())
row=next(r for r in data['rows'] if r['method']=='orthogonal' and r['line_losses'][-1]>r['line_losses'][0]+1e-12 and min(r['line_losses'][1:-1])<r['line_losses'][0]-1e-12)
seed=row['seed'];rng=np.random.default_rng(seed);truth=rng.uniform(-.2,.2,(3,8));x=rng.uniform(-1,1,(24,8))[:8]
weights=base.family.make_weights(3,8);v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
old=np.array(row['old_parameters'])[None];new=np.array(row['proposed_parameters'])[None]
answer,alpha,meta=line.solve(old,new,x,v,weights);grid=np.linspace(0,1,1001)
curve=line.values(old+grid[:,None,None]*(new-old),x,v,weights);chosen=line.values(answer,x,v,weights)[0]
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
ax.plot(grid,curve,label='True support write loss along parameter segment',lw=2)
predeclared=np.array([0,.125,.25,.5,1.]);losses=line.values(old+predeclared[:,None,None]*(new-old),x,v,weights)
ax.scatter(predeclared,losses,label='Predeclared cheap grid',s=45,zorder=3)
ax.scatter(alpha[0],chosen,marker='*',s=180,label='Piecewise analytic minimum',zorder=4,color='#d35400')
ax.axhline(losses[0],ls='--',color='gray',alpha=.7,label='Old loss')
ax.set(xlabel='Accepted fraction alpha of the proposed parameter change',ylabel='Observed noise-band loss',
       title=f"Support-only illustration: seed {seed}, update {row['iteration']}, start {row['restart']}")
ax.legend(fontsize=8);ax.grid(alpha=.15)
out=root/'results/forward_alignment/analysis';fig.savefig(out/'exact_line_example.png',dpi=170);plt.close(fig)
(out/'line_example.json').write_text(json.dumps(dict(seed=seed,iteration=row['iteration'],restart=row['restart'],
    selection='first orthogonal full-step-increase/short-step-decrease event in frozen trace order; not query-selected',
    old_loss=float(losses[0]),full_loss=float(losses[-1]),analytic_alpha=float(alpha[0]),analytic_loss=float(chosen),grid_losses=losses.tolist(),**meta),indent=2),encoding='utf-8')
print(json.dumps(dict(seed=seed,iteration=row['iteration'],restart=row['restart'],old_loss=float(losses[0]),full_loss=float(losses[-1]),analytic_alpha=float(alpha[0]),analytic_loss=float(chosen))),flush=True)

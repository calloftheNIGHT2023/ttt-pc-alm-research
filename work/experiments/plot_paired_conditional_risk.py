"""Static evaluator figure; no runtime dependency or new statistical choices."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/paired_conditional_risk/diagnostic'); data=json.loads((root/'summary.json').read_text())['summary']
labels=['Local K24 - no credit','Local K24 - Adam16 BP','Local K24 - Adam60 BP','Local K24 - full local',
        'BP + local - BP credit','Full local - full Adam16','Full local - Adam60 H2','Full local - F4096 H2']
y=np.arange(len(data)); fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
for j,r in enumerate(data):
    lo,hi=r['descriptive_task_bootstrap_ci95']; m=r['conditional_mean_delta']
    axes[0].plot([lo,hi],[j,j],color='#167e79',lw=2); axes[0].plot(m,j,'o',color='#167e79')
    axes[0].plot([m-1.96*r['conditional_mean_mc_se'],m+1.96*r['conditional_mean_mc_se']],[j,j],color='#202a32',lw=5)
    axes[1].plot(r['conditional_mean_delta'],j-.15,'o',color='#167e79'); axes[1].plot(r['actual_teacher_mean_delta'],j,'s',color='#d79434'); axes[1].plot(r['empirical_first_query_mean_delta'],j+.15,'x',color='#684b88')
for ax in axes:
    ax.axvline(0,color='#55626a',lw=.8); ax.grid(axis='x',alpha=.2); ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0)); ax.set_xlabel('Candidate minus comparator risk; negative favors candidate')
axes[0].set_yticks(y,labels,fontsize=8); axes[0].invert_yaxis(); axes[0].set_title('Conditional risk: task CI (green) and MC CI (black)',fontsize=10)
axes[1].set_title('Conditional / realized teacher / 2048 queries',fontsize=10)
axes[1].plot([],[],'o',color='#167e79',label='Conditional posterior'); axes[1].plot([],[],'s',color='#d79434',label='Realized teacher, exact input integral'); axes[1].plot([],[],'x',color='#684b88',label='Empirical 2048 queries'); axes[1].legend(loc='lower left',fontsize=7)
fig.suptitle('First-write paired risk decomposition: 16 old contexts, fixed saved predictors')
fig.tight_layout(); fig.savefig(root/'paired_risk_decomposition.png',dpi=180); plt.close(fig)

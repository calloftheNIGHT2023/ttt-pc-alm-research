"""Summary plots; full tables remain linked, no baseline deleted."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("outputs/ttt-pc-alm-research/results/contextual_proposals")
rows=json.loads((root/"development/episodes.json").read_text(encoding="utf-8"))
print(sorted({r["method"] for r in rows}))
credit=json.loads((root/"credit_budget/episodes.json").read_text(encoding="utf-8"))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
for prefix,label,color in [("alm","With duals","#087f8c"),("no_dual","Without duals","#a05c80")]:
    budgets=[1,5,10,20]; values=[]; feasible=[]
    for b in budgets:
        group=[r for r in credit if r["method"]==prefix+str(b)]; assert len(group)==64
        values.append(np.mean([r["query_mse"] for r in group])); feasible.append(sum(r["support_feasible"] for r in group))
    axes[0].plot(budgets,values,"o-",color=color,label=label)
    axes[1].plot(budgets,feasible,"o-",color=color,label=label)
axes[0].set(xlabel="Local sweeps",ylabel="Mean unseen-query MSE",title="Same 4096-task prior proposals")
axes[1].set(xlabel="Local sweeps",ylabel="Feasible writes / 64",title="Finite-budget constraint restoration",ylim=(48,65))
axes[0].legend(); axes[1].legend(); fig.suptitle("16 observed development streams: same parameters, seeds, geometry")
fig.savefig(root/"credit_budget_analysis/credit_budget.png",dpi=160)

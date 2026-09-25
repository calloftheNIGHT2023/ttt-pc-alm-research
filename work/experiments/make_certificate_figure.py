import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("outputs/ttt-pc-alm-research/results/certified_branch_search")
a=json.loads((root/"certificate_audit/regions.json").read_text(encoding="utf-8"))
b=json.loads((root/"certificate_audit_v2/regions.json").read_text(encoding="utf-8"))
a=[r for r in a if r["lp_status"]==2]; b=[r for r in b if r["lp_status"]==2]
assert [(r["seed"],r["n_context"]) for r in a]==[(r["seed"],r["n_context"]) for r in b]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={"width_ratios":[1.2,1]})
idx=np.arange(len(a)); axes[0].bar(idx-.19,[r["polish_steps"] for r in a],width=.38,color="#aab9ca",label="Raw dual only")
axes[0].bar(idx+.19,[r["polish_steps"] for r in b],width=.38,color="#1665a5",label="Raw + dual displacement")
axes[0].axhline(8000,color="#a34522",ls="--",label="Original regional budget")
axes[0].set(xticks=idx,xticklabels=[f"{r['seed']-5700000}:{r['n_context']}" for r in a],xlabel="Development seed offset : context count",
            ylabel="PDHG steps before certificate / budget",ylim=(0,9000),title="A. Nine independently audited infeasible regions")
axes[0].legend(fontsize=8,loc="upper right")
axes[1].axis("off")
boxes=[(.83,"Local adjacent-layer multipliers","No global chain derivative"),
       (.55,r"$d(\lambda)=\min_{z\in D}\lambda^T(Kz-c)>0$","Exact rational check before pruning"),
       (.27,"Skip certified-impossible region","Spend remaining budget on another branch")]
for y,title,subtitle in boxes:
    axes[1].text(.5,y,title+"\n"+subtitle,ha="center",va="center",transform=axes[1].transAxes,
                 bbox={"boxstyle":"round,pad=.8","fc":"#edf4fb","ec":"#89abc8"},fontsize=10)
for y in [.73,.45]: axes[1].annotate("",xy=(.5,y-.08),xytext=(.5,y),xycoords="axes fraction",arrowprops={"arrowstyle":"->","color":"#356886","lw":2})
axes[1].text(.5,.04,"9 / 9 detected; 0 / 119 feasible regions pruned\nDevelopment replay, not fresh task confirmation",
             ha="center",transform=axes[1].transAxes,fontsize=9,color="#425365")
fig.suptitle("Local infeasibility certificates: a computation-allocation signal",fontsize=14)
fig.tight_layout(rect=(0,0,1,.94)); out=root/"development_analysis"; out.mkdir(exist_ok=True)
fig.savefig(out/"certificate_mechanism.png",dpi=170)
print(out/"certificate_mechanism.png")

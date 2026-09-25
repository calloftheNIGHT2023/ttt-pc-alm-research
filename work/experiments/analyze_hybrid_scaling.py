"""Complete development matrix, with no best-depth or best-episode filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
methods=[c["name"] for c in protocol["configs"]]; depths=protocol["depths"]; stages=protocol["stages"]
assert len(rows)==len(methods)*len(depths)*len(stages)*protocol["count"]
hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==s for f,s in protocol["source_sha256"].items()}
assert all(hashes.values())
table="|深度|方法|8/24/64查询MSE|8/24/64可行数|全流中位适应秒|64点时读取2048查询秒|\n|---:|---|---|---|---:|---:|\n"
summary=[]
for d in depths:
    for m in methods:
        group=[r for r in rows if r["depth"]==d and r["method"]==m]
        mse=[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in stages]
        feas=[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in stages]
        med=float(np.median([sum(r["adaptation_seconds"] for r in group if r["seed"]==s) for s in range(protocol["seed0"],protocol["seed0"]+protocol["count"])]))
        read=float(np.median([r["read_2048_seconds"] for r in group if r["n_context"]==stages[-1]]))
        summary.append({"depth":d,"method":m,"stage_mse":mse,"stage_feasible":feas,"stream_median_adaptation_seconds":med,"final_stage_read_seconds":read})
        table+=f"|{d}|{m}|"+" / ".join(f"{v:.6g}" for v in mse)+f"|{feas}|{med:.4f}|{read:.6g}|\n"
audits=[]
for d in depths:
    group=[r for r in rows if r["depth"]==d and r["method"]==methods[0] and r["n_context"]==stages[-1]]
    audits.append({"depth":d,"mean_target_variance":float(np.mean([r["target_query_variance"] for r in group])),
                   "mean_initial_jacobian_zero_fraction":float(np.mean([r["common_initial_support_jacobian_zero_fraction"] for r in group]))})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(parents=True,exist_ok=True)
(out/"table.md").write_text(table,encoding="utf-8")
(out/"analysis.json").write_text(json.dumps({"phase":protocol["phase"],"count_per_depth":protocol["count"],"hashes_match":hashes,"summary":summary,"context_audits":audits},indent=2),encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for m in methods:
    rr=[r for r in summary if r["method"]==m]
    axes[0].semilogy(depths,[r["stage_mse"][-1] for r in rr],"o-",label=m)
    axes[1].plot(depths,[r["stage_feasible"][-1]/protocol["count"] for r in rr],"o-",label=m)
axes[0].set(xlabel="Memory depth",ylabel="Final unseen-query MSE",xticks=depths,title="After 64 observed contexts")
axes[1].set(xlabel="Memory depth",ylabel="True-forward feasible write fraction",xticks=depths,ylim=(-.05,1.1),title="Same feasible-set objective")
axes[0].legend(fontsize=7); fig.suptitle(f"Development only: {protocol['count']} streams per depth; all configurations")
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(out/"scaling.png",dpi=160)
print(json.dumps({"output":str(out),"context_audits":audits}))

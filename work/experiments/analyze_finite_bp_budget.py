"""Full-cost BP stopping curves, descriptive paired stream comparisons."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
proto=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
assert len(rows)==proto["count"]*len(proto["configs"])*len(proto["stages"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto["source_sha256"].items())
seeds=list(range(proto["seed0"],proto["seed0"]+proto["count"])); losses={}; costs={}; summary=[]
for cfg in proto["configs"]:
    name=cfg["name"]; group=[r for r in rows if r["method"]==name]
    losses[name]=np.array([np.mean([r["query_mse"] for r in group if r["seed"]==seed]) for seed in seeds])
    costs[name]=np.array([sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in group if r["seed"]==seed) for seed in seeds])
    summary.append({"method":name,"trajectory_mse":float(losses[name].mean()),"median_total_stream_seconds":float(np.median(costs[name])),
        "all_stage_feasible":sum(all(r["support_feasible"] for r in group if r["seed"]==seed) for seed in seeds),
        "derivative_evaluations":sum(r.get("function_derivative_evaluations",0) for r in group)})
indices=np.random.default_rng(918643).integers(len(seeds),size=(20000,len(seeds))); comparisons=[]
for cfg in proto["configs"]:
    name=cfg["name"]
    if name=="alm20":continue
    diff=losses["alm20"]-losses[name]; dt=costs["alm20"]-costs[name]
    comparisons.append({"a":"alm20","b":name,"mean_mse_difference":float(diff.mean()),"mse_descriptive95ci":np.quantile(diff[indices].mean(1),[.025,.975]).tolist(),
        "mean_cost_difference":float(dt.mean()),"cost_descriptive95ci":np.quantile(dt[indices].mean(1),[.025,.975]).tolist()})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
(out/"budget_costs.json").write_text(json.dumps({"summary":summary,"comparisons":comparisons,"source_hashes_match":True,"scope":"descriptive development; not multiplicity adjusted"},indent=2),encoding="utf-8")
table=["|method|MSE|median full cost seconds|all stages feasible/16|","|---|---:|---:|---:|"]
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.8f}|{r["median_total_stream_seconds"]:.6f}|{r["all_stage_feasible"]}|')
(out/"budget_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
for prefix,color in [("trf","#b67826"),("lbfgs","#746bb0")]:
    group=[next(r for r in summary if r["method"]==prefix+str(b)) for b in [5,20,60,300]]
    ax.plot([r["median_total_stream_seconds"] for r in group],[r["trajectory_mse"] for r in group],"o-",color=color,label=prefix)
    offsets={"trf60":(-8,18,"right"),"trf300":(8,-18,"left"),"lbfgs20":(-5,18,"right"),
        "lbfgs60":(-6,-16,"right"),"lbfgs300":(0,-32,"right")}
    for r in group:
        dx,dy,ha=offsets.get(r["method"],(3,5,"left"))
        ax.annotate(r["method"],(r["median_total_stream_seconds"],r["trajectory_mse"]),xytext=(dx,dy),textcoords="offset points",fontsize=8,ha=ha)
for name,color,marker in [("alm20","#087f8c","*"),("no_dual20","#a05c80","s")]:
    r=next(r for r in summary if r["method"]==name); ax.scatter(r["median_total_stream_seconds"],r["trajectory_mse"],s=120,color=color,marker=marker,label=name)
ax.set(xlabel="Full fit + read seconds",ylabel="Unseen-query MSE",title="Finite BP budgets: same 256-task proposals / 64 starts")
ax.legend(); ax.grid(alpha=.12); fig.savefig(out/"finite_bp_budget.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary,"comparisons":comparisons}),flush=True)

"""Batched fairness controls: complete cost, paired uncertainty, all methods."""
import argparse,hashlib,json
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
        "stage_mse":[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in proto["stages"]],
        "mean_write_discovery_seconds":float(np.mean([r["discovery_seconds"] for r in group])),
        "mean_write_geometry_seconds":float(np.mean([r["geometry_seconds"] for r in group])),
        "total_retained_positive_regions":sum(r["positive_volume_regions"] for r in group)})
indices=np.random.default_rng(419781).integers(len(seeds),size=(20000,len(seeds))); comparisons=[]
for cfg in proto["configs"]:
    name=cfg["name"]
    if name=="alm20":continue
    diff=losses["alm20"]-losses[name]; dt=costs["alm20"]-costs[name]
    comparisons.append({"a":"alm20","b":name,"mean_mse_difference":float(diff.mean()),"mse_descriptive95ci":np.quantile(diff[indices].mean(1),[.025,.975]).tolist(),
        "mean_cost_difference":float(dt.mean()),"cost_descriptive95ci":np.quantile(dt[indices].mean(1),[.025,.975]).tolist()})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
(out/"batched_costs.json").write_text(json.dumps({"summary":summary,"comparisons":comparisons,"source_hashes_match":True,"scope":"descriptive development; not multiplicity adjusted"},indent=2),encoding="utf-8")
table=["|method|MSE|median full cost seconds|all stages feasible/16|mean discovery / write seconds|mean geometry / write seconds|","|---|---:|---:|---:|---:|---:|"]
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.8f}|{r["median_total_stream_seconds"]:.6f}|{r["all_stage_feasible"]}|{r["mean_write_discovery_seconds"]:.6f}|{r["mean_write_geometry_seconds"]:.6f}|')
(out/"batched_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,ax=plt.subplots(figsize=(9,5.5),layout="constrained")
for r in summary:
    color="#087f8c" if r["method"]=="alm20" else ("#a05c80" if r["method"]=="no_dual20" else "#d58b2f" if "gn" in r["method"] else "#7269ac" if "adam" in r["method"] else "#778d9a")
    ax.scatter(r["median_total_stream_seconds"],r["trajectory_mse"],s=70 if r["method"]=="alm20" else 35,color=color,label=r["method"])
ax.set(xlabel="Full online fit + read seconds",ylabel="Unseen-query MSE",title="Batched BP: same 256-task library / 64 initial candidates")
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8); ax.grid(alpha=.12)
fig.savefig(out/"batched_bp.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary,"comparisons":comparisons}),flush=True)

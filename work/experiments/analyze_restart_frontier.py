"""All resource curves and stages retained; empirical Pareto is not a theorem."""
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
seeds=list(range(proto["seed0"],proto["seed0"]+proto["count"])); summary=[]; losses={}; costs={}
for cfg in proto["configs"]:
    name=cfg["name"]; group=[r for r in rows if r["method"]==name]
    losses[name]=np.array([[next(r["query_mse"] for r in group if r["seed"]==seed and r["n_context"]==n) for n in proto["stages"]] for seed in seeds])
    costs[name]=np.array([sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in group if r["seed"]==seed) for seed in seeds])
    summary.append({"method":name,"restarts":cfg["restarts"],"trajectory_mse":float(losses[name].mean()),
        "stage_mse":losses[name].mean(0).tolist(),"stage_feasible":[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in proto["stages"]],
        "all_stage_feasible":sum(all(r["support_feasible"] for r in group if r["seed"]==seed) for seed in seeds),
        "median_total_stream_seconds":float(np.median(costs[name])),"mean_write_discovery_seconds":float(np.mean([r["discovery_seconds"] for r in group])),
        "total_positive_regions":sum(r["positive_volume_regions"] for r in group)})
indices=np.random.default_rng(146977).integers(len(seeds),size=(20000,len(seeds))); contrasts=[]
for r in [1,4,16,64]:
    for competitor in ["no_dual20","adam60","adam240","gn20"]:
        a=f"alm20_r{r}"; b=f"{competitor}_r{r}"; diff=losses[a].mean(1)-losses[b].mean(1); dt=costs[a]-costs[b]
        contrasts.append({"a":a,"b":b,"mean_mse_difference":float(diff.mean()),"mse_descriptive95ci":np.quantile(diff[indices].mean(1),[.025,.975]).tolist(),
            "mean_cost_difference":float(dt.mean()),"cost_descriptive95ci":np.quantile(dt[indices].mean(1),[.025,.975]).tolist()})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
(out/"frontier.json").write_text(json.dumps({"summary":summary,"contrasts":contrasts,"source_hashes_match":True,
    "scope":"observed development, exploratory comparisons; stage and resource selection not independently confirmed"},indent=2),encoding="utf-8")
table=["|method|MSE|4/8/16/24 MSE|4/8/16/24 feasible|median full stream seconds|","|---|---:|---|---|---:|"]
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.8f}|'+" / ".join(f'{a:.6g}' for a in r["stage_mse"])+f'|{r["stage_feasible"]}|{r["median_total_stream_seconds"]:.5f}|')
(out/"frontier_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
colors={"alm20":"#087f8c","no_dual20":"#a05c80","adam60":"#746bb0","adam240":"#458658","gn20":"#c28a29"}
for method,color in colors.items():
    selected=[next(r for r in summary if r["method"]==f"{method}_r{count}") for count in [1,4,16,64]]
    axes[0].plot([r["median_total_stream_seconds"] for r in selected],[r["trajectory_mse"] for r in selected],"o-",color=color,label=method)
    axes[1].semilogx([1,4,16,64],[r["stage_mse"][-1] for r in selected],"o-",color=color,label=method,base=2)
axes[0].set(xlabel="Full fit + read seconds",ylabel="Mean trajectory MSE",title="Full resource curves, not one matched restart count")
axes[1].set(xlabel="Independent starts",ylabel="Final 24-context query MSE",title="Later-stage behavior retained separately",xticks=[1,4,16,64])
axes[1].set_xticklabels([1,4,16,64]); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
fig.suptitle("16 observed development streams; same frozen optimizers and downstream code")
fig.savefig(out/"restart_frontier.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary,"contrasts":contrasts}),flush=True)

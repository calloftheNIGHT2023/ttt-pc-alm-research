"""Finite predeclared budget contrasts; exploratory, no confirmation claims."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
proto=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
assert len(rows)==proto["count"]*len(proto["stages"])*len(proto["configs"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto["source_sha256"].items())
seeds=list(range(proto["seed0"],proto["seed0"]+proto["count"])); rng=np.random.default_rng(217594)
indices=rng.integers(len(seeds),size=(20000,len(seeds))); summary=[]; losses={}; costs={}
for cfg in proto["configs"]:
    name=cfg["name"]; group=[r for r in rows if r["method"]==name]
    losses[name]=np.array([np.mean([r["query_mse"] for r in group if r["seed"]==seed]) for seed in seeds])
    costs[name]=np.array([sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in group if r["seed"]==seed) for seed in seeds])
    summary.append({"method":name,"trajectory_mse":float(losses[name].mean()),"median_total_stream_seconds":float(np.median(costs[name])),
        "all_stage_feasible":sum(all(r["support_feasible"] for r in group if r["seed"]==seed) for seed in seeds),
        "prior_parameter_bank_bytes":max(r.get("proposal_bank_parameter_bytes",0) for r in group),
        "max_support_feature_array_bytes":max(r.get("proposal_support_array_bytes",0) for r in group),
        "prior_storage_note":"Regenerated at writes; streamed generation could reduce peak memory. Not an unavoidable persistent-state lower bound.",
        "sum_candidates":sum(r["candidates"] for r in group),"sum_positive_modes":sum(r["positive_volume_regions"] for r in group)})
pairs=[(f"bank256_alm{b}",f"bank256_no_dual{b}") for b in [20,60,120]]
pairs+=[("bank4096_alm20","bank4096_no_dual20"),("random64_alm120","random64_no_dual120")]
pairs+=[(f"bank256_alm{b}",ref) for b in [5,20,60,120] for ref in ["bank256_direct128","bank256_trf64","bank256_lbfgs64","bank65536_direct64","random128_lbfgs_priorbox"]]
contrasts=[]
for a,b in pairs:
    dl=losses[a]-losses[b]; dt=costs[a]-costs[b]
    contrasts.append({"a":a,"b":b,"mse_difference":float(dl.mean()),"mse_descriptive_95ci":np.quantile(dl[indices].mean(1),[.025,.975]).tolist(),
        "mean_total_cost_difference_seconds":float(dt.mean()),"cost_descriptive_95ci":np.quantile(dt[indices].mean(1),[.025,.975]).tolist(),
        "scope":"exploratory paired development contrasts, not multiplicity-adjusted"})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
(out/"budget_mechanism.json").write_text(json.dumps({"summary":summary,"contrasts":contrasts,"source_hashes_match":True},indent=2),encoding="utf-8")
table=["|method|MSE|median full cost s|all stages feasible/16|prior parameter bytes (not peak)|","|---|---:|---:|---:|---:|"]
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.8f}|{r["median_total_stream_seconds"]:.5f}|{r["all_stage_feasible"]}|{r["prior_parameter_bank_bytes"]}|')
(out/"budget_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
for prefix,label,color in [("alm","With duals","#087f8c"),("no_dual","Without duals","#a05c80")]:
    budgets=[20,60,120]; selected=[next(r for r in summary if r["method"]=="bank256_"+prefix+str(b)) for b in budgets]
    axes[0].plot(budgets,[r["trajectory_mse"] for r in selected],"o-",label=label,color=color)
axes[0].set(xlabel="Local sweeps",ylabel="Unseen-query MSE",title="256-task proposals: isolated multiplier effect"); axes[0].legend()
names=["bank256_alm20","bank256_alm60","bank256_alm120","bank256_no_dual120","bank256_trf64","bank256_lbfgs64","bank4096_alm20","bank65536_direct64","random128_lbfgs_priorbox"]
for name in names:
    row=next(r for r in summary if r["method"]==name)
    axes[1].scatter(row["median_total_stream_seconds"],row["trajectory_mse"],label=name,s=40)
axes[1].set(xlabel="Full online fit + read seconds",ylabel="Unseen-query MSE",title="Matched common screening + adaptive readout")
axes[1].legend(fontsize=7,loc="upper right"); fig.suptitle("16 observed development streams; full 20-method table retained")
fig.savefig(out/"prior_budget.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary,"credit_contrasts":contrasts[:5]}),flush=True)

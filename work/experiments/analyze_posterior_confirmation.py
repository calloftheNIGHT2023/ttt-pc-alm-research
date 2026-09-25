"""Predeclared paired confirmation gate; no selection of favorable stages."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
methods=[c["name"] for c in protocol["configs"]]; seeds=list(range(protocol["seed0"],protocol["seed0"]+protocol["count"]))
stages=protocol["stages"]; candidate=protocol["primary_candidate"]
assert len(rows)==len(methods)*len(seeds)*len(stages)
assert len({(r["method"],r["seed"],r["n_context"]) for r in rows})==len(rows)
hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==s for f,s in protocol["source_sha256"].items()}
assert all(hashes.values())
losses={}; costs={}; summary=[]; reasons={}
for m in methods:
    group=[r for r in rows if r["method"]==m]
    losses[m]=np.array([np.mean([r["query_mse"] for r in group if r["seed"]==s]) for s in seeds])
    costs[m]=np.array([sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in group if r["seed"]==s) for s in seeds])
    summary.append({"method":m,"trajectory_mse":float(losses[m].mean()),
        "stage_mse":[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in stages],
        "stage_feasible":[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in stages],
        "all_stage_feasible":sum(all(r["support_feasible"] for r in group if r["seed"]==s) for s in seeds),
        "median_total_seconds":float(np.median(costs[m])),"mean_total_seconds":float(costs[m].mean()),
        "median_adaptation_seconds":float(np.median([sum(r["adaptation_seconds"] for r in group if r["seed"]==s) for s in seeds])),
        "median_query_read_seconds":float(np.median([sum(r["read_2048_seconds"] for r in group if r["seed"]==s) for s in seeds])),
        "max_reported_persistent_state_bytes":max((r.get("persistent_state_bytes",8*r.get("persistent_scalars",0)) for r in group),default=0)})
    reasons[m]=dict(Counter(t["reason"] for r in group for t in r.get("geometry_trace",[])))
rng=np.random.default_rng(944914); index=rng.integers(len(seeds),size=(100000,len(seeds)))
alpha=protocol["primary_family_error"]/len(protocol["primary_quality_comparators"])
comparisons=[]
for m in methods:
    if m==candidate: continue
    diff=losses[candidate]-losses[m]; boot=diff[index].mean(axis=1)
    cdelta=costs[candidate]-costs[m]; cboot=cdelta[index].mean(axis=1)
    primary=m in protocol["primary_quality_comparators"]
    comparisons.append({"candidate":candidate,"baseline":m,"primary":primary,"mean_mse_difference":float(diff.mean()),
        "paired_two_sided_95_ci":np.quantile(boot,[.025,.975]).tolist(),
        "primary_bonferroni_upper":float(np.quantile(boot,1-alpha)) if primary else None,
        "primary_pass":bool(np.quantile(boot,1-alpha)<0) if primary else None,
        "candidate_lower_mse_streams":int(np.sum(diff<0)),"tied_streams":int(np.sum(diff==0)),
        "mean_total_seconds_difference":float(cdelta.mean()),"total_seconds_paired_95_ci":np.quantile(cboot,[.025,.975]).tolist()})
passed=all(c["primary_pass"] for c in comparisons if c["primary"])
out=args.results.parent/"confirmation_analysis"; out.mkdir(parents=True,exist_ok=True)
result={"phase":protocol["phase"],"streams":len(seeds),"hashes_match":hashes,"primary_quality_gate_passed":passed,
    "bootstrap_replicates":len(index),"primary_one_sided_confidence":1-alpha,"summary":summary,"comparisons":comparisons,"geometry_reasons":reasons,
    "boundary":"Bootstrap approximate coverage; no equivalence from non-significance; compare all larger-budget controls and full resource audit."}
(out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
table="|方法|全程MSE|4/8/16/24查询MSE|各阶段可行数|全阶段可行流|中位适应秒|中位查询秒|中位总秒|\n|---|---:|---|---|---:|---:|---:|---:|\n"
for s in summary:
    table+=f"|{s['method']}|{s['trajectory_mse']:.8f}|"+" / ".join(f"{v:.6g}" for v in s["stage_mse"])+f"|{s['stage_feasible']}|{s['all_stage_feasible']}|{s['median_adaptation_seconds']:.4f}|{s['median_query_read_seconds']:.4f}|{s['median_total_seconds']:.4f}|\n"
(out/"table.md").write_text(table,encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1,1.15]}); colors=plt.get_cmap("tab20")(np.linspace(0,1,len(methods)))
for i,(s,color) in enumerate(zip(summary,colors)):
    axes[0].semilogy(stages,s["stage_mse"],"o-",color=color,linewidth=1.3)
    axes[1].barh(i,s["trajectory_mse"],color=color)
    axes[1].text(s["trajectory_mse"]+max(r["trajectory_mse"] for r in summary)*.01,i,
        f"{s['trajectory_mse']:.5f} | {s['median_total_seconds']:.2f}s",va="center",fontsize=7)
axes[0].set(xlabel="Observed contexts",ylabel="Unseen-query MSE",xticks=stages,title="Same color = same method")
axes[0].grid(alpha=.15)
axes[1].set(yticks=range(len(methods)),yticklabels=methods,xlabel="Trajectory MSE | median fit + 4 x 2048 query seconds",
    xlim=(0,max(r["trajectory_mse"] for r in summary)*1.48))
axes[1].tick_params(axis="y",labelsize=7); axes[1].invert_yaxis()
fig.suptitle(f"Frozen fresh confirmation: {len(seeds)} streams; primary quality gate = {passed}")
fig.tight_layout(); fig.savefig(out/"overview.png",dpi=160)
print(json.dumps({"output":str(out),"primary_quality_gate_passed":passed,"summary":summary,
    "primary_comparisons":[c for c in comparisons if c["primary"]]}),flush=True)

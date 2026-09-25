"""All frozen results and paired stream-level comparisons, no test selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"single_region":"Single region","certified_search":"Certified branch search",
    "timeout_search":"Fixed timeout search","same_bank_global_lp_qp":"Same-bank global LP/QP",
    "restored_slsqp64":"L-BFGS64 + SLSQP","restored_slsqp128":"L-BFGS128 + SLSQP",
    "no_discovery_dual":"No discovery multipliers","linear_ls":"Linear least squares",
    "rbf":"RBF ridge","prior1024":"Matched-prior ridge 1024","shallow64":"Trainable shallow head 64"}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
    rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    methods=[c["name"] for c in protocol["configs"]]; n=protocol["count"]; stages=[4,8,16,24]
    seeds=range(protocol["seed0"],protocol["seed0"]+n)
    index={(r["method"],r["seed"],r["n_context"]):r for r in rows}
    assert len(index)==len(rows)==len(methods)*n*4
    hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==s
            for f,s in protocol["source_sha256"].items()}
    assert all(hashes.values()),hashes
    errors={m:np.array([[index[m,s,k]["query_mse"] for k in stages] for s in seeds]) for m in methods}
    feas={m:np.array([[index[m,s,k]["support_feasible"] for k in stages] for s in seeds]) for m in methods}
    times={m:np.array([[index[m,s,k]["seconds"] for k in stages] for s in seeds]) for m in methods}
    sample=np.random.default_rng(7419).integers(0,n,(20000,n)); comparisons=[]; summary=[]
    for m in methods:
        records=[r for r in rows if r["method"]==m]
        summary.append({"method":m,"trajectory_mean_mse":float(errors[m].mean()),
            "stage_mean_mse":errors[m].mean(0).tolist(),"stage_feasible_counts":feas[m].sum(0).tolist(),
            "all_stages_feasible_streams":int(feas[m].all(1).sum()),
            "median_stream_seconds":float(np.median(times[m].sum(1))),
            "regional_steps_total":sum(r.get("regional_steps",0) for r in records),
            "regions_attempted":sum(r.get("regions_attempted",0) for r in records),
            "regions_certified_infeasible":sum(r.get("regions_certified_infeasible",0) for r in records)})
        if m=="certified_search": continue
        for j in [None,0,1,2,3]:
            diff=errors["certified_search"].mean(1)-errors[m].mean(1) if j is None else errors["certified_search"][:,j]-errors[m][:,j]
            comparisons.append({"against":m,"metric":"trajectory_mean" if j is None else f"n{stages[j]}",
                "mean_paired_difference":float(diff.mean()),"bootstrap95":np.quantile(diff[sample].mean(1),[.025,.975]).tolist(),
                "strict_wins":int((diff<0).sum()),"n":n})
    analysis={"phase":protocol["phase"],"n_independent_streams":n,"source_hashes_match":hashes,
        "summary":summary,"paired_comparisons":comparisons,
        "primary_comparison":"certified_search minus single_region, trajectory_mean",
        "bootstrap_note":"20000 paired episode resamples, seed7419; secondary contrasts descriptive and unadjusted"}
    (args.out/"analysis.json").write_text(json.dumps(analysis,indent=2),encoding="utf-8")
    table="|方法|4点MSE|8点MSE|16点MSE|24点MSE|全程MSE|可行数4/8/16/24|全阶段可行流|全程中位秒|\n|---|---:|---:|---:|---:|---:|---|---:|---:|\n"
    for r in summary:
        table+=f"|{LABELS[r['method']]}|"+"|".join(f"{v:.6g}" for v in r["stage_mean_mse"])
        table+=f"|{r['trajectory_mean_mse']:.6g}|{r['stage_feasible_counts']}|{r['all_stages_feasible_streams']}/{n}|{r['median_stream_seconds']:.3f}|\n"
    (args.out/"table.md").write_text(table,encoding="utf-8")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(16.8,5.8))
    colors={"certified_search":"#1558a6","single_region":"#7496b8","timeout_search":"#6f7480",
        "restored_slsqp128":"#bd570e","same_bank_global_lp_qp":"#16836d","prior1024":"#843fba"}
    for m,c in colors.items():
        if m not in methods: continue
        axes[0].semilogy(stages,errors[m].mean(0),"o-",color=c,label=LABELS[m],markersize=4)
    axes[0].set(xlabel="Observed contexts",ylabel="Mean unseen-query MSE",xticks=stages,title="A. All four read-after-write stages")
    axes[0].legend(fontsize=7,loc="lower left")
    for m in ["single_region","certified_search","timeout_search","restored_slsqp128","no_discovery_dual"]:
        if m in methods: axes[1].plot(stages,feas[m].mean(0),"o-",label=LABELS[m],color=colors.get(m,"#ae3b78"))
    axes[1].set(xlabel="Observed contexts",ylabel="True-forward write feasibility",xticks=stages,ylim=(-.02,1.08),title="B. Not free output activity fit")
    axes[1].legend(fontsize=7,loc="lower right")
    yy=np.arange(len(methods)); axes[2].barh(yy,[r["median_stream_seconds"] for r in summary],color=[colors.get(m,"#b4bfca") for m in methods])
    axes[2].set(yticks=yy,yticklabels=[LABELS[m] for m in methods],xlabel="Median total seconds per stream",xscale="log",title="C. Full adaptation cost")
    axes[2].tick_params(axis="y",labelsize=7); axes[2].invert_yaxis()
    fig.suptitle(f"Certified local memory | {n} streams | {protocol['phase']}",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(args.out/"overview.png",dpi=170); plt.close(fig)
    print(json.dumps({"summary":summary,"trajectory_comparisons":[r for r in comparisons if r["metric"]=="trajectory_mean"]}))


if __name__=="__main__": main()

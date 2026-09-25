"""Frozen stream-level comparisons; all stages and all methods, no test tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS={
    "restore_120x64":"Feasibility-first local ALM",
    "anchored_120x64":"Anchored ALM 120 x 64",
    "previous_600x16":"Previous ALM 600 x 16",
    "no_dual_120x64":"No multipliers",
    "derivative_120x64":"Derivative blocks 120 x 64",
    "derivative_480x64":"Derivative blocks 480 x 64",
    "restored_slsqp64":"L-BFGS64 + SLSQP",
    "restored_slsqp128":"L-BFGS128 + SLSQP",
    "rbf":"RBF ridge",
    "prior1024":"Matched-prior ridge 1024",
    "shallow64":"Trainable shallow head 64",
    "linear_rls":"Linear ridge RLS"}


def algebra_checks():
    # Exact cross-branch threshold, with a finite numerical proximal term.
    c,t,trust,beta=.1,.014,.01,.0125
    def threshold(k): return c*(k+np.sqrt(k*(1+k)))
    def solve(k):
        pos=(c+t)/(1+k)
        return pos if k*(c+t)**2/(1+k)<t*t else 0.
    anchored,restored=solve(trust+beta),solve(trust)
    assert threshold(trust)<t<threshold(trust+beta)
    assert anchored==0 and restored>c
    # Audit RLS against closed-form batch ridge after every streaming stage.
    rng=np.random.default_rng(1842)
    x=rng.uniform(0,1,24); target=rng.uniform(0,1,24)
    phi=np.column_stack([np.ones(24),x]); ridge=1e-4
    coef=np.zeros(2); inv=np.eye(2)/ridge; errors=[]
    for i in range(24):
        gain=inv@phi[i]/(1+phi[i]@inv@phi[i])
        coef+=gain*(target[i]-phi[i]@coef)
        inv-=np.outer(gain,phi[i]@inv)
        if i+1 in [4,8,16,24]:
            closed=np.linalg.solve(phi[:i+1].T@phi[:i+1]+ridge*np.eye(2),phi[:i+1].T@target[:i+1])
            err=float(np.max(np.abs(closed-coef))); errors.append(err); assert err<1e-8
    return {"passed":True,"threshold_without_anchor":threshold(trust),
            "threshold_with_anchor":threshold(trust+beta),"target":t,
            "anchored_selected_b":anchored,"restoration_selected_b":restored,
            "rls_vs_closed_max_error":max(errors)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
    rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    configs=protocol["configs"]; count=protocol["count"]
    assert protocol["phase"]=="confirmation"
    assert len(rows)==len(configs)*count*4
    assert len({(r["method"],r["seed"],r["n_context"]) for r in rows})==len(rows)
    root=Path(__file__).parent
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in protocol["source_sha256"].items()}
    assert all(hashes.values()),hashes
    stages=[4,8,16,24]; seeds=range(protocol["seed0"],protocol["seed0"]+count)
    methods=[cfg["name"] for cfg in configs]
    index={(r["method"],r["seed"],r["n_context"]):r for r in rows}
    errors={m:np.array([[index[m,seed,n]["query_mse"] for n in stages] for seed in seeds]) for m in methods}
    times={m:np.array([[index[m,seed,n]["seconds"] for n in stages] for seed in seeds]) for m in methods}
    feasible={m:np.array([[index[m,seed,n]["support_feasible"] for n in stages] for seed in seeds]) for m in methods}
    rng=np.random.default_rng(8462)
    bootstrap=rng.integers(0,count,(20000,count))
    comparisons=[]
    candidate="restore_120x64"
    for m in methods:
        if m==candidate: continue
        for j in [None,0,1,2,3]:
            v1=errors[candidate].mean(axis=1) if j is None else errors[candidate][:,j]
            v0=errors[m].mean(axis=1) if j is None else errors[m][:,j]
            diff=v1-v0; means=diff[bootstrap].mean(axis=1)
            comparisons.append({"against":m,"metric":"trajectory_mean" if j is None else f"n{stages[j]}",
                                "mean_paired_difference":float(diff.mean()),
                                "bootstrap95":np.quantile(means,[.025,.975]).tolist(),
                                "strict_wins":int(np.sum(diff<0)),"count":count})
    summary=[]
    for m in methods:
        summary.append({"method":m,"trajectory_mean_mse":float(errors[m].mean()),
                        "stage_mean_mse":errors[m].mean(axis=0).tolist(),
                        "stage_feasible_counts":feasible[m].sum(axis=0).tolist(),
                        "all_stages_feasible_streams":int(np.all(feasible[m],axis=1).sum()),
                        "median_stream_seconds":float(np.median(times[m].sum(axis=1))),
                        "stage_median_seconds":np.median(times[m],axis=0).tolist(),
                        "max_search_array_bytes":max((r.get("main_search_arrays_bytes",0) for r in rows if r["method"]==m),default=0)})
    analysis={"n_independent_streams":count,"n_stages_per_stream":4,"source_hashes_match":hashes,
              "algebra_checks":algebra_checks(),"summary":summary,"paired_comparisons":comparisons,
              "bootstrap_note":"20,000 paired episode resamples, seed8462; descriptive intervals, no multiplicity adjustment"}
    (args.out/"analysis.json").write_text(json.dumps(analysis,indent=2),encoding="utf-8")
    table="|方法|4点MSE|8点MSE|16点MSE|24点MSE|四阶段平均|24点可行|全程中位秒|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in summary:
        table+=f"|{LABELS[r['method']]}|"+"|".join(f"{v:.6g}" for v in r["stage_mean_mse"])
        table+=f"|{r['trajectory_mean_mse']:.6g}|{r['stage_feasible_counts'][-1]}/{count}|{r['median_stream_seconds']:.3f}|\n"
    (args.out/"table.md").write_text(table,encoding="utf-8")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    palette={candidate:"#1259aa","previous_600x16":"#7b9cba","anchored_120x64":"#16a085",
             "restored_slsqp64":"#d97706","restored_slsqp128":"#9a3412","prior1024":"#7c3aed",
             "shallow64":"#667085"}
    for m,color in palette.items():
        axes[0].semilogy(stages,errors[m].mean(axis=0),"o-",label=LABELS[m],color=color,markersize=4)
    axes[0].set(xlabel="Cumulative observed contexts",ylabel="Mean independent query MSE",xticks=stages,
                title="A. Frozen configurations, fresh streams")
    axes[0].legend(fontsize=7,loc="lower left")
    plotmethods=[candidate,"previous_600x16","anchored_120x64","no_dual_120x64","derivative_480x64","restored_slsqp64","restored_slsqp128"]
    mechanism_colors={**palette,"no_dual_120x64":"#687076","derivative_480x64":"#b83280"}
    for i,m in enumerate(plotmethods):
        axes[1].plot(stages,feasible[m].mean(axis=0),"o--" if m in ["no_dual_120x64","derivative_480x64"] else "o-",
                     color=mechanism_colors[m],label=LABELS[m],alpha=.9)
    axes[1].set(xlabel="Cumulative observed contexts",ylabel="Fraction of feasible writes",xticks=stages,
                ylim=(-.03,1.07),title="B. True forward support feasibility")
    axes[1].legend(fontsize=7,loc="lower right")
    ypos=np.arange(len(summary))
    axes[2].barh(ypos,[r["median_stream_seconds"] for r in summary],
                 color=[palette.get(r["method"],"#b7bec9") for r in summary])
    axes[2].set(yticks=ypos,yticklabels=[LABELS[r["method"]] for r in summary],xlabel="Median total adaptation seconds / stream",
                title="C. All methods: complete stream cost",xscale="log")
    axes[2].tick_params(axis="y",labelsize=7.5); axes[2].invert_yaxis()
    fig.suptitle(f"Feasibility-first local memory: {count} independent confirmation streams",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(args.out/"confirmation.png",dpi=170); plt.close(fig)
    # A mathematical mechanism sketch with an exact, explicitly local threshold.
    check=analysis["algebra_checks"]
    bs=np.linspace(0,.15,700); c,t=.1,.014
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    for k,label,color in [(.0225,"Anchored discovery", "#b45309"),(.01,"Feasibility discovery", "#1259aa")]:
        energy=(np.maximum(bs-c,0)-t)**2+k*bs**2
        ax.plot(bs,energy,label=label,color=color)
        chosen=check["anchored_selected_b"] if k==.0225 else check["restoration_selected_b"]
        ax.scatter([chosen],[(max(chosen-c,0)-t)**2+k*chosen**2],color=color,zorder=3,s=70)
    ax.axvline(.1,color="gray",linestyle="--",label="Activation boundary")
    ax.set(xlabel="Local fast parameter b",ylabel="Exact local discovery energy",
           title="Removing the anchor can unlock a previously disfavored branch",ylim=(0,.001))
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(args.out/"anchor_threshold.png",dpi=170); plt.close(fig)
    print(json.dumps({"summary":summary,"primary_comparisons":[r for r in comparisons if r["metric"]=="trajectory_mean"]}),flush=True)


if __name__=="__main__": main()

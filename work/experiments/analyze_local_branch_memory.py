"""Audit frozen confirmation and verify a constructive primal-dual branch theorem."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def constrained_write(a,y,rho,steps=12,dual=True,exact=True):
    h=float(a)
    u=0.
    hs=[]
    us=[]
    for _ in range(steps):
        target=y+u
        if exact:
            candidates=[min(a,0.),max(0.,(a+rho*target)/(1+rho))]
            losses=[.5*(p-a)**2+.5*rho*(max(p,0.)-target)**2 for p in candidates]
            h=candidates[int(np.argmin(losses))]
        else:
            grad=h-a+rho*(max(h,0.)-target)*(h>0)
            h-=grad/(1+rho)
        if dual:
            u+=y-max(h,0.)
        hs.append(h)
        us.append(u)
    return np.asarray(hs),np.asarray(us)


def theorem_audit():
    hs,_=constrained_write(-1,1,1)
    assert np.allclose(hs,[-1,-1]+[1]*10)
    pc,_=constrained_write(-1,1,1,dual=False)
    gd,_=constrained_write(-1,1,1,exact=False)
    assert np.all(pc==-1) and np.all(gd==-1)
    rng=np.random.default_rng(313)
    for _ in range(100):
        rho=rng.uniform(.3,3)
        c=rng.uniform(.2,2)
        threshold=c/(np.sqrt(1+rho)-1)
        lower=c/np.sqrt(1+rho)
        y=rng.uniform(lower*1.05,threshold*.95)
        k=int(np.floor(threshold/y))+1
        states,_=constrained_write(-c,y,rho,steps=k+80)
        assert np.all(states[:k-1]<0)
        assert np.all(states[k-1:]>0)
        errors=states[k-1:]-y
        assert np.max(np.abs(errors[1:]-errors[:-1]/(1+rho)))<1e-10
    return {"passed":True,"random_parameter_cases":100,"example_states":hs.tolist(),
            "without_dual_states":pc.tolist(),"without_branch_solve_states":gd.tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    labels=["Local ALM 120x16","Local ALM 240x32","No multipliers","Derivative blocks",
            "Adam 8 restarts","L-BFGS 16 restarts","L-BFGS 64 restarts","Linear LS",
            "RBF ridge","Matched-prior ridge","Shallow 12","Shallow 64"]
    grouped=[[r for r in rows if r["config_index"]==i] for i in range(len(labels))]
    assert all(len(group)==64 for group in grouped)
    errors=np.array([[r["query_mse"] for r in sorted(group,key=lambda r:r["seed"])] for group in grouped])
    timings=np.array([[r["seconds"] for r in group] for group in grouped])
    rng=np.random.default_rng(421)
    bootstrap_ids=rng.integers(0,64,(20000,64))
    comparisons=[]
    for i in [2,3,4,5,6,9,10,11]:
        diff=errors[1]-errors[i]
        ci=np.quantile(diff[bootstrap_ids].mean(axis=1),[.025,.975])
        comparisons.append({"candidate":labels[1],"baseline":labels[i],"mean_paired_difference":float(diff.mean()),
                            "bootstrap_95_percent_ci":ci.tolist(),"win_rate":float(np.mean(diff<0)),
                            "relative_mean_mse_reduction":float(1-errors[1].mean()/errors[i].mean()) if errors[i].mean()>1e-12 else None})
    stats=[{"label":label,"mean_mse":float(errors[i].mean()),"median_mse":float(np.median(errors[i])),
            "p90_mse":float(np.quantile(errors[i],.9)),"adaptation_median_seconds":float(np.median(timings[i]))}
           for i,label in enumerate(labels)]
    theorem=theorem_audit()
    output={"theorem_verification":theorem,"summary":stats,"paired_comparisons":comparisons,
            "timing_scope":"CPU adaptation only, not end-to-end query latency; no exact compute-match claim",
            "confirm_seeds":[5300000,5300063],"interpretation":"Mechanistic confirmation, not PC-ALM/TTT indispensability or downstream validation"}
    (args.out/"analysis.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    colors=["#23785c" if i in [0,1] else "#d37a32" if i in [2,3] else "#416c97" for i in range(len(labels))]
    axes[0].barh(labels,np.maximum(errors.mean(axis=1),1e-14),color=colors)
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean unseen-query MSE (log scale)")
    axes[0].set_title("64 frozen confirmation episodes; all controls shown")
    for i,label in enumerate(labels):
        if i==6:
            axes[0].annotate("3.87e-15",(1e-14,i),xytext=(5,0),textcoords="offset points",va="center",fontsize=8)
    axes[1].scatter(np.median(timings,axis=1),errors.mean(axis=1),c=colors,s=65)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Median CPU adaptation seconds (log scale)")
    axes[1].set_ylabel("Mean query MSE (log scale)")
    axes[1].set_title("Accuracy and adaptation cost are separate")
    for i in [1,2,3,6,9]:
        axes[1].annotate(labels[i],(np.median(timings[i]),errors[i].mean()),xytext=(4,8),textcoords="offset points",fontsize=8,
                         ha="right" if i in [1,2] else "left")
    fig.savefig(args.out/"confirmation_overview.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    timeline=np.arange(1,13)
    for kwargs,label,color,style,marker in [({},"Exact local solve + dual","#23785c","-","o"),({"dual":False},"Exact solve, no dual","#d37a32","--","s"),
                              ({"exact":False},"Dual, derivative steps","#416c97",":","x")]:
        states,duals=constrained_write(-1,1,1,**kwargs)
        axes[0].plot(timeline,states,linestyle=style,marker=marker,label=label,color=color,alpha=.85)
        axes[1].plot(timeline,duals,linestyle=style,marker=marker,label=label,color=color,alpha=.85)
    axes[0].axhline(1,color="gray",linestyle=":")
    axes[0].set(xlabel="Local iteration",ylabel="State h",title="Both components enable a feasible write in 3 steps")
    axes[1].set(xlabel="Local iteration",ylabel="Scaled multiplier u",title="Accumulated residual triggers a branch change")
    axes[0].legend(fontsize=8)
    fig.savefig(args.out/"constructive_write_theorem.png",dpi=180)
    plt.close(fig)
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()

"""Analyze development only; certify branch obstructions; show all baselines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streaming_branch_projection as base
from streaming_projection_refinement import farkas_certificate


def algebra_checks():
    rng=np.random.default_rng(407)
    maxerr=0.
    for _ in range(20):
        old=rng.normal(size=(4,9)); new=rng.normal(size=(3,9)); d=rng.normal(size=3)
        proj=np.eye(9)-np.linalg.pinv(old)@old
        delta=proj@new.T@np.linalg.pinv(new@proj@new.T)@d
        err=max(np.max(np.abs(old@delta)),np.max(np.abs(new@delta-d)))
        maxerr=max(maxerr,float(err)); assert err<1e-10
    # Exact zero reachable subspace; the old-feature span cannot accept a new write.
    old=np.eye(9)[:4]; new=old[:2]; proj=np.eye(9)-np.linalg.pinv(old)@old
    assert np.array_equal(new@proj,np.zeros((2,9)))
    mat=rng.normal(size=(3,7)); anchor=rng.normal(size=7); target=rng.normal(size=3)
    olddual=np.linalg.solve(mat@mat.T,mat@anchor-target)
    b=anchor-mat.T@olddual
    assert np.linalg.norm((b-anchor)+mat.T@olddual)<1e-10
    assert np.linalg.norm(mat@b-target)<1e-10
    stale_force=float(np.linalg.norm(mat.T@olddual))
    assert stale_force>1e-3
    # With the new anchor b, zero multipliers satisfy both stationarity and feasibility.
    assert np.linalg.norm((b-b)+mat.T@np.zeros(3))==0
    return {"passed":True,"fixed_feature_projection_cases":20,"max_constraint_error":maxerr,
            "old_span_obstruction_checked":True,"reanchoring_stale_dual_force_example":stale_force}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    out=args.root/"analysis"; out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.root/"development/episodes.json").read_text(encoding="utf-8"))
    rows+=json.loads((args.root/"refinement_development/episodes.json").read_text(encoding="utf-8"))
    # Known response range is a shared prior: grant clipping to ALL regressors.
    cliprows=[]
    for row in rows:
        if row["method"] not in ["linear","rbf","prior1024"]: continue
        rng=np.random.default_rng(row["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
        q=rng.uniform(0,1,2048); target=base.forward(q,truth)
        n=row["n_context"]
        family={"linear":"linear","rbf":"rbf","prior1024":"prior_bank"}[row["method"]]
        pred,_=base.fit_regression(x[:n],v[:n],4,family=family,features=1024)
        raw=pred(q)
        assert abs(np.mean((raw-target)**2)-row["query_mse"])<1e-8
        row["query_mse_unclipped"]=row["query_mse"]
        row["query_mse"]=float(np.mean((np.clip(raw,0,1)-target)**2))
        row["range_clipped"]=True
        cliprows.append({k:row[k] for k in ["method","seed","n_context","query_mse_unclipped","query_mse"]})
    certificates=[]
    for row in rows:
        if row["method"]!="local_branch_refined": continue
        if row["current_branch_feasible"] is not False: continue
        rng=np.random.default_rng(row["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
        n=row["n_context"]
        cert=farkas_certificate(x[:n],v[:n],np.array(row["anchor"]))
        assert cert["certified"],cert
        certificates.append({"seed":row["seed"],"n_context":n,"solver_found_feasible":row["support_feasible"],**cert})
    labels={"local_warm":"ALM: cached states", "local_reset":"ALM: reset states",
            "local_no_dual":"No multipliers", "local_derivative":"Local derivative blocks",
            "slsqp16":"SLSQP 16", "linear":"Linear LS [clipped]", "rbf":"RBF ridge [clipped]",
            "prior1024":"Prior ridge 1024 [clipped]", "local_branch_refined":"ALM + local convex polish",
            "restored_slsqp64":"L-BFGS64 + SLSQP"}
    summary=[]
    for method in labels:
        for n in [4,8,16,24]:
            selected=[r for r in rows if r["method"]==method and r["n_context"]==n]
            summary.append({"method":method,"n_context":n,"mean_query_mse":float(np.mean([r["query_mse"] for r in selected])),
                            "feasible":sum(r["support_feasible"] for r in selected),"count":len(selected),
                            "median_seconds":float(np.median([r["seconds"] for r in selected])),
                            "old_support_feasible":sum(r.get("old_support_max_error") is not None and r["old_support_max_error"]<=base.EPS+base.TOL for r in selected),
                            "max_support_error":float(max(r["support_max_error"] for r in selected))})
    stats={"phase":"development_not_confirmation","algebra_checks":algebra_checks(),"summary":summary,
           "farkas_certificates":certificates,"certified_cross_branch_writes":len(certificates),
           "certified_obstructions_then_solved":sum(r["solver_found_feasible"] for r in certificates),
           "regression_range_clipping":cliprows}
    (out/"analysis.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    (out/"table.md").write_text("|方法|24点查询MSE|全部支持可行|中位写入秒|\n|---|---:|---:|---:|\n"+
                                "\n".join(f"|{labels[r['method']]}|{r['mean_query_mse']:.8g}|{r['feasible']}/{r['count']}|{r['median_seconds']:.4f}|"
                                          for r in summary if r["n_context"]==24),encoding="utf-8")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),gridspec_kw={"width_ratios":[1.25,1.1,1]})
    styles={"local_branch_refined":("#1259aa","o"),"local_reset":("#689fdb","s"),
            "restored_slsqp64":("#d97706","D"),"prior1024":("#7c3aed","^"),"rbf":("#6b7280","v")}
    for method,(color,marker) in styles.items():
        group=[r for r in summary if r["method"]==method]
        axes[0].semilogy([r["n_context"] for r in group],[r["mean_query_mse"] for r in group],
                         marker=marker,color=color,label=labels[method])
    axes[0].set(xlabel="Cumulative observed contexts",ylabel="Independent query MSE",xticks=[4,8,16,24],
                title="A. True forward reads (8 development streams)")
    axes[0].legend(fontsize=7.4,loc="lower left")
    final=[r for r in summary if r["n_context"]==24]
    ypos=np.arange(len(final))
    colors=[styles.get(r["method"],("#b8c1cb",None))[0] for r in final]
    axes[1].barh(ypos,[r["mean_query_mse"] for r in final],color=colors)
    axes[1].set(xscale="log",yticks=ypos,yticklabels=[labels[r["method"]] for r in final],xlabel="Mean query MSE",
                title="B. All methods at 24 contexts")
    axes[1].tick_params(axis="y",labelsize=8); axes[1].invert_yaxis()
    xs=np.arange(4)
    necessary=[sum(r["n_context"]==n for r in certificates) for n in [4,8,16,24]]
    solved=[sum(r["n_context"]==n and r["solver_found_feasible"] for r in certificates) for n in [4,8,16,24]]
    axes[2].bar(xs-.18,necessary,width=.36,color="#475569",label="Crossing required (certificate)")
    axes[2].bar(xs+.18,solved,width=.36,color="#1259aa",label="Required and solved by refined ALM")
    axes[2].set(xticks=xs,xticklabels=[4,8,16,24],ylim=(0,9),xlabel="Cumulative contexts",ylabel="Count out of 8",
                title="C. A mathematical obstruction, then a write")
    axes[2].legend(fontsize=7.4)
    fig.suptitle("Streaming minimum-change memory: mechanism development, not a final superiority claim",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/"streaming_results.png",dpi=170); plt.close(fig)
    # A concise diagram separates the two mathematically different subproblems.
    fig,ax=plt.subplots(figsize=(12,4.6)); ax.axis("off")
    boxes=[(.02,.55,.27,.25,"1. Observe new key/value\nRetain earlier observed pairs\nNo query labels"),
           (.365,.55,.27,.25,"2. Discover an activation region\nLocal exact branch solves + duals\nNonconvex; no global guarantee"),
           (.71,.55,.27,.25,"3. Enforce the chosen region\nLocal convex primal-dual projection\nCheck real forward feasibility + gap")]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",facecolor="#edf4fb",edgecolor="#2864a4"))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=10.3)
    for a,b in [(.30,.35),(.65,.70)]:
        ax.annotate("",xy=(b,.675),xytext=(a,.675),arrowprops={"arrowstyle":"->","lw":2})
    ax.text(.5,.37,"Why crossing can be necessary:  G b <= r has no solution in the old region.",ha="center",fontsize=12)
    ax.text(.5,.22,"Certificate: w >= 0,  G^T w = 0,  r^T w < 0.\nThis rules out staying in that region, not every competing optimizer.",ha="center",fontsize=11,color="#334155")
    ax.set_title("The candidate's two jobs: find a feasible region, then make a controlled write",fontsize=14,pad=16)
    fig.tight_layout(); fig.savefig(out/"write_mechanism.png",dpi=170); plt.close(fig)
    print(json.dumps({"certified_cross_branch_writes":len(certificates),
                      "certified_obstructions_then_solved":stats["certified_obstructions_then_solved"],
                      "minimum_farkas_margin":min(r["margin"] for r in certificates),
                      "final":[r for r in summary if r["n_context"]==24]}),flush=True)


if __name__=="__main__": main()

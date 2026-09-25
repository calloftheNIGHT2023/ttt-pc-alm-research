"""Readout-only paired differences, sample/state costs, and conditional bounds."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
proto=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
assert len(rows)==proto["count"]*len(proto["stages"])*len(proto["configs"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto["source_sha256"].items())
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
summary=[]
for g in proto["generators"]:
    ref={(r["seed"],r["n_context"]):r for r in rows if r["generator"]==g["name"] and r["readout"]=="iid512"}
    for mode in proto["readouts"]:
        group=[r for r in rows if r["generator"]==g["name"] and r["readout"]==mode]
        seeds=sorted({r["seed"] for r in group}); differences=[]; totals=[]; readtimes=[]
        for seed in seeds:
            selected=[r for r in group if r["seed"]==seed]
            differences.append(np.mean([r["query_mse"]-ref[(seed,r["n_context"])]["query_mse"] for r in selected]))
            totals.append(sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in selected))
            readtimes.append(sum(r["read_2048_seconds"] for r in selected))
        rng=np.random.default_rng(88749); boot=np.mean(rng.choice(differences,(20000,len(seeds))),axis=1)
        stages=[]
        for n in proto["stages"]:
            selected=[r for r in group if r["n_context"]==n]
            stages.append({"n":n,"mean_samples":float(np.mean([r["readout_samples"] for r in selected])),
                "min_samples":min(r["readout_samples"] for r in selected),"max_samples":max(r["readout_samples"] for r in selected),
                "target_met":sum(r["readout_tolerance_met"] for r in selected),"mean_state_bytes":float(np.mean([r["persistent_state_bytes"] for r in selected])),
                "max_bound":max(r["readout_mc_variance_bound"] for r in selected)})
        summary.append({"generator":g["name"],"readout":mode,"trajectory_mse":float(np.mean([r["query_mse"] for r in group])),
            "paired_mse_difference_vs_iid512":float(np.mean(differences)),"descriptive_95ci":np.quantile(boot,[.025,.975]).tolist(),
            "median_total_stream_seconds":float(np.median(totals)),"median_read_stream_seconds":float(np.median(readtimes)),"stages":stages})
(out/"readout_mechanism.json").write_text(json.dumps({"summary":summary,"scope":"Development only. Variance bound vs truncated posterior mean; not missing mass or task-risk guarantee."},indent=2),encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
colors={"iid512":"#7c8798","adaptive_iid":"#087f8c","iid64":"#c26424","adaptive":"#9764ba"}
for name,color in colors.items():
    row=next(s for s in summary if s["generator"]=="bank4096_alm20" and s["readout"]==name)
    axes[0].plot(proto["stages"],[s["mean_samples"] for s in row["stages"]],"o-",label=name,color=color)
selected=[s for s in summary if s["generator"]=="bank4096_alm20"]
axes[0].set(xlabel="Observed contexts",ylabel="Mean readout sample count",title="Support-only budget reduction"); axes[0].legend()
axes[1].barh([s["readout"] for s in selected],[s["median_read_stream_seconds"] for s in selected],color=[colors.get(s["readout"],"#668399") for s in selected])
axes[1].set(xlabel="Seconds: four x 2048 queries",title="Readout cost (not full online adaptation)")
fig.suptitle("16 observed development streams; same discovered posterior")
fig.savefig(out/"readout_budget.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary}),flush=True)

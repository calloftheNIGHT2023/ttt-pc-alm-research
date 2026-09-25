"""Show the confirmed effects and the equivalent-implementation cost control."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]/"outputs/ttt-pc-alm-research/results/region_posterior/confirmation_analysis"
d=json.loads((root/"analysis.json").read_text(encoding="utf-8")); fast=json.loads((root/"compiled_prior_benchmark.json").read_text(encoding="utf-8"))
primary=[r for r in d["comparisons"] if r["primary"]]
fig,axes=plt.subplots(1,2,figsize=(14,5.6),gridspec_kw={"width_ratios":[1.1,1]})
labels={"trf64_wide":"TRF64 + same readout","lbfgs64_wide":"L-BFGS64 + same readout",
    "dictionary65536_nn16":"65,536-task nearest neighbors","prior4096_noiseaware":"Prior-moment closed-form head"}
for i,r in enumerate(primary):
    lo,hi=r["paired_two_sided_95_ci"]; mean=r["mean_mse_difference"]
    axes[0].errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt="o",color="#197b68" if r["primary_pass"] else "#b16b21",capsize=5)
    axes[0].text(-.031,i+.25,f"Adjusted upper bound: {r['primary_bonferroni_upper']:+.7f}",fontsize=8,color="#444444")
axes[0].axvline(0,color="#666666",linestyle="--",linewidth=1)
axes[0].set(yticks=range(len(primary)),yticklabels=[labels[r["baseline"]] for r in primary],
    xlim=(-.033,.005),ylim=(3.7,-.65),xlabel="Candidate minus baseline trajectory MSE (lower is better)",
    title="3 of 4 prespecified quality comparisons supported\nJoint four-comparison gate not passed")
axes[0].tick_params(axis="y",labelsize=8)
axes[0].text(.02,-.19,"Lines: paired 95% CIs. Labels: one-sided 98.75% upper bounds.",transform=axes[0].transAxes,fontsize=8)
byname={r["method"]:r for r in d["summary"]}
items=[("alm64_wide","Local ALM64",(-65,12)),("trf64_wide","TRF64",(-48,8)),
    ("lbfgs64_wide","L-BFGS64",(8,2)),("lbfgs128_priorbox","L-BFGS128 (prior box)",(-110,-22)),
    ("raw_prior2048_volume","Raw-region 2048",(-72,12)),("dictionary65536_nn16","Task dictionary",(-45,-18))]
for i,(name,label,offset) in enumerate(items):
    r=byname[name]; axes[1].scatter(r["median_total_seconds"],r["trajectory_mse"],s=70 if i==0 else 35,color="#197b68" if i==0 else "#416993")
    axes[1].annotate(label,(r["median_total_seconds"],r["trajectory_mse"]),xytext=offset,textcoords="offset points",fontsize=8)
r=byname["prior4096_noiseaware"]; before=r["median_total_seconds"]; after=fast["median_warm_cache_full_stream_fit_plus_query_seconds"]
axes[1].scatter([after,before],[r["trajectory_mse"]]*2,color="#888888",s=35)
axes[1].annotate("",xy=(after,r["trajectory_mse"]),xytext=(before,r["trajectory_mse"]),arrowprops={"arrowstyle":"->","color":"#888888"})
axes[1].annotate("Same closed-form predictor; compiled readout",(after,r["trajectory_mse"]),xytext=(5,12),textcoords="offset points",fontsize=8)
axes[1].set(xscale="log",xlim=(.012,8),ylim=(.012,.05),xlabel="Median fit + 4 x 2048 query seconds",ylabel="Trajectory MSE",
    title="Accuracy and cost must remain separate")
axes[1].grid(alpha=.15)
axes[1].text(.02,-.19,"Compiled prior: separate post-pass; cache build 0.530s, extra state counted.",transform=axes[1].transAxes,fontsize=8)
fig.suptitle("64 fresh streams | Fixed parameters, observations and common posterior readout for optimizer comparisons",fontsize=12)
fig.tight_layout(rect=(0,.06,1,.94)); fig.savefig(root/"confirmed_takeaways.png",dpi=180,bbox_inches="tight")
print(str(root/"confirmed_takeaways.png"))

"""Source-verified, paired stream-level analysis; all configurations retained."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
methods=[c["name"] for c in protocol["configs"]]; stages=protocol["stages"]
seeds=list(range(protocol["seed0"],protocol["seed0"]+protocol["count"]))
assert len(rows)==len(methods)*len(stages)*len(seeds)
assert len({(r["method"],r["seed"],r["n_context"]) for r in rows})==len(rows)
hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==s for f,s in protocol["source_sha256"].items()}
assert all(hashes.values())
summary=[]; byseed={}
table="|方法|全程MSE|4/8/16/24 MSE|4/8/16/24可行数|全流中位适应秒|全流中位读取秒|最大持久状态字节|\n|---|---:|---|---|---:|---:|---:|\n"
for m in methods:
    group=[r for r in rows if r["method"]==m]
    losses=np.array([np.mean([r["query_mse"] for r in group if r["seed"]==s]) for s in seeds]); byseed[m]=losses
    mse=[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in stages]
    feas=[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in stages]
    med=float(np.median([sum(r["adaptation_seconds"] for r in group if r["seed"]==s) for s in seeds]))
    read=float(np.median([sum(r["read_2048_seconds"] for r in group if r["seed"]==s) for s in seeds]))
    recorded=[r["persistent_state_bytes"] if "persistent_state_bytes" in r else 8*r["persistent_float64_scalars"] for r in group if "persistent_state_bytes" in r or "persistent_float64_scalars" in r]
    state=max(recorded) if recorded else None
    summary.append({"method":m,"trajectory_mse":float(losses.mean()),"stage_mse":mse,"stage_feasible":feas,
        "all_stage_feasible":sum(all(r["support_feasible"] for r in group if r["seed"]==s) for s in seeds),
        "median_stream_adaptation_seconds":med,"median_stream_read_seconds":read,"max_persistent_state_bytes":state})
    table+=f"|{m}|{losses.mean():.8f}|"+" / ".join(f"{v:.6g}" for v in mse)+f"|{feas}|{med:.4f}|{read:.6g}|{state}|\n"
rng=np.random.default_rng(967341); indexes=rng.integers(len(seeds),size=(20000,len(seeds)))
contrasts=[]
for a in methods:
    for b in methods:
        if a==b: continue
        diff=byseed[a]-byseed[b]; boot=diff[indexes].mean(axis=1)
        contrasts.append({"a":a,"b":b,"mean_difference":float(diff.mean()),"paired_bootstrap_95_ci":np.quantile(boot,[.025,.975]).tolist(),
            "a_better_streams":int(np.sum(diff<0)),"count":len(seeds),"interpretation":"development descriptive; not multiplicity-adjusted"})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(parents=True,exist_ok=True)
(out/"table.md").write_text(table,encoding="utf-8")
(out/"analysis.json").write_text(json.dumps({"phase":protocol["phase"],"hashes_match":hashes,"summary":summary,"paired_contrasts":contrasts},indent=2),encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(14,6.2),gridspec_kw={"width_ratios":[1,1.1]})
colors=plt.get_cmap("tab20")(np.linspace(0,1,max(2,len(summary))))
for i,(r,color) in enumerate(zip(summary,colors)):
    axes[0].semilogy(stages,r["stage_mse"],"o-",color=color,label=r["method"],linewidth=1.4)
    axes[1].barh(i,r["trajectory_mse"],color=color)
    axes[1].text(r["trajectory_mse"]+max(s["trajectory_mse"] for s in summary)*.015,i,
        f"{r['trajectory_mse']:.5f} | {r['median_stream_adaptation_seconds']:.3f}s",fontsize=8,va="center")
axes[0].set(xlabel="Observed contexts",ylabel="Unseen-query MSE",xticks=stages,title="Same colors identify the same methods")
axes[0].grid(alpha=.15)
axes[1].set(yticks=range(len(summary)),yticklabels=methods,xlabel="Trajectory MSE | median total adaptation seconds",
    xlim=(0,max(s["trajectory_mse"] for s in summary)*1.55))
axes[1].tick_params(axis="y",labelsize=8); axes[1].invert_yaxis()
fig.suptitle(f"Development replay: {len(seeds)} streams, all configurations; query labels evaluator-only")
fig.tight_layout(); fig.savefig(out/"overview.png",dpi=160)
print(json.dumps({"output":str(out),"hashes_match":hashes,"summary":summary}),flush=True)

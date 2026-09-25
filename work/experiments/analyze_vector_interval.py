"""Small pilot: exact source validation and all methods, no significance claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vector_interval_memory as model

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
proto=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
assert len(rows)==proto["count"]*len(proto["configs"])*len(proto["stages"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto["source_sha256"].items())
seeds=list(range(proto["seed0"],proto["seed0"]+proto["count"])); summary=[]
weights=model.family.make_weights(proto["depth"],proto["width"]); references=[]
for seed in seeds:
    rng=np.random.default_rng(seed); truth=rng.uniform(-model.PRIOR,model.PRIOR,(proto["depth"],proto["width"]))
    x=rng.uniform(-1,1,(max(proto["stages"]),proto["width"])); q=rng.uniform(-1,1,(proto["queries"],proto["width"]))
    target=model.forward(truth[None],q,weights)[0]; initial=model.forward(np.zeros_like(truth)[None],q,weights)[0]
    references.append({"seed":seed,"frozen_initial_model_mse":float(np.mean((initial-target)**2)),"target_coordinate_centered_variance":float(np.mean((target-target.mean(0))**2))})
for cfg in proto["configs"]:
    name=cfg["name"]; group=[r for r in rows if r["method"]==name]
    summary.append({"method":name,"trajectory_mse":float(np.mean([r["query_mse"] for r in group])),
        "stage_mse":[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in proto["stages"]],
        "stage_feasible":[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in proto["stages"]],
        "stage_mean_max_support_error":[float(np.mean([r["support_max_error"] for r in group if r["n_context"]==n])) for n in proto["stages"]],
        "median_total_stream_seconds":float(np.median([sum(r["adaptation_seconds"]+r["read_queries_seconds"] for r in group if r["seed"]==seed) for seed in seeds])),
        "max_predictor_state_bytes":max(r["persistent_state_bytes"] for r in group)})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
(out/"analysis.json").write_text(json.dumps({"summary":summary,"reference":references,"source_hashes_match":True,"scope":f'{proto["count"]}-stream development pilot, not a confirmation or significance test'},indent=2),encoding="utf-8")
table=["|method|trajectory MSE|8/16/24 MSE|8/16/24 feasible|median full stream seconds|","|---|---:|---|---|---:|"]
for r in summary:table.append(f'|{r["method"]}|{r["trajectory_mse"]:.7g}|'+" / ".join(f'{a:.6g}' for a in r["stage_mse"])+f'|{r["stage_feasible"]}|{r["median_total_stream_seconds"]:.4f}|')
(out/"table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
names=["local120","local480","no_dual480","gradient_dual480","batch_gn20","batch_adam240_003","batch_adam240_010","lbfgs16","prior1024","tangent_ridge",
    "consensus30","consensus120","consensus480","alternating480","batch_adam1000_010","prior4096",
    "joint240","joint240_no_dual","coordinate100_240","local240",
    "input_exact240","input_exact240_no_dual","input_gradient240",
    "affine_exact240","affine_exact240_no_dual","affine_activity_gradient240","affine_bias_gradient240",
    "adam240_r16","adam1000_r64","lbfgs64"]
for name in [name for name in names if any(r["method"]==name for r in summary)]:
    r=next(r for r in summary if r["method"]==name); ax.semilogy(proto["stages"],r["stage_mse"],"o-",label=name)
ax.set(xlabel="Observed vector contexts",ylabel="Unseen-query coordinate MSE",title=f'{proto["depth"]} layers x {proto["width"]} channels: {proto["depth"]*proto["width"]} fast biases; {proto["count"]} development streams',xticks=proto["stages"])
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8); ax.grid(alpha=.12); fig.savefig(out/"vector_pilot.png",dpi=160); plt.close(fig)
print(json.dumps({"summary":summary,"reference":references}),flush=True)

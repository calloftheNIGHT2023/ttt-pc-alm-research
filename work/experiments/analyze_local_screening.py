"""Preservation and full-cost accounting, never infer benefit from prune count."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
proto=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8")); audits=json.loads((args.results/"audit.json").read_text(encoding="utf-8"))
assert len(rows)==proto["count"]*len(proto["stages"])*len(proto["configs"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto["source_sha256"].items())
assert all(a["all_positive_regions_points_predictions_identical"] for a in audits)
summary=[]
for g in proto["generators"]:
    for name in proto["screens"]:
        group=[r for r in rows if r["generator"]==g["name"] and r["screen"]==name]
        sums=[]; geometry=[]; times=[]
        for seed in range(proto["seed0"],proto["seed0"]+proto["count"]):
            stream=[r for r in group if r["seed"]==seed]
            sums.append(sum(r["adaptation_seconds"]+r["read_2048_seconds"] for r in stream))
            geometry.append(sum(r["geometry_seconds"] for r in stream)); times.append(sum(r["screen_seconds"] for r in stream))
        summary.append({"generator":g["name"],"screen":name,"candidate_regions":sum(r["candidates"] for r in group),
            "screened":sum(r["screened"] for r in group),"positive_regions":sum(r["positive_volume_regions"] for r in group),
            "median_total_stream_seconds":float(np.median(sums)),"median_geometry_seconds":float(np.median(geometry)),
            "median_screen_seconds":float(np.median(times)),"trajectory_mse":float(np.mean([r["query_mse"] for r in group])),
            "max_screen_main_arrays_bytes_subtotal":max(r.get("main_arrays_bytes_subtotal",0) for r in group)})
out=args.results.parent/(args.results.name+"_analysis"); out.mkdir(exist_ok=True)
proof_count=sum(len(a["certificates"]) for a in audits)
result={"summary":summary,"all_predictions_points_positive_regions_identical":True,"writes_audited":len(audits),
    "exact_binary_rational_certificate_checks":proof_count,"pdhg60_rejections_beyond_contract20":sum(a["pdhg60_beyond_contract20"] for a in audits),
    "source_hashes_match":True,"scope":"Observed development streams; common implementation improvement, not PC-specific task superiority."}
(out/"mechanism.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
table=["|generator|screen|pruned / candidates|median total seconds|screen seconds|geometry seconds|MSE|","|---|---|---:|---:|---:|---:|---:|"]
for s in summary: table.append(f'|{s["generator"]}|{s["screen"]}|{s["screened"]} / {s["candidate_regions"]}|{s["median_total_stream_seconds"]:.6f}|{s["median_screen_seconds"]:.6f}|{s["median_geometry_seconds"]:.6f}|{s["trajectory_mse"]:.8f}|')
(out/"table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
selected=[s for s in summary if s["generator"]=="bank4096_alm20"]
axes[0].barh([s["screen"] for s in selected],[s["median_total_stream_seconds"] for s in selected],color="#087f8c")
axes[0].set(xlabel="Seconds per complete stream",title="ALM20: full fit + 4 x 2048 reads")
axes[1].barh([s["screen"] for s in selected],[s["screened"]/s["candidate_regions"] for s in selected],color="#647f9c")
axes[1].set(xlabel="Fraction of candidate regions safely rejected",title="Every posterior mode / prediction preserved",xlim=(0,1))
fig.suptitle("8 observed development streams; shared-variable interval controls included")
fig.savefig(out/"screening_cost.png",dpi=160); plt.close(fig)
print(json.dumps(result),flush=True)

"""Check repair/mass mechanisms independently of unseen-query error."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8")); rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
assert len(rows)==protocol["count"]*len(protocol["stages"])*len(protocol["configs"])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in protocol["source_sha256"].items())
summary=[]
for cfg in protocol["configs"]:
    group=[r for r in rows if r["method"]==cfg["name"] and "initial_feasible_parameter_volume" in r]
    if not group: continue
    for r in group: assert r["initial_region_preservation_verified"] and r["added_feasible_parameter_volume"]>=0
    repaired=[r for r in group if r["initial_feasible_parameter_volume"]==0 and r["added_feasible_parameter_volume"]>0]
    initially_empty=[r for r in group if r["initial_feasible_parameter_volume"]==0]
    summary.append({"method":cfg["name"],"stages":len(group),"initial_region_preservation_all":True,
        "initially_no_feasible_seed_region":len(initially_empty),"repaired_empty_seed_pool":len(repaired),
        "new_positive_volume_stages":sum(r["added_feasible_parameter_volume"]>0 for r in group),
        "mean_fraction_of_found_mass_added":float(np.mean([r["added_feasible_parameter_volume"]/(r["initial_feasible_parameter_volume"]+r["added_feasible_parameter_volume"])
            for r in group if r["initial_feasible_parameter_volume"]+r["added_feasible_parameter_volume"]>0])),
        "mean_proposal_seconds":float(np.mean([r["proposal_seconds"] for r in group])),
        "mean_refinement_seconds":float(np.mean([r["refinement_seconds"] for r in group])),
        "mean_geometry_seconds":float(np.mean([r["geometry_seconds"] for r in group])),
        "mean_sampling_seconds":float(np.mean([r["sampling_seconds"] for r in group]))})
checks={}
if any(r["method"]=="alm1" for r in rows):
    a={(r["seed"],r["n_context"]):r for r in rows if r["method"]=="alm1"}
    b={(r["seed"],r["n_context"]):r for r in rows if r["method"]=="no_dual1"}
    difference=max(abs(a[k]["query_mse"]-b[k]["query_mse"]) for k in a)
    assert difference==0
    checks["first_sweep_with_without_duals_identical"]=True
out=args.results.parent/(args.results.name+"_analysis")/"mechanism.json"
out.write_text(json.dumps({"summary":summary,"checks":checks},indent=2),encoding="utf-8")
print(json.dumps({"summary":summary,"checks":checks}),flush=True)

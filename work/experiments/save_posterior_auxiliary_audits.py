"""Persist readout-equivalence and numerical solver-status audits."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import compiled_piecewise_readout as spline
import audit_recursive_prior as recursive

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
statuses=Counter(tuple(t["statuses"]) for r in rows for t in r.get("geometry_trace",[]) if "statuses" in t)
reasons=Counter(t["reason"] for r in rows for t in r.get("geometry_trace",[]))
assert all(all(s in [0,2] for s in ss) and 2 in ss for ss in statuses),statuses
assert not any(reasons[r] for r in ["qhull_error","no_strict_numerical_interior","numerically_zero_width","zero_volume"])
result={"passed":True,"compiled_predictor":spline.verify(),"recursive_prior":recursive.verify(),
    "lp_status_combinations":{str(k):v for k,v in statuses.items()},"geometry_reasons":dict(reasons),
    "scope":"Floating geometry audited, not exact rational volume proof. Recursive equivalence tested on an independent audit fixture.",
    "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(spline.__file__),Path(recursive.__file__)]}}
out=args.results.parent/"confirmation_analysis"/"auxiliary_audits.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result),flush=True)

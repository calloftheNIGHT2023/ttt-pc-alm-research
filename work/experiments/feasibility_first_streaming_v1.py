"""Restore nonlinear write feasibility first, then project to the old anchor.

The local fit never sees query inputs/targets, teacher parameters, or BP credits.
Reuses audited kernels without changing previous experiment sources or hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import streaming_projection_refinement as refine


def fit(x,v,anchor,config):
    if config["kind"]=="restored_slsqp":
        return refine.fit_restored_slsqp(x,v,anchor,restarts=config.get("restarts",64))
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),{"skipped":True,"iterations":0,"polish_steps":0}
    discovery={k:config[k] for k in ["sweeps","restarts","dual_rate","gradient_blocks"] if k in config}
    # eta=1/(rho*n) in the exact local kernel. Infinity sets eta to zero;
    # trust remains positive, so all block denominators stay positive.
    discovery["rho"]=float("inf") if config.get("unanchored",True) else 10.
    b,_,meta=base.fit_local(x,v,anchor,warm=False,**discovery)
    rawerr=float(np.max(np.abs(base.forward(x,b)-v)))
    b,polish=refine.branch_refine(x,v,anchor,b,steps=config.get("polish_steps",8000))
    return b,{**meta,**polish,"before_polish_max_error":rawerr}


def summarize(rows):
    result=[]
    for method in dict.fromkeys(r["method"] for r in rows):
        for n in [4,8,16,24]:
            group=[r for r in rows if r["method"]==method and r["n_context"]==n]
            result.append({"method":method,"n_context":n,"count":len(group),
                           "mean_query_mse":float(np.mean([r["query_mse"] for r in group])),
                           "feasible":sum(r["support_feasible"] for r in group),
                           "median_seconds":float(np.median([r["seconds"] for r in group]))})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--confirm",action="store_true")
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    configs=json.loads(args.config.read_text(encoding="utf-8"))["configs"]
    seed0=5700000 if args.confirm else 5600000
    sources=[Path(__file__),Path(base.__file__),Path(refine.__file__),Path(base.__file__).with_name("local_branch_memory.py")]
    protocol={"phase":"confirmation" if args.confirm else "development","seed0":seed0,"count":args.count,
              "configs":configs,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "source_sha256":{a.name:hashlib.sha256(a.read_bytes()).hexdigest() for a in sources},
              "stages":[4,8,16,24],"query_count":2048,"epsilon":base.EPS,"slack":base.TOL}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for cfg in configs:
        for seed in range(seed0,seed0+args.count):
            rng=np.random.default_rng(seed)
            truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
            q=rng.uniform(0,1,2048); yq=base.forward(q,truth)
            b=np.zeros(4)
            for n in [4,8,16,24]:
                anchor=b.copy()
                start=time.perf_counter()
                b,meta=fit(x[:n],v[:n],anchor,cfg)
                seconds=time.perf_counter()-start
                er=float(np.max(np.abs(base.forward(x[:n],b)-v[:n])))
                rows.append({"method":cfg["name"],"seed":seed,"n_context":n,"seconds":seconds,
                             "query_mse":float(np.mean((base.forward(q,b)-yq)**2)),
                             "support_max_error":er,"support_feasible":bool(er<=base.EPS+base.TOL),
                             "move_squared":float(np.sum((b-anchor)**2)),"b":b.tolist(),"anchor":anchor.tolist(),**meta})
        print(json.dumps(summarize([r for r in rows if r["method"]==cfg["name"]])),flush=True)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        (args.out/"summary.json").write_text(json.dumps(summarize(rows),indent=2),encoding="utf-8")


if __name__=="__main__": main()

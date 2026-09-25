"""Predefined development matrix, with separate adaptation and query-read cost."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import joint_hybrid_memory as hybrid
import hybrid_branch_memory as oldhybrid
import joint_activity_bias_memory as discovery
import streaming_branch_projection as base
import streaming_projection_refinement as refine
from local_branch_memory import fit_regression


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    cfgall=json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True,exist_ok=True)
    sources=[Path(oldhybrid.__file__),Path(oldhybrid.__file__).with_name("hybrid_discovery_bank.py"),Path(__file__),Path(hybrid.__file__),Path(discovery.__file__),Path(base.__file__),Path(refine.__file__),
             Path(base.__file__).with_name("local_branch_memory.py")]
    protocol={**cfgall,"phase":"predeclared_development_scaling","source_sha256":{a.name:hashlib.sha256(a.read_bytes()).hexdigest() for a in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"n_queries":2048}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; ordering=np.random.default_rng(967)
    for depth in cfgall["depths"]:
        for seed in range(cfgall["seed0"],cfgall["seed0"]+cfgall["count"]):
            rng=np.random.default_rng(seed)
            truth=rng.uniform(-.12,.12,depth); x=rng.uniform(0,1,max(cfgall["stages"])); v=base.forward(x,truth)
            q=rng.uniform(0,1,2048); yq=base.forward(q,truth)
            # Observational audit only, never fed to the fit: common initialization.
            starts=np.vstack([np.zeros(depth),np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(63,depth))])
            zeros=float(np.mean([np.mean(base.forward_jacobian(x,z)[1]==0) for z in starts]))
            for ci in ordering.permutation(len(cfgall["configs"])):
                cfg=cfgall["configs"][int(ci)]; b=np.zeros(depth)
                for n in cfgall["stages"]:
                    anchor=b.copy(); begin=time.perf_counter()
                    if cfg["kind"]=="hybrid": b,meta=hybrid.fit(x[:n],v[:n],anchor,cfg)
                    elif cfg["kind"]=="old_hybrid": b,meta=oldhybrid.fit(x[:n],v[:n],anchor,cfg)
                    elif cfg["kind"]=="global": b,meta=refine.fit_restored_slsqp(x[:n],v[:n],anchor,restarts=cfg["restarts"])
                    else: predict,meta=fit_regression(x[:n],v[:n],depth,family=cfg["family"],features=cfg["features"])
                    if cfg["kind"]!="regression": predict=lambda z:base.forward(z,b)
                    elapsed=time.perf_counter()-begin
                    begin=time.perf_counter(); qp=np.clip(predict(q),0,1); readtime=time.perf_counter()-begin
                    err=float(np.max(np.abs(np.clip(predict(x[:n]),0,1)-v[:n])))
                    rows.append({"depth":depth,"seed":seed,"method":cfg["name"],"n_context":n,
                        "adaptation_seconds":elapsed,"read_2048_seconds":readtime,"adapt_plus_read_seconds":elapsed+readtime,
                        "query_mse":float(np.mean((qp-yq)**2)),"target_query_variance":float(np.var(yq)),
                        "support_max_error":err,"support_feasible":bool(err<=base.EPS+base.TOL),
                        "common_initial_support_jacobian_zero_fraction":zeros,
                        "b":b.tolist() if cfg["kind"]!="regression" else None,
                        "anchor":anchor.tolist() if cfg["kind"]!="regression" else None,**meta})
            (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            print(json.dumps({"depth":depth,"seed":seed,"completed_rows":len(rows)}),flush=True)
    summary=[]
    for depth in cfgall["depths"]:
        for cfg in cfgall["configs"]:
            for n in cfgall["stages"]:
                group=[r for r in rows if (r["depth"],r["method"],r["n_context"])==(depth,cfg["name"],n)]
                summary.append({"depth":depth,"method":cfg["name"],"n_context":n,
                    "query_mse":float(np.mean([r["query_mse"] for r in group])),
                    "feasible_count":sum(r["support_feasible"] for r in group),"count":len(group),
                    "median_adaptation_seconds":float(np.median([r["adaptation_seconds"] for r in group])),
                    "median_read_2048_seconds":float(np.median([r["read_2048_seconds"] for r in group]))})
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()

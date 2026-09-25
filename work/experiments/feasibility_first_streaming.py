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
from local_branch_memory import fit_regression, fit_shallow


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
    # In confirmation, interleave whole method streams per episode. Ordering is
    # independent of labels, to reduce drift from running one method for minutes.
    jobs=[(cfg,seed) for cfg in configs for seed in range(seed0,seed0+args.count)]
    if args.confirm:
        jobs=[]
        ordering=np.random.default_rng(998)
        for seed in range(seed0,seed0+args.count):
            jobs.extend((configs[int(i)],seed) for i in ordering.permutation(len(configs)))
    for ji,(cfg,seed) in enumerate(jobs):
            rng=np.random.default_rng(seed)
            truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
            q=rng.uniform(0,1,2048); yq=base.forward(q,truth)
            b=np.zeros(4)
            rls_w=np.zeros(2); rls_p=np.eye(2)/1e-4; rls_n=0
            for n in [4,8,16,24]:
                anchor=b.copy()
                start=time.perf_counter()
                if cfg["kind"]=="regression":
                    predict,meta=fit_regression(x[:n],v[:n],4,family=cfg["family"],features=cfg.get("features",256))
                elif cfg["kind"]=="shallow":
                    predict,meta=fit_shallow(x[:n],v[:n],4,width=cfg["width"],steps=1000,restarts=4)
                elif cfg["kind"]=="rls":
                    for i in range(rls_n,n):
                        phi=np.array([1.,x[i]])
                        gain=rls_p@phi/(1+phi@rls_p@phi)
                        rls_w+=gain*(v[i]-phi@rls_w)
                        rls_p-=np.outer(gain,phi@rls_p)
                    rls_n=n
                    predict=lambda z:rls_w[0]+rls_w[1]*z
                    meta={"ridge":1e-4,"persistent_scalars":6,"new_samples_processed":n-(0 if n==4 else {8:4,16:8,24:16}[n])}
                else:
                    b,meta=fit(x[:n],v[:n],anchor,cfg)
                    predict=lambda z:base.forward(z,b)
                seconds=time.perf_counter()-start
                # All methods receive the known bounded output prior.
                er=float(np.max(np.abs(np.clip(predict(x[:n]),0,1)-v[:n])))
                rows.append({"method":cfg["name"],"seed":seed,"n_context":n,"seconds":seconds,
                             "query_mse":float(np.mean((np.clip(predict(q),0,1)-yq)**2)),
                             "support_max_error":er,"support_feasible":bool(er<=base.EPS+base.TOL),
                             "move_squared":float(np.sum((b-anchor)**2)) if cfg["kind"] in ["local","restored_slsqp"] else None,
                             "b":b.tolist() if cfg["kind"] in ["local","restored_slsqp"] else None,
                             "anchor":anchor.tolist() if cfg["kind"] in ["local","restored_slsqp"] else None,**meta})
            if (ji+1)%len(configs)==0 or ji==len(jobs)-1:
                print(json.dumps({"completed_method_streams":ji+1,"total_method_streams":len(jobs)}),flush=True)
                (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    summary=summarize(rows)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()

"""Frozen-config streaming evaluator; query labels are outside every fit API."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize
import streaming_branch_projection as base
import streaming_projection_refinement as refine
import certified_branch_search as search
from feasibility_first_streaming import summarize
from local_branch_memory import fit_regression,fit_shallow


def global_bank(x,v,anchor,cfg):
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),{"skipped":True}
    bank,meta=search.discover(x,v,anchor,sweeps=cfg.get("sweeps",120),restarts=cfg.get("restarts",64))
    best=bank[0].copy(); err0,mov0=base.score(best[None],x,v,anchor)
    tried=0
    for start in bank[:cfg.get("max_regions",8)]:
        tried+=1
        _,_,mat,rhs=base.branch_polytope(x,v,start)
        lp=linprog(np.zeros(len(anchor)),A_ub=mat,b_ub=rhs,bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                   options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        if not lp.success: continue
        res=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),lp.x,jac=True,method="SLSQP",
             constraints={"type":"ineq","fun":lambda b:rhs-mat@b,"jac":lambda b:-mat},
             bounds=[(-base.BOUND,base.BOUND)]*len(anchor),options={"ftol":1e-12,"maxiter":300})
        for b in [lp.x,res.x]:
            err,mov=base.score(b[None],x,v,anchor)
            if base.better(err,mov,err0,mov0)[0]: best,err0,mov0=b.copy(),err,mov
        if err0[0]<=base.EPS+base.TOL: break
    return best,{**meta,"regions_attempted":tried}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--seed0",type=int,default=5600000)
    p.add_argument("--phase",default="development")
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    configs=json.loads(args.config.read_text(encoding="utf-8"))["configs"]
    sources=[Path(__file__),Path(search.__file__),Path(search.regional.__file__),
             Path(search.regional.__file__).with_name("certified_branch_solver.py"),
             Path(base.__file__),Path(refine.__file__),Path(base.__file__).with_name("local_branch_memory.py")]
    protocol={"phase":args.phase,"seed0":args.seed0,"count":args.count,"configs":configs,
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256":{a.name:hashlib.sha256(a.read_bytes()).hexdigest() for a in sources},
        "stages":[4,8,16,24],"query_count":2048,"epsilon":base.EPS,"slack":base.TOL}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; ordering=np.random.default_rng(1076)
    for seed in range(args.seed0,args.seed0+args.count):
        rng=np.random.default_rng(seed)
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
        q=rng.uniform(0,1,2048); yq=base.forward(q,truth)
        for ci in ordering.permutation(len(configs)):
            cfg=configs[int(ci)]; b=np.zeros(4)
            for n in [4,8,16,24]:
                anchor=b.copy(); begin=time.perf_counter()
                if cfg["kind"]=="local": b,meta=search.fit(x[:n],v[:n],anchor,cfg)
                elif cfg["kind"]=="global_bank": b,meta=global_bank(x[:n],v[:n],anchor,cfg)
                elif cfg["kind"]=="restored_slsqp":
                    b,meta=refine.fit_restored_slsqp(x[:n],v[:n],anchor,restarts=cfg.get("restarts",64))
                elif cfg["kind"]=="regression":
                    predict,meta=fit_regression(x[:n],v[:n],4,family=cfg["family"],features=cfg.get("features",256))
                elif cfg["kind"]=="shallow":
                    predict,meta=fit_shallow(x[:n],v[:n],4,width=64,steps=1000,restarts=4)
                else: raise ValueError(cfg)
                if cfg["kind"] not in ["regression","shallow"]: predict=lambda z:base.forward(z,b)
                seconds=time.perf_counter()-begin
                er=float(np.max(np.abs(np.clip(predict(x[:n]),0,1)-v[:n])))
                rows.append({"method":cfg["name"],"seed":seed,"n_context":n,"seconds":seconds,
                    "query_mse":float(np.mean((np.clip(predict(q),0,1)-yq)**2)),
                    "support_max_error":er,"support_feasible":bool(er<=base.EPS+base.TOL),
                    "b":b.tolist(),"anchor":anchor.tolist(),"move_squared":float(np.sum((b-anchor)**2)),**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed_seeds":seed-args.seed0+1,"total_seeds":args.count}),flush=True)
    summary=summarize(rows)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()

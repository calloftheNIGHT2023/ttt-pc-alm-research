"""Compare discovery with a common regional projector and stronger prior heads."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import common_projected_memory as common
import matched_discovery_baselines as bp
import prior_moment_regression as prior
import streaming_branch_projection as base
import streaming_projection_refinement as refine


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); config=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(common.__file__),Path(bp.__file__),Path(prior.__file__),Path(base.__file__),Path(refine.__file__),
        Path(base.__file__).with_name("hybrid_discovery_bank.py"),Path(base.__file__).with_name("local_branch_memory.py")]
    protocol={**config,"phase":"development_replay_observed_5800000","source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"stages":[4,8,16,24],"queries":2048,
        "verification":{"bp":bp.verify_gradient(),"prior":prior.verify()}}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    ordering=np.random.default_rng(968); rows=[]; stages=protocol["stages"]
    for seed in range(config["seed0"],config["seed0"]+config["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
        q=rng.uniform(0,1,2048); yq=base.forward(q,truth)
        for ci in ordering.permutation(len(config["configs"])):
            cfg=config["configs"][int(ci)]; b=np.zeros(4)
            for n in stages:
                anchor=b.copy(); begin=time.perf_counter()
                if cfg["kind"]=="common": b,meta=common.fit(x[:n],v[:n],anchor,cfg)
                elif cfg["kind"]=="global": b,meta=refine.fit_restored_slsqp(x[:n],v[:n],anchor,restarts=cfg["restarts"])
                else: predict,meta=prior.fit(x[:n],v[:n],4,features=cfg["features"])
                if cfg["kind"]!="prior": predict=lambda z:base.forward(z,b)
                elapsed=time.perf_counter()-begin
                begin=time.perf_counter(); qp=predict(q); read=time.perf_counter()-begin
                err=float(np.max(np.abs(np.clip(predict(x[:n]),0,1)-v[:n])))
                rows.append({"method":cfg["name"],"seed":seed,"n_context":n,"adaptation_seconds":elapsed,
                    "read_2048_seconds":read,"query_mse":float(np.mean((np.clip(qp,0,1)-yq)**2)),
                    "raw_query_mse":float(np.mean((qp-yq)**2)),"support_max_error":err,"support_feasible":bool(err<=base.EPS+base.TOL),
                    "anchor":anchor.tolist() if cfg["kind"]!="prior" else None,"b":b.tolist() if cfg["kind"]!="prior" else None,**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-config["seed0"]+1,"count":config["count"]}),flush=True)
    summary=[]
    for cfg in config["configs"]:
        group=[r for r in rows if r["method"]==cfg["name"]]
        summary.append({"method":cfg["name"],"trajectory_mse":float(np.mean([r["query_mse"] for r in group])),
            "stage_mse":[float(np.mean([r["query_mse"] for r in group if r["n_context"]==n])) for n in stages],
            "stage_feasible":[sum(r["support_feasible"] for r in group if r["n_context"]==n) for n in stages],
            "median_stream_adaptation_seconds":float(np.median([sum(r["adaptation_seconds"] for r in group if r["seed"]==seed) for seed in range(config["seed0"],config["seed0"]+config["count"])])),
            "median_stream_read_seconds":float(np.median([sum(r["read_2048_seconds"] for r in group if r["seed"]==seed) for seed in range(config["seed0"],config["seed0"]+config["count"])]))})
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()

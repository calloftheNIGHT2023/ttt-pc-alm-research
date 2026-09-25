"""Predeclared append-only experiment; teacher and queries are evaluator-only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import persistent_branch_memory as model
import streaming_branch_projection as base


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(model.__file__),Path(base.__file__),Path(__file__).with_name("matched_discovery_baselines.py"),Path(__file__).with_name("hybrid_discovery_bank.py")]
    protocol={**cfg,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audit":model.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; ordering=np.random.default_rng(9321)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,cfg["depth"])
        x=rng.uniform(0,1,max(cfg["stages"])); v=base.forward(x,truth)
        q=rng.uniform(0,1,cfg["queries"]); target=base.forward(q,truth)
        for mi in ordering.permutation(len(cfg["configs"])):
            c=cfg["configs"][mi]; b=np.zeros(cfg["depth"]); state=None
            for n in cfg["stages"]:
                anchor=b.copy(); start=time.perf_counter(); b,state,meta=model.fit(x[:n],v[:n],anchor,c,state)
                adapt=time.perf_counter()-start; start=time.perf_counter(); pred=base.forward(q,b); read=time.perf_counter()-start
                error=float(np.max(np.abs(base.forward(x[:n],b)-v[:n])))
                rows.append({"method":c["name"],"seed":seed,"n_context":n,"depth":cfg["depth"],"query_mse":float(np.mean((pred-target)**2)),
                    "support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),"adaptation_seconds":adapt,
                    "read_2048_seconds":read,"anchor":anchor.tolist(),"b":b.tolist(),**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__": main()

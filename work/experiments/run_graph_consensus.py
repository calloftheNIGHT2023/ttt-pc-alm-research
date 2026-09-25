"""Fixed graph-consensus pilot; all query targets evaluator-only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import vector_interval_memory as task
import graph_consensus_memory as model


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/"protocol.json").exists()
    sources=[Path(__file__),Path(model.__file__),Path(task.__file__),Path(task.family.__file__)]
    protocol={**cfg,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":{"joint_projection":model.verify(),"shared_task":task.verify()},
        "teacher_prior_bound":task.PRIOR,"discovery_bound":task.BOUND,"epsilon":task.EPS,"tolerance":task.TOL,
        "all_internal_default_restarts":16,"all_internal_prior_tasks":256}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    weights=task.family.make_weights(cfg["depth"],cfg["width"]); rows=[]; order=np.random.default_rng(531709)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-task.PRIOR,task.PRIOR,(cfg["depth"],cfg["width"]))
        x=rng.uniform(-1,1,(max(cfg["stages"]),cfg["width"])); q=rng.uniform(-1,1,(cfg["queries"],cfg["width"]))
        v=task.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-task.EPS,task.EPS,x.shape)
        target=task.forward(truth[None],q,weights)[0]
        for ci in order.permutation(len(cfg["configs"])):
            c=cfg["configs"][ci]; point=np.zeros_like(truth)
            for n in cfg["stages"]:
                start=time.perf_counter()
                if c["method"]=="prior":predict,meta=task.fit_closed(x[:n],v[:n],weights,c)
                else:predict,point,meta=model.fit(x[:n],v[:n],point,weights,c)
                fit=time.perf_counter()-start; start=time.perf_counter(); pred=predict(q); read=time.perf_counter()-start
                error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                rows.append({"method":c["name"],"seed":seed,"n_context":n,"query_mse":float(np.mean((pred-target)**2)),
                    "support_max_error":error,"support_feasible":bool(error<=task.EPS+task.TOL),"adaptation_seconds":fit,"read_queries_seconds":read,
                    "common_context_bytes":x[:n].nbytes+v[:n].nbytes,**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__":main()

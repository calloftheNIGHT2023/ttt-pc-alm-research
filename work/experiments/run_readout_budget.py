"""Shared posterior, five fixed readouts, and evaluators-only query answers."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import readout_budget_pipeline as model
import variance_controlled_readout as readout


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/"protocol.json").exists()
    names=["readout_budget_pipeline.py","variance_controlled_readout.py","contextual_candidate_bank.py","contextual_posterior_memory.py",
        "streaming_branch_projection.py","region_posterior_memory.py","matched_discovery_baselines.py","local_branch_memory.py"]
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    methods=[{"name":c["name"]+"_"+r} for c in cfg["generators"] for r in cfg["readouts"]]
    protocol={**cfg,"configs":methods,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":model.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; ordering=np.random.default_rng(94316)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        q=rng.uniform(0,1,2048); target=base.forward(q,truth)
        for ci in ordering.permutation(len(cfg["generators"])):
            c=cfg["generators"][ci]; point=np.zeros(4)
            for n in cfg["stages"]:
                polys,point,meta=model.construct(x[:n],v[:n],point,c)
                for ri in ordering.permutation(len(cfg["readouts"])):
                    mode=cfg["readouts"][ri]; predict,rm=readout.build(polys,point,mode,cfg["readout_variance_target"])
                    begin=time.perf_counter(); pred=predict(q); elapsed=time.perf_counter()-begin
                    error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                    rows.append({"method":c["name"]+"_"+mode,"generator":c["name"],"readout":mode,"seed":seed,"n_context":n,
                        "query_mse":float(np.mean((pred-target)**2)),"support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),
                        "adaptation_seconds":meta["common_discovery_and_geometry_seconds"]+rm["readout_seconds"],"read_2048_seconds":elapsed,
                        "persistent_state_bytes":rm["readout_state_bytes"]+point.nbytes,**meta,**rm})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__": main()

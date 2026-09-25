"""Finite-budget BP ablation, source-frozen before observing results."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import finite_bp_budget as model


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/"protocol.json").exists()
    names=["finite_bp_budget.py","screened_contextual_memory.py","run_local_screening.py","local_region_screen.py","variance_controlled_readout.py",
        "contextual_candidate_bank.py","posterior_confirmation_pipeline.py","streaming_branch_projection.py","region_posterior_memory.py",
        "region_posterior_controls.py","matched_discovery_baselines.py","hybrid_discovery_bank.py","local_branch_memory.py","certified_branch_solver.py"]
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    protocol={**cfg,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":model.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; order=np.random.default_rng(12953)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        q=rng.uniform(0,1,2048); target=base.forward(q,truth)
        for ci in order.permutation(len(cfg["configs"])):
            c=cfg["configs"][ci]; b=np.zeros(4)
            for n in cfg["stages"]:
                begin=time.perf_counter(); predict,b,meta=model.fit(x[:n],v[:n],b,c); elapsed=time.perf_counter()-begin
                begin=time.perf_counter(); pred=predict(q); read=time.perf_counter()-begin
                error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                rows.append({"method":c["name"],"seed":seed,"n_context":n,"query_mse":float(np.mean((pred-target)**2)),
                    "support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),"adaptation_seconds":elapsed,"read_2048_seconds":read,**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__":main()

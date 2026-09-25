"""Noisy-context development with common posterior readout for all discovery."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import region_posterior_memory as model
import streaming_branch_projection as base


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(model.__file__),Path(base.__file__),Path(__file__).with_name("hybrid_discovery_bank.py"),Path(__file__).with_name("matched_discovery_baselines.py")]
    methods=[{"name":c["name"]+"_"+mode} for c in cfg["configs"] for mode in ["point","equal","volume"]]+[{"name":"prior4096_noiseaware"}]
    protocol={**cfg,"generators":cfg["configs"],"configs":methods,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":model.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    ordering=np.random.default_rng(193); rows=[]
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,cfg["depth"])
        x=rng.uniform(0,1,max(cfg["stages"])); v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,len(x))
        q=rng.uniform(0,1,cfg["queries"]); yq=base.forward(q,truth)
        for gi in ordering.permutation(len(cfg["configs"])+1):
            anchor=np.zeros(cfg["depth"])
            for n in cfg["stages"]:
                start=time.perf_counter()
                if gi==len(cfg["configs"]):
                    pred,meta=model.prior_moments(x[:n],v[:n],cfg["depth"]); predictions={"prior4096_noiseaware":pred}
                else:
                    c=cfg["configs"][gi]; preds,anchor,meta=model.fit(x[:n],v[:n],anchor,{**c,"posterior_samples":cfg["posterior_samples"]})
                    predictions={c["name"]+"_"+mode:predict for mode,predict in preds.items()}
                elapsed=time.perf_counter()-start
                for name,predict in predictions.items():
                    begin=time.perf_counter(); pq=predict(q); read=time.perf_counter()-begin
                    error=float(np.max(np.abs(np.clip(predict(x[:n]),0,1)-v[:n])))
                    mode=name.rsplit("_",1)[-1]
                    state=meta.get("retained_predictor_bytes",{}).get(mode,meta.get("persistent_state_bytes"))
                    rows.append({"method":name,"seed":seed,"n_context":n,"query_mse":float(np.mean((np.clip(pq,0,1)-yq)**2)),
                        "raw_query_mse":float(np.mean((pq-yq)**2)),"support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),
                        "adaptation_seconds":elapsed,"read_2048_seconds":read,"persistent_state_bytes":state,**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__": main()

"""Strong observation/prior-matched controls for region posterior discovery."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import region_posterior_controls as model
import region_posterior_memory as posterior


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    names=["hybrid_discovery_bank.py","matched_discovery_baselines.py","local_branch_memory.py"]
    sources=[Path(__file__),Path(base.__file__),Path(model.__file__),Path(posterior.__file__)]+[Path(__file__).with_name(n) for n in names]
    protocol={**cfg,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":model.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; ordering=np.random.default_rng(14629)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,cfg["depth"])
        x=rng.uniform(0,1,max(cfg["stages"])); v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,len(x))
        q=rng.uniform(0,1,cfg["queries"]); target=base.forward(q,truth)
        for ci in ordering.permutation(len(cfg["configs"])):
            c=cfg["configs"][ci]; b=np.zeros(cfg["depth"])
            for n in cfg["stages"]:
                begin=time.perf_counter()
                if c["kind"]=="region": pred,b,meta=model.fit(x[:n],v[:n],b,{**c,"posterior_samples":cfg["posterior_samples"]})
                elif c["kind"]=="dictionary": pred,meta=model.dictionary(x[:n],v[:n],cfg["depth"],c)
                else: pred,meta=posterior.prior_moments(x[:n],v[:n],cfg["depth"],c["features"])
                adapt=time.perf_counter()-begin; begin=time.perf_counter(); qp=pred(q); read=time.perf_counter()-begin
                error=float(np.max(np.abs(np.clip(pred(x[:n]),0,1)-v[:n])))
                rows.append({"method":c["name"],"seed":seed,"n_context":n,"query_mse":float(np.mean((np.clip(qp,0,1)-target)**2)),
                    "raw_query_mse":float(np.mean((qp-target)**2)),"support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),
                    "adaptation_seconds":adapt,"read_2048_seconds":read,**meta})
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__": main()

"""Frozen unseen-stream confirmation; safe complete-stream checkpoint resume."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import local_branch_memory as baseline
import region_posterior_controls as controls
import region_posterior_memory as posterior
import posterior_confirmation_pipeline as model


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--resume",action="store_true"); args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    names=["hybrid_discovery_bank.py","matched_discovery_baselines.py"]
    sources=[Path(__file__),Path(base.__file__),Path(baseline.__file__),Path(controls.__file__),Path(posterior.__file__),Path(model.__file__)]+[Path(__file__).with_name(n) for n in names]
    hashes={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}; config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.resume:
        protocol=json.loads((args.out/"protocol.json").read_text(encoding="utf-8"))
        assert protocol["source_sha256"]==hashes and protocol["config_sha256"]==config_hash
        rows=json.loads((args.out/"episodes.json").read_text(encoding="utf-8"))
        completed={r["seed"] for r in rows}
        assert len(rows)==len(completed)*len(cfg["configs"])*len(cfg["stages"])
    else:
        assert not (args.out/"protocol.json").exists(),"Refusing to overwrite an existing run"
        protocol={**cfg,"source_sha256":hashes,"config_sha256":config_hash,"audit":model.verify()}
        (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
        rows=[]; completed=set()
    ordering=np.random.default_rng(14597)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        order=ordering.permutation(len(cfg["configs"]))
        if seed in completed: continue
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,cfg["depth"])
        x=rng.uniform(0,1,max(cfg["stages"])); v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,len(x))
        q=rng.uniform(0,1,cfg["queries"]); target=base.forward(q,truth)
        episode=[]
        for ci in order:
            c=cfg["configs"][ci]; b=np.zeros(cfg["depth"])
            for n in cfg["stages"]:
                begin=time.perf_counter()
                if c["kind"]=="region": pred,b,meta=model.fit(x[:n],v[:n],b,{**c,"posterior_samples":cfg["posterior_samples"]})
                elif c["kind"]=="dictionary": pred,meta=controls.dictionary(x[:n],v[:n],cfg["depth"],c)
                elif c["kind"]=="prior": pred,meta=posterior.prior_moments(x[:n],v[:n],cfg["depth"],c["features"])
                elif c["kind"]=="regression": pred,meta=baseline.fit_regression(x[:n],v[:n],cfg["depth"],family=c["family"])
                else: pred,meta=baseline.fit_shallow(x[:n],v[:n],cfg["depth"],width=c["width"])
                adapt=time.perf_counter()-begin; begin=time.perf_counter(); qp=pred(q); read=time.perf_counter()-begin
                error=float(np.max(np.abs(np.clip(pred(x[:n]),0,1)-v[:n])))
                episode.append({"method":c["name"],"seed":seed,"n_context":n,"query_mse":float(np.mean((np.clip(qp,0,1)-target)**2)),
                    "raw_query_mse":float(np.mean((qp-target)**2)),"support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),
                    "adaptation_seconds":adapt,"read_2048_seconds":read,**meta})
        rows.extend(episode)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"]}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__": main()

"""Same candidate pool and readout; screening must preserve every positive mode."""
import argparse,hashlib,json,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.optimize import minimize
import streaming_branch_projection as base
import contextual_candidate_bank as candidates
import region_posterior_memory as geometry
import variance_controlled_readout as readout
import local_region_screen as screen
from certified_branch_solver import exact_certificate


def screen_bank(x,v,bank,mode):
    start=time.perf_counter(); regs=np.stack([base.pattern(x,b) for b in bank]); proofs={}; meta={}
    if mode=="none": mask=np.zeros(len(bank),dtype=bool)
    elif mode=="ibp": mask=screen.forward_intervals(x,v,regs)
    elif mode.startswith("contract"):
        mask=screen.contract(x,v,regs,20 if mode=="contract20" else 5)
        if mode=="contract5_pdhg60":
            ids=np.flatnonzero(~mask)
            if len(ids):
                extra,meta=screen.pdhg(x,v,bank[ids],regs[ids],60)
                proofs={int(ids[k]):p for k,p in meta.pop("certificates").items()}; mask[ids]|=extra
    else:
        mask,meta=screen.pdhg(x,v,bank,regs,int(mode[4:])); proofs=meta.pop("certificates")
    elapsed=time.perf_counter()-start
    if mode=="none": elapsed=0. # No pattern work is necessary without a screen.
    return mask,proofs,meta,elapsed


def solve_bank(x,v,anchor,bank,mask):
    start=time.perf_counter(); polys=[]; point=None; notes=[]; ids=[]
    for i,b in enumerate(bank):
        if mask[i]: notes.append({"reason":"screened_infeasible"}); continue
        _,_,g,rhs=base.branch_polytope(x,v,b); poly,note=geometry.polytope(g,rhs); notes.append(note)
        if poly is None: continue
        ids.append(i); polys.append(poly)
        if point is None:
            initial=poly["center"]+poly["scale"]*poly["interior"]
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method="SLSQP",
                constraints={"type":"ineq","fun":lambda b:poly["rhs"]-poly["a"]@b,"jac":lambda b:-poly["a"]},
                bounds=[(-.12,.12)]*len(anchor),options={"maxiter":300,"ftol":1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    if point is None: point=np.clip(bank[0],-.12,.12)
    return polys,point,ids,notes,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8")); args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/"protocol.json").exists()
    names=["local_region_screen.py","contextual_candidate_bank.py","streaming_branch_projection.py","region_posterior_memory.py",
        "variance_controlled_readout.py","local_branch_memory.py","matched_discovery_baselines.py","certified_branch_solver.py"]
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    protocol={**cfg,"configs":[{"name":g["name"]+"_"+s} for g in cfg["generators"] for s in cfg["screens"]],
        "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"audits":screen.verify()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]; audit=[]; order=np.random.default_rng(20918)
    for seed in range(cfg["seed0"],cfg["seed0"]+cfg["count"]):
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        q=rng.uniform(0,1,2048); target=base.forward(q,truth)
        for gi in order.permutation(len(cfg["generators"])):
            cfg_g=cfg["generators"][gi]; anchor=np.zeros(4)
            for n in cfg["stages"]:
                xx,vv=x[:n],v[:n]; begin=time.perf_counter()
                starts,pm=candidates.proposals(xx,vv,anchor,cfg_g["features"],cfg_g["restarts"])
                if cfg_g["generator"]=="direct": bank=starts
                elif cfg_g["generator"]=="alm":
                    refined,_=candidates.refine_local(starts,xx,vv,anchor,cfg_g["sweeps"]); bank=np.vstack([starts,refined])
                else:
                    refined,_=candidates.refine_bp(starts,xx,vv,anchor,cfg_g["generator"]); bank=np.vstack([starts,refined])
                bank=candidates.deduplicate(bank,xx,vv,anchor); proposal_time=time.perf_counter()-begin
                same=[]; masks={}; local_audit=[]
                for si in order.permutation(len(cfg["screens"])):
                    name=cfg["screens"][si]; mask,proofs,sm,screen_time=screen_bank(xx,vv,bank,name)
                    polys,point,ids,notes,geometry_time=solve_bank(xx,vv,anchor,bank,mask)
                    predict,rm=readout.build(polys,point,"adaptive_iid",1e-6)
                    begin=time.perf_counter(); pred=predict(q); read_time=time.perf_counter()-begin
                    error=float(np.max(np.abs(predict(xx)-vv)))
                    same.append({"ids":ids,"point":point,"pred":pred}); masks[name]=mask
                    rows.append({"method":cfg_g["name"]+"_"+name,"generator":cfg_g["name"],"screen":name,"seed":seed,"n_context":n,
                        "query_mse":float(np.mean((pred-target)**2)),"support_max_error":error,"support_feasible":bool(error<=base.EPS+base.TOL),
                        "adaptation_seconds":proposal_time+screen_time+geometry_time+rm["readout_seconds"],"read_2048_seconds":read_time,
                        "persistent_state_bytes":rm["readout_state_bytes"]+point.nbytes,"proposal_refinement_seconds":proposal_time,
                        "screen_seconds":screen_time,"geometry_seconds":geometry_time,"candidates":len(bank),"screened":int(mask.sum()),
                        "remaining_geometry_calls":int((~mask).sum()),"positive_volume_regions":len(polys),"geometry_trace":notes,**sm,**rm,**pm})
                    # Exact proof checks are audit-only, AFTER online timings.
                    saved=base.BOUND; base.BOUND=.12
                    try:
                        for idx,proof in proofs.items():
                            regs=base.pattern(xx,bank[idx]); ex=exact_certificate(xx,vv,regs,proof["p"],proof["a"])
                            ef=F(int(ex["numerator"]),int(ex["denominator"]))
                            assert ex["positive"] and F(proof["lower"])<=ef
                            local_audit.append({"screen":name,"candidate_index":idx,"lower":proof["lower"],"exact_value":ex["value"],
                                "numerator":ex["numerator"],"denominator":ex["denominator"],"step":proof["step"],"dual_proposal":proof["proposal"]})
                    finally: base.BOUND=saved
                assert all(t["ids"]==same[0]["ids"] and np.array_equal(t["point"],same[0]["point"]) and np.array_equal(t["pred"],same[0]["pred"]) for t in same)
                audit.append({"seed":seed,"generator":cfg_g["name"],"n_context":n,"all_positive_regions_points_predictions_identical":True,
                    "pdhg60_beyond_contract20":int(np.sum(masks["pdhg60"]&~masks["contract20"])),"certificates":local_audit})
                anchor=same[0]["point"].copy()
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        (args.out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
        print(json.dumps({"completed":seed-cfg["seed0"]+1,"total":cfg["count"],"rows":len(rows)}),flush=True)
    print(json.dumps({"complete":True,"rows":len(rows)}),flush=True)


if __name__=="__main__":main()

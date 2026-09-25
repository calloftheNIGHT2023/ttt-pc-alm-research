"""Isolated full-stream state audit plus mathematically equivalent fast reads.

Run AFTER the timed confirmation finishes. Timings inside tracemalloc are marked
instrumented and must not be substituted for the main interleaved fit timings.
"""
import argparse,hashlib,json,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import streaming_branch_projection as base
import posterior_confirmation_pipeline as model
import region_posterior_controls as controls
import region_posterior_memory as posterior
import compiled_piecewise_readout as spline


def captured(function):
    return dict(zip(function.__code__.co_freevars,(cell.cell_contents for cell in function.__closure__)))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--method",required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); config=json.loads(args.config.read_text(encoding="utf-8")); cfg=next(c for c in config["configs"] if c["name"]==args.method)
    seed=5800000; rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
    v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
    q=rng.uniform(0,1,2048); b=np.zeros(4)
    rss_before=psutil.Process().memory_info().rss; tracemalloc.start(); stages=[]
    for n in [4,8,16,24]:
        begin=time.perf_counter()
        if cfg["kind"]=="region": pred,b,meta=model.fit(x[:n],v[:n],b,{**cfg,"posterior_samples":512})
        elif cfg["kind"]=="dictionary": pred,meta=controls.dictionary(x[:n],v[:n],4,cfg)
        elif cfg["kind"]=="prior": pred,meta=posterior.prior_moments(x[:n],v[:n],4,cfg["features"])
        else: raise ValueError("Audit currently covers task-family predictors only")
        elapsed=time.perf_counter()-begin; pred(q)
        stages.append({"n_context":n,"instrumented_fit_seconds":elapsed,"reported_persistent_state_bytes":meta.get("persistent_state_bytes")})
    tracked_current,tracked_peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    info=psutil.Process().memory_info()
    capture=captured(pred)
    if cfg["kind"]=="prior":
        bank=captured(capture["values"])["bank"]; weights=capture["weights"]
    else:
        bank=capture["bank"]; weights=np.full(len(bank),1/len(bank))
    begin=time.perf_counter(); cache=spline.prepare(bank); prepare=time.perf_counter()-begin
    begin=time.perf_counter(); compiled,cm=spline.compile_weights(cache,weights); reweight=time.perf_counter()-begin
    auditq=np.r_[q,np.linspace(0,1,4097)]
    error=float(np.max(np.abs(pred(auditq)-compiled(auditq)))); assert error<1e-8,error
    times={}
    for name,fn in [("direct",pred),("compiled",compiled)]:
        fn(q); samples=[]
        for _ in range(20):
            begin=time.perf_counter(); fn(q); samples.append(time.perf_counter()-begin)
        times[name]=float(np.median(samples))
    saved=times["direct"]-times["compiled"]
    result={"method":args.method,"audit_seed":seed,"stages":stages,"tracemalloc_peak_bytes":tracked_peak,"tracemalloc_retained_bytes":tracked_current,
        "process_rss_before_bytes":rss_before,"process_rss_after_bytes":info.rss,"process_peak_working_set_bytes":getattr(info,"peak_wset",None),
        "memory_boundary":"Process peak includes interpreter/imports; tracemalloc can miss native solver allocations. One audited stream is not a global peak bound.",
        "compiled_readout":{"function_bank_size":len(bank),"prepare_seconds":prepare,"reweight_seconds":reweight,
            "cache_bytes":sum(a.nbytes for a in cache.values()),**cm,"same_prediction_max_abs_error":error,
            "median_2048_query_seconds":times,"estimated_break_even_query_count":2048*(prepare+reweight)/saved if saved>0 else None,
            "reuse_boundary":"Fixed prior bank geometry can be cached across tasks; posterior sample banks change at each write. Count cache storage and construction separately."},
        "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(spline.__file__),Path(model.__file__)]}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"method":args.method,"tracked_peak_bytes":tracked_peak,"compiled_readout":result["compiled_readout"]}),flush=True)


if __name__=="__main__": main()

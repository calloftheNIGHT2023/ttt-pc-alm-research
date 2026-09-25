"""Isolated common-pipeline peak/state audit, run after uninstrumented timings."""
import argparse,hashlib,json,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import streaming_branch_projection as base
import batched_bp_pipeline as model


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--method",required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    protocol=json.loads(args.config.read_text(encoding="utf-8")); cfg=next(c for c in protocol["configs"] if c["name"]==args.method)
    seed=5800000; rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
    v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24); q=rng.uniform(0,1,2048)
    before=psutil.Process().memory_info().rss; tracemalloc.start(); point=np.zeros(4); stages=[]
    for n in [4,8,16,24]:
        start=time.perf_counter(); predict,point,meta=model.fit(x[:n],v[:n],point,cfg); fit=time.perf_counter()-start; predict(q)
        current,peak=tracemalloc.get_traced_memory()
        stages.append({"n_context":n,"instrumented_fit_seconds":fit,"tracked_peak_so_far":peak,
            "reported_predictor_plus_anchor_bytes":meta["persistent_state_bytes"],"observed_context_bytes":x[:n].nbytes+v[:n].nbytes,
            "readout_samples":meta["readout_samples"],"candidates":meta["candidates"],"positive_regions":meta["positive_volume_regions"]})
    current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); info=psutil.Process().memory_info()
    result={"method":args.method,"audit_seed":seed,"stages":stages,"tracemalloc_peak_bytes":peak,"tracemalloc_retained_bytes":current,
        "process_rss_before_bytes":before,"process_rss_after_bytes":info.rss,"process_peak_working_set_bytes":getattr(info,"peak_wset",None),
        "scope":"One prior development stream; tracemalloc can miss native allocations. Process peak includes interpreter/imports. Instrumented times are not comparative benchmark times.",
        "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(model.__file__)]}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"method":args.method,"tracked_peak_bytes":peak,"rss_before":before,"rss_after":info.rss}),flush=True)


if __name__=="__main__":main()

"""Post-confirmation equivalent implementation benchmark; no model selection.

Compare all 256 compiled readout MSEs with the already-frozen direct predictor.
The shared cache uses only the fixed prior function bank, never task data.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import region_posterior_memory as prior
import streaming_branch_projection as base
import compiled_piecewise_readout as spline

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); args=p.parse_args()
protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
original=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
original={(r["seed"],r["n_context"]):r for r in original if r["method"]=="prior4096_noiseaware"}
bank=np.random.default_rng(731).uniform(-.12,.12,(4096,4)); begin=time.perf_counter(); cache=spline.prepare(bank)
prepare=time.perf_counter()-begin; rows=[]
for seed in range(protocol["seed0"],protocol["seed0"]+protocol["count"]):
    rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
    v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
    q=rng.uniform(0,1,2048); target=base.forward(q,truth)
    for n in protocol["stages"]:
        begin=time.perf_counter(); pred,meta=prior.prior_moments(x[:n],v[:n],4,features=4096)
        closure=dict(zip(pred.__code__.co_freevars,(c.cell_contents for c in pred.__closure__)))
        compiled,cm=spline.compile_weights(cache,closure["weights"]); adapt=time.perf_counter()-begin
        begin=time.perf_counter(); prediction=compiled(q); read=time.perf_counter()-begin
        mse=float(np.mean((np.clip(prediction,0,1)-target)**2)); error=abs(mse-original[(seed,n)]["query_mse"])
        assert error<1e-10,error
        rows.append({"seed":seed,"n_context":n,"query_mse":mse,"mse_difference_from_frozen_predictor":error,
            "fit_and_compile_weights_seconds":adapt,"read_2048_seconds":read,**cm})
times=[sum(r["fit_and_compile_weights_seconds"]+r["read_2048_seconds"] for r in rows if r["seed"]==s)
       for s in range(protocol["seed0"],protocol["seed0"]+protocol["count"])]
result={"passed":True,"phase":"post_confirmation_equivalent_implementation_only","stages_checked":len(rows),
    "same_prediction_max_mse_difference":max(r["mse_difference_from_frozen_predictor"] for r in rows),
    "shared_prior_cache_construction_seconds":prepare,"shared_cache_bytes":sum(a.nbytes for a in cache.values()),
    "median_warm_cache_full_stream_fit_plus_query_seconds":float(np.median(times)),
    "cold_cache_estimate_median_seconds":float(np.median(times)+prepare),
    "cold_boundary":"Cold number adds one measured cache construction to each warm-stream cost; cache construction was not repeated 64 times.",
    "benchmark_boundary":"Noisy CPU post-pass, not interleaved with all methods. Exact same estimator; learning hyperparameters and query losses were not retuned.",
    "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(prior.__file__),Path(spline.__file__)]},"rows":rows}
out=args.results.parent/"confirmation_analysis"/"compiled_prior_benchmark.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)

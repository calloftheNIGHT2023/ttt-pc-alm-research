"""One-method, one-stream isolated allocation audit; not speed benchmarking."""
import argparse,hashlib,json,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import stateful_posterior_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--seed',type=int,default=5900000);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());cfg=next(c for c in protocol['configs'] if c['name']==a.method)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    old=json.loads((a.input/'episodes.json').read_text());lookup={r['n_context']:r for r in old if r['method']==a.method and r['seed']==a.seed};assert len(lookup)==4
    rng=np.random.default_rng(a.seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries'])
    v=model.base.forward(x,truth)+np.random.default_rng(a.seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
    target=model.base.forward(q,truth);process=psutil.Process();baseline=process.memory_info()._asdict()
    tracemalloc.start();begin=time.perf_counter();state=None;stages=[]
    for n in protocol['stages']:
        before=0 if state is None else state.anchor.nbytes+state.samples.nbytes
        predict,state,meta=model.fit(x[:n],v[:n],state,dict(**cfg,posterior_samples=protocol['posterior_samples']))
        prediction=predict(q);mse=float(np.mean((prediction-target)**2));r=lookup[n]
        assert mse==r['raw_query_mse'] and np.array_equal(state.anchor,np.array(r['anchor_output']))
        current,peak=tracemalloc.get_traced_memory()
        stages.append(dict(n_context=n,query_mse=mse,original_result_exact=True,tracked_current_bytes=current,tracked_peak_bytes=peak,
            previous_state_bytes=before,persistent_state_bytes=meta['persistent_state_bytes'],
            pool_parameter_bytes=meta['pool_parameter_bytes'],pool_signature_bytes=meta['pool_signature_bytes'],
            geometry_numeric_arrays_subtotal=meta['geometry_numeric_arrays_subtotal'],process_memory=process.memory_info()._asdict()))
    elapsed=time.perf_counter()-begin;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    output=dict(method=a.method,seed=a.seed,source_sha256=protocol['source_sha256'],
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline_process_memory=baseline,
        tracked_final_bytes=current,tracked_peak_bytes=peak,instrumented_elapsed_seconds=elapsed,stages=stages,
        scope='one old stream in separate process; tracemalloc may omit native allocations; process lifetime peaks include imports; instrumented wall time is not a speed result')
    a.out.parent.mkdir(parents=True,exist_ok=True);assert not a.out.exists();a.out.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(method=a.method,tracked_peak_bytes=peak,instrumented_elapsed_seconds=elapsed,all_original_results_exact=True)))


if __name__=='__main__':main()

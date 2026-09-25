"""Isolated single-stream allocation audit of actual completion pipelines."""
import argparse,hashlib,json,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import neighbor_mode_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());cfg=next(c for c in protocol['configs'] if c['name']==a.method)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    seed=5900000;records=json.loads((a.input/'episodes.json').read_text());lookup={r['n_context']:r for r in records if r['method']==a.method and r['seed']==seed};assert len(lookup)==4
    rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
    v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
    process=psutil.Process();baseline=process.memory_info()._asdict();tracemalloc.start();begin=time.perf_counter();state=None;rows=[]
    for n in protocol['stages']:
        oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
        predict,state,meta=model.fit(x[:n],v[:n],state,dict(**cfg,archive=True,pool='posterior_mix',posterior_samples=protocol['posterior_samples'],proposal_budget=protocol['proposal_budget']))
        mse=float(np.mean((predict(q)-target)**2));assert mse==lookup[n]['raw_query_mse'];assert np.array_equal(state.anchor,lookup[n]['anchor_output'])
        assert meta['positive_mode_keys']==lookup[n]['positive_mode_keys']
        current,peak=tracemalloc.get_traced_memory()
        rows.append(dict(n_context=n,original_result_exact=True,tracked_current_bytes=current,tracked_peak_bytes=peak,
            old_plus_new_state_bytes=oldbytes+meta['persistent_state_bytes'],pool_parameter_bytes=meta['pool_parameter_bytes'],pool_signature_bytes=meta['pool_signature_bytes'],
            maximum_neighbor_pattern_array_bytes=meta['maximum_neighbor_pattern_array_bytes'],tested_pattern_key_numeric_bytes=meta['tested_pattern_key_numeric_bytes'],
            geometry_numeric_arrays_subtotal=meta['geometry_numeric_arrays_subtotal'],process_memory=process.memory_info()._asdict()))
    current,peak=tracemalloc.get_traced_memory();elapsed=time.perf_counter()-begin;tracemalloc.stop()
    result=dict(method=a.method,seed=seed,source_sha256=protocol['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_process_memory=baseline,tracked_peak_bytes=peak,instrumented_elapsed_seconds=elapsed,stages=rows,
        scope='single old stream per fresh process; tracemalloc may omit native allocation; process lifetime peak includes imports and audit data; instrumented time not a benchmark')
    a.out.parent.mkdir(parents=True,exist_ok=True);assert not a.out.exists();a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(method=a.method,tracked_peak_bytes=peak,all_original_outputs_exact=True)))


if __name__=='__main__':main()

"""One implementation and task per fresh process; full tracked peak."""
import argparse,hashlib,json,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import deferred_typed_memory as model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--method',required=True);parser.add_argument('--implementation',choices=['eager','deferred'],required=True);args=parser.parse_args()
    p=json.loads((args.input/'protocol.json').read_text());cfg=next(c for c in p['configs'] if c['name']==args.method and c['routing_implementation']==args.implementation)
    for name,value in p['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value,name
    seed=5900001;refs={r['n_context']:r for r in json.loads((args.input/'episodes.json').read_text()) if r['repetition']==0 and r['seed']==seed and r['method']==args.method and r['implementation']==args.implementation}
    rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,p['queries']);target=model.base.forward(q,truth)
    v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
    process=psutil.Process();baseline=process.memory_info()._asdict();fit=model.fit if args.implementation=='deferred' else model.original.fit;state=None;stages=[];tracemalloc.start()
    for n in p['stages']:
        oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
        predict,state,meta=fit(x[:n],v[:n],state,dict(**cfg,archive=True,pool='posterior_mix',posterior_samples=p['posterior_samples'],proposal_budget=p['proposal_budget']))
        assert float(np.mean((predict(q)-target)**2))==refs[n]['raw_query_mse'] and meta['positive_mode_keys']==refs[n]['positive_mode_keys']
        with np.load(args.input/refs[n]['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples'])
        current,peak=tracemalloc.get_traced_memory();details=meta.get('feedback_details',[])
        stages.append(dict(n_context=n,current_bytes=current,peak_bytes=peak,old_plus_new_state_bytes=oldbytes+meta['persistent_state_bytes'],
            pending_key_numeric_bytes=sum(d.get('deferred_pending_key_numeric_bytes',0) for d in details),
            route_key_numeric_bytes=sum(d.get('route_key_numeric_bytes',0) for d in details),
            feedback_library_numeric_bytes=sum(d['library_numeric_bytes'] for d in details),process_memory=process.memory_info()._asdict(),original_outputs_exact=True))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();result=dict(method=args.method,implementation=args.implementation,seed=seed,tracked_peak_bytes=peak,
        baseline_process_memory=baseline,stages=stages,source_sha256=p['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='single task in fresh process; includes Python-tracked router dictionaries and contractor arrays, may omit native allocations; instrumented time is not benchmark')
    (args.input/f'resources_{args.method}_{args.implementation}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(method=args.method,implementation=args.implementation,tracked_peak_bytes=peak,outputs_exact=True)),flush=True)


if __name__=='__main__':main()

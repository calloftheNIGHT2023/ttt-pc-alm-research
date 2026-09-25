"""Fresh-process resource audit on predeclared active-feedback seed 5900001."""
import argparse,hashlib,json,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import conflict_feedback_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);a=p.parse_args();protocol=json.loads((a.input/'protocol.json').read_text());cfg=next(c for c in protocol['configs'] if c['name']==a.method)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    seed=5900001;refs={r['n_context']:r for r in json.loads((a.input/'episodes.json').read_text()) if r['seed']==seed and r['method']==a.method and r['repetition']==0}
    rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth);v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
    proc=psutil.Process();baseline=proc.memory_info()._asdict();state=None;stages=[];tracemalloc.start()
    for n in protocol['stages']:
        oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
        predict,state,meta=model.fit(x[:n],v[:n],state,dict(**cfg,archive=True,pool='posterior_mix',posterior_samples=protocol['posterior_samples'],proposal_budget=protocol['proposal_budget']))
        assert float(np.mean((predict(q)-target)**2))==refs[n]['raw_query_mse'] and meta['positive_mode_keys']==refs[n]['positive_mode_keys']
        with np.load(a.input/refs[n]['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples'])
        current,peak=tracemalloc.get_traced_memory();details=meta.get('feedback_details',[])
        stages.append(dict(n_context=n,current_bytes=current,peak_bytes=peak,old_plus_new_state_bytes=oldbytes+meta['persistent_state_bytes'],
            feedback_library_numeric_bytes=sum(d['library_numeric_bytes'] for d in details),feedback_clauses=sum(d['clauses'] for d in details),
            repairs_triggered=sum(d['repairs_triggered'] for d in details),repairs_accepted=sum(d['repairs_accepted'] for d in details),process_memory=proc.memory_info()._asdict(),original_outputs_exact=True))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();result=dict(method=a.method,seed=seed,tracked_peak_bytes=peak,baseline_process_memory=baseline,stages=stages,
        source_sha256=protocol['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='one predeclared active-support stream per fresh process; tracked peak can omit native allocations; instrumented times not benchmark')
    (a.input/f'resources_{a.method}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(method=a.method,seed=seed,tracked_peak_bytes=peak,original_outputs_exact=True)))


if __name__=='__main__':main()

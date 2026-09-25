"""Fresh-process complete-stream allocation audit, not a timing benchmark."""
import argparse,hashlib,json,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import optimized_credit_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());cfg=next(c for c in protocol['configs'] if c['name']==a.method)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    seed=protocol['seed0'];refs={r['n_context']:r for r in json.loads((a.input/'episodes.json').read_text()) if r['seed']==seed and r['method']==a.method and r['repetition']==0}
    rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth);v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
    proc=psutil.Process();baseline=proc.memory_info()._asdict();state=None;stages=[];tracemalloc.start()
    for n in protocol['stages']:
        oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
        predict,state,meta=model.fit(x[:n],v[:n],state,dict(**cfg,archive=True,pool='posterior_mix',posterior_samples=512,proposal_budget=1024))
        assert float(np.mean((predict(q)-target)**2))==refs[n]['raw_query_mse'] and meta['positive_mode_keys']==refs[n]['positive_mode_keys']
        current,peak=tracemalloc.get_traced_memory();stages.append(dict(n_context=n,current_bytes=current,peak_bytes=peak,old_plus_new_state_bytes=oldbytes+meta['persistent_state_bytes'],normal_cost=meta.get('normal_cost',{}),credit_details=meta.get('credit_details',{}),process_memory=proc.memory_info()._asdict(),original_outputs_exact=True))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    result=dict(method=a.method,seed=seed,tracked_peak_bytes=peak,baseline_process_memory=baseline,stages=stages,source_sha256=protocol['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='fresh single-stream process; tracemalloc can omit native memory; instrumented times excluded from benchmark')
    (a.input/f'resources_{a.method}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(method=a.method,tracked_peak_bytes=peak,original_outputs_exact=True)))


if __name__=='__main__':main()

"""Six fresh process checks of full generation and complete hull solves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
import numpy as np
import psutil
import joint_forward_realization as model
import exact_credit_hull as hull

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int);args=ap.parse_args()
    root=args.project.resolve();base=root/'results/joint_forward_realization';inp=base/'development';out=base/'resources';out.mkdir(parents=True,exist_ok=True);assert read(base/'audit/summary.json')['passed']
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    if args.worker is None:
        for seed in [5920000,5920015]:
            for method in ['generation',*model.METHODS]:
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method,'--seed',str(seed)],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if p.returncode:raise RuntimeError(p.stderr[-6000:])
                print(p.stdout.strip(),flush=True)
        return
    protocol=read(inp/'protocol.json')
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    seed=args.seed;gen=next(r for r in read(inp/'generations.json') if r['seed']==seed);pool=root/f'results/common_pool_credit/development/pool_{seed}.npz';assert sha(pool)==protocol['input_hashes'][str(seed)]['pool']
    with np.load(pool) as z:x=z['x'];v=z['v'];regs=z['regs']
    assert sha(inp/gen['arrays_file'])==gen['arrays_sha256']
    with np.load(inp/gen['arrays_file']) as z:expected={k:z[k] for k in z.files}
    if args.worker!='generation':
        bank=expected['bank']
        if args.worker=='old_plus_constructed_joint':
            path=root/'results/baseline_history/development'/gen['old_bank_source']['arrays_file'];assert sha(path)==gen['old_bank_source']['arrays_sha256']
            with np.load(path) as z:bank=np.concatenate([z['bank'],bank])
        rec=next(r for r in read(inp/'banks.json') if r['seed']==seed and r['method']==args.worker);assert sha(inp/rec['rows_file'])==rec['rows_sha256'];gold=read(inp/rec['rows_file'])[0]['result']
    proc=psutil.Process();before=proc.memory_info();tracemalloc.start();started=time.perf_counter()
    if args.worker=='generation':arrays,generation,meta=model.generate(x,v)
    else:rb=hull.RationalBank(bank);actual=hull.solve(x,v,regs[0],rb)
    elapsed=time.perf_counter()-started;retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    if args.worker=='generation':
        for k,a in expected.items():assert a.shape==arrays[k].shape and a.dtype==arrays[k].dtype and a.tobytes()==arrays[k].tobytes()
        assert generation==read(inp/gen['generation_file']);detail=dict(checked_arrays=len(arrays),points=meta['actual_evaluations'],bank_bytes=meta['bank_bytes'],diagnostic_array_bytes=meta['diagnostic_array_bytes'])
        scope='Full inverse generation, all joint forwards, normalization and source checks; expected arrays preloaded; instrumented time not benchmark'
    else:
        assert actual['status']==gold['status']
        if actual['status']=='positive':assert actual['weights']==gold['weights'] and actual['credit']==gold['credit'] and actual['dual']['proof']==gold['dual']['proof']
        elif actual['status']=='nonpositive':assert actual['witness']==gold['witness']
        detail=dict(index=0,status=actual['status'],directions=len(bank),bank_bytes=bank.nbytes,exact_output_reproduced=True);scope='Rational preparation plus complete one-region solve; inputs preloaded; not online peak'
    result=dict(passed=True,seed=seed,method=args.worker,**detail,python_traced_peak_bytes=peak,python_traced_retained_bytes=retained,rss_before_bytes=before.rss,rss_after_bytes=after.rss,
        process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),instrumented_seconds_not_benchmark=elapsed,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),scope=scope)
    target=out/f'{seed}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

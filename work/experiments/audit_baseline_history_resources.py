"""Fresh-process full-capture and solver allocation audits, no speedup claim."""
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
import baseline_history_capture as history
import exact_credit_hull as hull


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def load_bank(root,source):
    if 'parts' in source:return np.concatenate([load_bank(root,r) for r in source['parts']])
    path=root/source['file'];assert sha(path)==source['sha256']
    with np.load(path) as z:return z['bank']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--phase',choices=['capture','solver']);ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();base=root/'results/baseline_history';inp=base/'development';out=base/'resources';out.mkdir(parents=True,exist_ok=True)
    assert read(base/'audit/summary.json')['passed'];p=read(inp/'protocol.json');assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    if args.worker is None:
        for seed in [5920000,5920015]:
            for phase,methods in [('capture',history.LEARNERS),('solver',history.VARIANTS)]:
                for method in methods:
                    process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method,'--phase',phase,'--seed',str(seed)],
                        capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if process.returncode:raise RuntimeError(process.stderr[-6000:])
                    print(process.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    pool=root/f'results/common_pool_credit/development/pool_{args.seed}.npz';assert sha(pool)==p['input_hashes'][str(args.seed)]['pool']
    with np.load(pool) as z:x=z['x'];v=z['v'];regs=z['regs']
    if args.phase=='capture':
        cap=next(r for r in read(inp/'captures.json') if r['seed']==args.seed and r['learner']==args.worker)
        assert sha(inp/cap['arrays_file'])==cap['arrays_sha256']
        with np.load(inp/cap['arrays_file']) as z:expected={k:z[k] for k in z.files}
    else:
        rec=next(r for r in read(inp/'banks.json') if r['seed']==args.seed and r['method']==args.worker);bank=load_bank(inp,rec['bank_source'])
        assert sha(inp/rec['rows_file'])==rec['rows_sha256'];gold=read(inp/rec['rows_file'])[0]['result']
    proc=psutil.Process();before=proc.memory_info();tracemalloc.start();started=time.perf_counter()
    if args.phase=='capture':
        old,arrays,meta=history.capture(x,v,args.worker,True)
    else:rb=hull.RationalBank(bank);result=hull.solve(x,v,regs[0],rb)
    elapsed=time.perf_counter()-started;retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    if args.phase=='capture':
        for key,value in expected.items():assert value.tobytes()==arrays[key].tobytes()
        assert meta['trajectory_sha256']==cap['actual']['trajectory_sha256']
        detail=dict(checked_arrays=len(arrays),directions=meta['directions'],raw_bytes=meta['raw_bytes'],bank_bytes=meta['bank_bytes'],diagnostic_array_bytes=meta['diagnostic_array_bytes'],trajectory_events=meta['trace_events'],evaluation_calls=meta['evaluation_calls'])
        scope='Entire original discovery plus read-only collection, normalization and inclusion checks; expected arrays loaded before trace; instrumented time is not a benchmark'
    else:
        assert result['status']==gold['status']
        if result['status']=='positive':assert result['dual']['proof']==gold['dual']['proof'] and result['weights']==gold['weights'] and result['credit']==gold['credit']
        elif result['status']=='nonpositive':assert result['witness']==gold['witness']
        detail=dict(index=0,directions=len(bank),bank_bytes=bank.nbytes,status=result['status'],exact_output_reproduced=True)
        scope='Rational bank preparation plus one complete primal/dual diagnostic solve; loaded inputs excluded; not online peak or performance'
    answer=dict(passed=True,seed=args.seed,phase=args.phase,method=args.worker,**detail,python_traced_peak_bytes=peak,python_traced_retained_bytes=retained,
        rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),instrumented_seconds_not_benchmark=elapsed,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),scope=scope)
    target=out/f'{args.seed}_{args.phase}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(answer,indent=2),encoding='utf-8');print(json.dumps(answer),flush=True)


if __name__=='__main__':main()

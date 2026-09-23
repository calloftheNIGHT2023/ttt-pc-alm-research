"""Sequential fresh-process resource audit; instrumented time is not a benchmark."""
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
from verify_credit_moment_step import guarded_solve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();base=root/'results/credit_moment_step';inp=base/'development';out=base/'resources'
    out.mkdir(parents=True,exist_ok=True);p=json.loads((inp/'protocol.json').read_text())
    assert json.loads((base/'audit/summary.json').read_text())['passed']
    assert json.loads((base/'timing/summary.json').read_text())['passed'], 'Finish sole timing job first'
    assert os.environ.get('OPENBLAS_NUM_THREADS')=='1' and os.environ.get('OMP_NUM_THREADS')=='1'
    if args.worker is None:
        for seed in [5920000,5920015]:
            for method in p['methods']:
                command=[sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method,'--seed',str(seed)]
                proc=subprocess.run(command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if proc.returncode:raise RuntimeError(proc.stderr[-5000:])
                print(proc.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rec=next(r for r in json.loads((inp/'banks.json').read_text()) if r['seed']==args.seed and r['method']==args.worker)
    source=root/'results/light_h2_credit/full_bank_ceiling'/rec['source_data_file'];assert sha(source)==rec['source_data_sha256']
    with np.load(source) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
    assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
    with np.load(inp/rec['arrays_file']) as z:arrays={key:z[key] for key in z.files}
    meta=json.loads((inp/rec['meta_file']).read_text());proc=psutil.Process();before=proc.memory_info();tracemalloc.start();start=time.perf_counter()
    again,am=guarded_solve(x,v,regs[arrays['indices']],bank)
    elapsed=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    for key,value in again.items():assert value.tobytes()==arrays[key].tobytes()
    assert am['proofs']==meta['proofs']
    result=dict(passed=True,seed=args.seed,method=args.worker,replayed_regions=len(arrays['indices']),replayed_arrays=len(again),oracle_pairs=am['oracle_pairs'],
        mean_evaluations=am['mean_evaluations'],variance_evaluations=am['variance_evaluations'],positive=am['positive'],
        python_traced_peak_bytes=peak,python_traced_retained_bytes=current,rss_before_bytes=before.rss,rss_after_bytes=after.rss,
        process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),instrumented_seconds_not_benchmark=elapsed,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Same two representative tasks as 224; guard and proof paths included; input/import baseline already resident; not full online peak')
    target=out/f'{args.seed}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Fresh-process common-pool state accounting after main timing is finished."""
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
import common_pool_credit as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();base=root/'results/common_pool_credit';inp=base/'development';out=base/'resources'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((base/'audit/summary.json').read_text())['passed'];out.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    if args.worker is None:
        for seed in [5920000,5920015]:
            for method in p['methods']:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method,'--seed',str(seed)],
                    capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if proc.returncode:raise RuntimeError(proc.stderr[-5000:])
                print(proc.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rec=next(r for r in json.loads((inp/'banks.json').read_text()) if r['seed']==args.seed and r['method']==args.worker)
    source=root/'results/light_h2_credit/full_bank_ceiling'/rec['source_file'];assert sha(source)==rec['source_sha256']
    assert sha(inp/rec['pool_file'])==rec['pool_sha256']
    with np.load(source) as z:bank=z['bank']
    with np.load(inp/rec['pool_file']) as z:x=z['x'];v=z['v'];regs=z['regs']
    assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
    with np.load(inp/rec['arrays_file']) as z:expected={key:z[key] for key in z.files}
    meta=json.loads((inp/rec['meta_file']).read_text());proc=psutil.Process();before=proc.memory_info();tracemalloc.start();start=time.perf_counter()
    actual,am=model.solve(x,v,regs,bank);elapsed=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    for key,value in actual.items():assert value.tobytes()==expected[key].tobytes()
    assert am['old_proofs']==meta['old_proofs'] and am['proofs']==meta['proofs']
    result=dict(passed=True,seed=args.seed,method=args.worker,pool_regions=len(regs),replayed_joint_regions=len(actual['indices']),replayed_arrays=len(actual),
        old=am['old_count'],new=am['positive'],total=am['total_positive'],bank_bytes=bank.nbytes,python_traced_peak_bytes=peak,
        python_traced_retained_bytes=current,rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
        instrumented_seconds_not_benchmark=elapsed,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Screen construction plus old screen plus joint solve, imports and source pool already resident; not full online peak')
    target=out/f'{args.seed}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

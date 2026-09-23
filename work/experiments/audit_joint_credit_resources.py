"""Fresh-process diagnostic resource cost; includes the LP solver in RSS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
import numpy as np
import psutil
import joint_credit_minimax as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();inp=root/'results/joint_credit_minimax/development';out=root/'results/joint_credit_minimax/resources';out.mkdir(parents=True,exist_ok=True)
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'summary.json').read_text())['execution_complete']
    if args.worker is None:
        for seed in [5920000,5920015]:
            for method in p['methods']:
                cmd=[sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method,'--seed',str(seed)]
                process=subprocess.run(cmd,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if process.returncode:raise RuntimeError(process.stderr[-5000:])
                print(process.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rec=next(r for r in json.loads((inp/'banks.json').read_text()) if r['seed']==args.seed and r['method']==args.worker)
    source=root/'results/light_h2_credit/full_bank_ceiling'/rec['source_data_file'];assert sha(source)==rec['source_data_sha256']
    with np.load(source) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
    assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['rows_file'])==rec['rows_sha256']
    with np.load(inp/rec['arrays_file']) as z:credits=z['credit'];weights=z['weights'];primal=z['primal']
    rows=json.loads((inp/rec['rows_file']).read_text());proc=psutil.Process();before=proc.memory_info();tracemalloc.start();start=time.perf_counter()
    checks=0;proofs=0
    for i,row in enumerate(rows):
        if row['old_positive']:continue
        data,meta=model.solve(x,v,regs[i],bank);assert meta['success']==row['joint']['success']
        if data is not None:
            assert data['credit'].tobytes()==credits[i].tobytes() and data['weights'].tobytes()==weights[i].tobytes() and data['primal'].tobytes()==primal[i].tobytes()
            assert meta['proof']==row['joint']['proof'];checks+=1;proofs+=int(meta['positive'])
    elapsed=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    result=dict(passed=True,seed=args.seed,method=args.worker,lp_replays=checks,positive_proofs=proofs,python_traced_peak_bytes=peak,
        python_traced_retained_bytes=current,rss_before_bytes=before.rss,rss_after_bytes=after.rss,
        process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),instrumented_seconds_not_benchmark=elapsed,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),scope='Two representative support tasks; native LP allocations included by OS RSS, not all-task worst-case or full online cost')
    target=out/f'{args.seed}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

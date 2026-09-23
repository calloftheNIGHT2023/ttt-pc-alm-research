"""Fresh-process replay-state audit, without perturbing live stopping times."""
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
import credit_wall_budget as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();base=root/'results/credit_wall_budget';inp=base/'development';out=base/'resources'
    p=read(inp/'protocol.json');assert read(base/'audit/summary.json')['passed'];out.mkdir(parents=True,exist_ok=True)
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
    row=next(r for r in read(inp/'rows.json') if r['seed']==args.seed and r['method']==args.worker and r['budget_seconds']==.4 and r['repeat']==0)
    source=p['input_hashes'][str(args.seed)]['banks'][args.worker];path=root/source['file'];assert sha(path)==source['sha256']
    with np.load(path) as z:bank=z[args.worker if args.worker.startswith('history_') else 'bank']
    pool=root/f'results/common_pool_credit/development/pool_{args.seed}.npz';assert sha(pool)==p['input_hashes'][str(args.seed)]['pool']
    with np.load(pool) as z:x=z['x'];v=z['v'];regs=z['regs']
    assert sha(inp/row['arrays_file'])==row['arrays_sha256'] and sha(inp/row['meta_file'])==row['meta_sha256']
    with np.load(inp/row['arrays_file']) as z:expected={key:z[key] for key in z.files}
    meta=read(inp/row['meta_file']);proc=psutil.Process();before=proc.memory_info();tracemalloc.start();start=time.perf_counter()
    actual,am=model.guarded_solve(x,v,regs,bank,budget=.4,max_steps=4096,replay_events=meta['events'])
    elapsed=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    for key,value in expected.items():assert value.tobytes()==actual[key].tobytes()
    assert am['proofs']==meta['proofs'] and am['old_proofs']==meta['old_proofs'] and am['events']==meta['events']
    result=dict(passed=True,seed=args.seed,method=args.worker,budget_seconds=.4,repeat=0,replayed_arrays=len(actual),pool_regions=len(regs),
        old=am['old_count'],new=am['new_count'],total=am['total_positive'],response_batches=am['response_batches'],oracle_pairs=am['oracle_pairs'],
        bank_bytes=bank.nbytes,output_array_bytes=sum(v.nbytes for v in actual.values()),python_replay_traced_peak_bytes=peak,python_replay_traced_retained_bytes=current,
        rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
        instrumented_replay_seconds_not_benchmark=elapsed,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Fixed original-work replay, including solver allocations; inputs/events already loaded and replay reuses timestamp floats; not exact live-run or online peak')
    target=out/f'{args.seed}_{args.worker}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

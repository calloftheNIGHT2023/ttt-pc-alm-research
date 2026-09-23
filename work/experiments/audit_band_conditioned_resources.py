"""Twenty-two predetermined fresh CPU processes, separate from primary timing."""
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
import torch
import run_band_conditioned_online as runner

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

NAMES=['band_inverse_alm_native_gate','band_inverse_alm_c5','band_inverse_adam60_c5','band_inverse_adam240_c5','band_inverse_gn20_c5',
    'band_inverse_pc_c5','band_inverse_nodual_c5','band_inverse_direct64_c5','prior4096_ridge','meta_ridge128','meta_shallow64_20']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int);args=ap.parse_args()
    root=args.project.resolve();base=root/'results/band_conditioned_online';inp=base/'development';out=base/'resources';out.mkdir(parents=True,exist_ok=True)
    p=read(inp/'protocol.json');assert read(inp/'run_audit.json')['execution_complete'] and read(base/'audit/summary.json')['passed']
    if args.worker is None:
        for seed in [5920000,5920015]:
            for name in NAMES:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',name,'--seed',str(seed)],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if proc.returncode:raise RuntimeError(proc.stderr[-6000:])
                print(proc.stdout.strip(),flush=True)
        return
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name,h in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==h,name
    assert args.worker in NAMES and args.seed in [5920000,5920015];seed=args.seed;cfg=next(c for c in p['configs'] if c['name']==args.worker)
    old=sorted([r for r in read(inp/'rows.json') if r['seed']==seed and r['method']==cfg['name'] and r['repetition']==0],key=lambda r:r['n'])
    ep=next(e for e in read(inp/'episodes.json') if e['seed']==seed and e['method']==cfg['name'] and e['repetition']==0)
    loaded,manifest=runner.meta_model.load(root);assert manifest==p['checkpoint_manifest'];xx,vv=runner.model.original.previous.olddriver.observations(seed);q=np.linspace(0,1,257)
    proc=psutil.Process();before=proc.memory_info();tracemalloc.start();start=time.perf_counter();rows,arrays,failure=runner.trial(xx,vv,q,cfg,seed,0,p,loaded)
    seconds=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=proc.memory_info()
    assert (failure is None)==ep['complete'] and len(rows)==len(old)
    for row,a,ref in zip(rows,arrays,old):
        assert row['n']==ref['n'] and sha(inp/ref['state_file'])==ref['state_sha256']
        with np.load(inp/ref['state_file']) as z:assert set(a)==set(z.files) and all(a[k].tobytes()==z[k].tobytes() for k in a)
        assert bool(row['recovery'] and row['recovery']['triggered'])==bool(ref['recovery'] and ref['recovery']['triggered'])
    result=dict(passed=True,seed=seed,method=cfg['name'],stage_replays=len(rows),complete=failure is None,recovery_stages=sum(bool(r['recovery'] and r['recovery']['triggered']) for r in rows),
        python_traced_peak_bytes=peak,python_traced_retained_bytes=current,rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
        instrumented_seconds_not_benchmark=seconds,last_state_bytes=rows[-1]['state_bytes'] if rows else None,shared_model_bytes=max((r['shared_model_bytes'] for r in rows),default=0),
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Full instrumented stream includes retained diagnostic arrays and support checks; models preloaded; all five checkpoints resident in RSS; not isolated deployment or worst-case native allocation')
    target=out/f'{seed}_{cfg["name"]}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

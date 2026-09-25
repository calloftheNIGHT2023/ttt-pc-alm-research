"""Fresh-process resource checks for two preselected tasks and every method."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import psutil
import run_recovered_online_comparison as prior


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');ap.add_argument('--seed',type=int)
    args=ap.parse_args();root=args.project.resolve();inp=root/'results/context_block_scaling/development'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    out=inp.parent/'resources';out.mkdir(parents=True,exist_ok=True)
    if args.worker is None:
        for seed in [5920001,5920015]:
            for cfg in p['configs']:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',cfg['name'],'--seed',str(seed)],
                    capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if proc.returncode:raise RuntimeError(proc.stderr[-6000:])
                print(proc.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    cfg=next(c for c in p['configs'] if c['name']==args.worker);seed=args.seed
    rows=[r for r in json.loads((inp/'rows.json').read_text()) if r['seed']==seed and r['method']==cfg['name'] and r['repetition']==0]
    xx,vv=prior.olddriver.observations(seed);q=np.linspace(0,1,p['query_points']);records=[];process=psutil.Process()
    for n in p['block_sizes']:
        row=next(r for r in rows if r['n']==n);rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,0,n,p['posterior_samples']]))
        before=process.memory_info();tracemalloc.start();start=time.perf_counter();complete=True;arrays=None
        try:
            if cfg['family']=='regression':predict,state,meta=prior.regression(xx[:n],vv[:n],cfg['name'],None)
            else:predict,state,meta=prior.recovery.fit(xx[:n],vv[:n],None,cfg,rng,p['posterior_samples'])
            arrays=dict(x=xx[:n],v=vv[:n],q=q,prediction=predict(q))
            if cfg['family']!='regression':arrays.update(points=state.samples,anchor=state.anchor)
            elif state is not None:arrays.update(rls_w=state['w'],rls_p=state['p'])
        except prior.recovery.RecoveryExhausted:complete=False
        elapsed=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()
        assert complete==row['complete']
        if complete:
            path=inp/row['state_file'];assert sha(path)==row['state_sha256']
            with np.load(path) as z:assert set(arrays)==set(z.files) and all(arrays[k].tobytes()==z[k].tobytes() for k in arrays)
        records.append(dict(n=n,complete=complete,python_traced_peak_bytes=peak,python_traced_retained_bytes=current,
            rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
            instrumented_seconds_not_benchmark=elapsed,persistent_state_bytes=row.get('state_bytes')))
        arrays=None
        if complete:del predict,state,meta
    result=dict(passed=True,method=cfg['name'],seed=seed,blocks=records,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Fresh process per method/task; four cold blocks sequentially release state; excludes file IO; two representative tasks not all-task maximum')
    (out/f'{seed}_{cfg["name"]}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,method=cfg['name'],seed=seed,blocks=len(records),failures=sum(not r['complete'] for r in records))),flush=True)


if __name__=='__main__':main()

"""Separate fresh-process peak tracking; never used as benchmark timings."""
import argparse,hashlib,json,subprocess,sys,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import rejection_memory_materialization as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker',choices=memory.METHODS);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/rejection_materialization/development';out=root/'results/rejection_materialization/resources';out.mkdir(parents=True,exist_ok=True)
    if args.worker is None:
        for method in memory.METHODS:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',method],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if proc.returncode:raise RuntimeError(proc.stderr[-6000:])
            print(proc.stdout.strip(),flush=True)
        return
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    method=args.worker;seed=5900001;count=2048;rep=0
    row=next(r for r in json.loads((inp/'rows.json').read_text()) if r['seed']==seed and r['method']==method and r['samples']==count and r['repetition']==rep)
    state=inp/row['state_file'];assert sha(state)==row['state_sha256']
    with np.load(root/f'results/rejection_credit/diagnostic/proposal_{seed}.npz') as z:x=z['x'];v=z['v']
    q=np.linspace(0,1,p['query_points']);rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,count]))
    process=psutil.Process();baseline=process.memory_info();tracemalloc.start();begin=time.perf_counter()
    data,meta=memory.prepare(x,v,p['config'],method);points,labels,note=memory.sample(data,count,rng,p['proposal_batch'],p['maximum_proposals']);prediction=memory.predict(points,q)
    instrumented_seconds=time.perf_counter()-begin;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()
    with np.load(state) as z:assert np.array_equal(points,z['points']) and np.array_equal(labels,z['labels']) and np.array_equal(prediction,z['prediction'])
    result=dict(passed=True,method=method,seed=seed,samples=count,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        original_state_sha256=row['state_sha256'],state_and_prediction_bitwise=True,python_traced_peak_bytes=peak,python_traced_retained_bytes=current,
        rss_before_bytes=baseline.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
        instrumented_seconds_not_benchmark=instrumented_seconds,particle_state_bytes=points.nbytes,
        scope='fresh process, representative seed 5900001 at 2048 particles; tracemalloc includes NumPy-tracked allocations, process peak includes imports; not all-task worst case')
    (out/f'{method}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

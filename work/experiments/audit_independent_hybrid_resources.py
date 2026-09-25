"""Nineteen serial fresh-process online resource audits, after main timing."""
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
import run_independent_hybrid_memory as runner


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';out=root/'results/independent_hybrid/resources'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    assert run['execution_complete']
    out.mkdir(parents=True,exist_ok=True)
    if args.worker is None:
        for cfg in p['configs']:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',cfg['name']],
                capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if proc.returncode:raise RuntimeError(proc.stderr[-6000:])
            print(proc.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    cfg=next(c for c in p['configs'] if c['name']==args.worker);seed=5920001;rep=0
    rows=[r for r in json.loads((inp/'rows.json').read_text()) if r['seed']==seed and r['method']==cfg['name'] and r['repetition']==rep]
    assert len(rows)==4,'Representative resource task must have a complete original trial'
    xx,vv=runner.observations(seed);q=np.linspace(0,1,p['query_points']);process=psutil.Process();before=process.memory_info()
    tracemalloc.start();begin=time.perf_counter();actual,artifacts=runner.trial(xx,vv,q,cfg,seed,rep,p)
    seconds=time.perf_counter()-begin;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()
    checks=0
    for row,arrays in zip(actual,artifacts):
        old=next(r for r in rows if r['n_context']==row['n_context']);path=inp/old['state_file'];assert sha(path)==old['state_sha256']
        with np.load(path) as z:
            assert set(z.files)==set(arrays)
            for key,value in arrays.items():assert np.array_equal(z[key],value),(cfg['name'],row['n_context'],key)
        assert row['new_state_digest']==old['new_state_digest'] and row['previous_state_digest']==old['previous_state_digest'];checks+=1
    result=dict(passed=True,method=cfg['name'],seed=seed,repetition=rep,stage_states_and_predictions_bitwise=checks,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        python_traced_peak_bytes=peak,python_traced_retained_bytes=current,rss_before_bytes=before.rss,rss_after_bytes=after.rss,
        process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),instrumented_seconds_not_benchmark=seconds,
        last_persistent_state_bytes=actual[-1]['persistent_state_bytes'],
        scope='fresh process, full four-stage trial at seed 5920001; includes in-memory audit artifacts, excludes file compression; not task worst case or benchmark timing; RSS lifetime peak includes imports')
    (out/f'{cfg["name"]}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

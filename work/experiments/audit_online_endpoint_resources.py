"""Serial representative fresh-process audits for all 35 configurations."""
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
import run_light_h2_credit as runner
import online_endpoint_h2 as memory
runner.memory=memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--worker');args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/online_endpoint_h2/development';out=root/'results/online_endpoint_h2/resources'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    out.mkdir(parents=True,exist_ok=True)
    if args.worker is None:
        for cfg in p['configs']:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(root),'--worker',cfg['name']],
                capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if proc.returncode:raise RuntimeError(proc.stderr[-6000:])
            print(proc.stdout.strip(),flush=True)
        return
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    seed=5920001;rep=0;cfg=next(c for c in p['configs'] if c['name']==args.worker)
    old=[r for r in json.loads((inp/'rows.json').read_text()) if r['seed']==seed and r['method']==cfg['name'] and r['repetition']==rep];assert len(old)==4
    xx,vv=runner.observations(seed);q=np.linspace(0,1,257);process=psutil.Process();before=process.memory_info()
    tracemalloc.start();start=time.perf_counter();rows,artifacts,failure=runner.trial(xx,vv,q,cfg,seed,rep,p)
    seconds=time.perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()
    assert failure is None
    for row,arrays in zip(rows,artifacts):
        ref=next(r for r in old if r['n']==row['n']);path=inp/ref['state_file'];assert sha(path)==ref['state_sha256']
        with np.load(path) as z:
            assert set(z.files)==set(arrays)
            for key,value in arrays.items():assert np.array_equal(z[key],value),(cfg['name'],row['n'],key)
    result=dict(passed=True,method=cfg['name'],seed=seed,stage_replays=4,python_traced_peak_bytes=peak,python_traced_retained_bytes=current,
        rss_before_bytes=before.rss,rss_after_bytes=after.rss,process_lifetime_peak_working_set_bytes=getattr(after,'peak_wset',None),
        instrumented_seconds_not_benchmark=seconds,last_state_bytes=rows[-1]['state_bytes'],source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(inp/'protocol.json'),scope='representative four-stage fresh process; includes audit arrays, excludes file compression; not all-task worst case or benchmark timing')
    (out/f'{cfg["name"]}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

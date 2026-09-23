"""All20 methods, untraced randomized timing; memory separate from timing."""
import argparse
import gc
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import psutil
import torch
import run_cold_stagnation_switch as run
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/cold_stagnation_switch';inp=base/'development';out=base/'resources';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);old=json.loads((inp/'protocol.json').read_text());audit=json.loads((base/'audit/summary.json').read_text());assert audit['passed']
    for n,h in old['source_sha256'].items():assert sha(src/n)==h,n
    loaded,manifest=run.meta.load(root);assert old['checkpoint_manifest']=={n:manifest[n] for n in old['checkpoint_manifest']}
    p=dict(source_sha256=sha(Path(__file__)),experiment_protocol_sha256=sha(inp/'protocol.json'),audit_sha256=sha(base/'audit/summary.json'),
        timing_repetitions=3,order_seed=247919,memory_seeds=[5900000,5900015],configs=old['configs'],timing_seeds=old['seeds'],
        memory_scope='fresh process; all5 frozen meta models loaded; reset tracemalloc per fit; RSS not isolated model deployment peak; native Torch allocations not fully traced',
        timing_scope='no trajectory trace, no geometry, no extra baseline replay; same prediction; shared model load and warmup separately disclosed')
    dump(out/'protocol.json',p);q=np.linspace(0,1,257);checks=0;timings=[];memory=[];proc=psutil.Process();begin=time.perf_counter()
    with run.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=run.observations(5900001);warm=time.perf_counter()
        for cfg in p['configs']:run.fit(cfg,x[:4],v[:4],q,loaded,False)
        warm=time.perf_counter()-warm;rng=np.random.default_rng(p['order_seed']);jobs=[(s,c,r) for s in p['timing_seeds'] for c in p['configs'] for r in range(3)]
        for index in rng.permutation(len(jobs)):
            seed,cfg,rep=jobs[index];x,v=run.observations(seed);arr,row=run.fit(cfg,x[:4],v[:4],q,loaded,False)
            expected=np.load(inp/f'{seed}_{cfg["name"]}.npz');assert arr['prediction'].tobytes()==expected['prediction'].tobytes();checks+=1
            if cfg['family'] in ['local','bp']:assert arr['selected_b'].tobytes()==expected['selected_b'].tobytes()
            timings.append(dict(seed=seed,method=cfg['name'],rep=rep,write_seconds=row['write_seconds'],read_seconds=row['read_seconds'],total_seconds=row['total_seconds'],
                detection_seconds=row['metadata'].get('detection_seconds',0)))
            if len(timings)%160==0:print(json.dumps(dict(timed=len(timings),total=len(jobs))),flush=True)
        dump(out/'timings.json',timings)
        for seed in p['memory_seeds']:
            x,v=run.observations(seed)
            for cfg in p['configs']:
                gc.collect();before=proc.memory_info().rss;tracemalloc.start();arr,row=run.fit(cfg,x[:4],v[:4],q,loaded,False);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                after=proc.memory_info().rss;expected=np.load(inp/f'{seed}_{cfg["name"]}.npz');assert arr['prediction'].tobytes()==expected['prediction'].tobytes();checks+=1
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,rss_before=before,rss_after=after,
                    output_numeric_bytes=sum(v.nbytes for v in arr.values()),metadata=row['metadata']))
        dump(out/'memory.json',memory)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),bytewise_predictions=checks,timing_runs=len(timings),memory_runs=len(memory),
        warmup_seconds=warm,total_seconds=time.perf_counter()-begin,output_sha256={n:sha(out/n) for n in ['timings.json','memory.json']},
        methods={c['name']:dict(mean_seconds=float(np.mean([r['total_seconds'] for r in timings if r['method']==c['name']])),median_seconds=float(np.median([r['total_seconds'] for r in timings if r['method']==c['name']])),
            maximum_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name'])) for c in p['configs']})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

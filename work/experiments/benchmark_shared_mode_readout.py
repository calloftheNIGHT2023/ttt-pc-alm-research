"""Real full pipeline timings, all10 methods; no free prior diagnostic cache."""
import argparse
import gc
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import psutil
import live_shared_mode_readout as live
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/shared_mode_readout';inp=base/'development';out=base/'resources';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((inp/'protocol.json').read_text());audit=json.loads((base/'audit/summary.json').read_text());assert audit['passed']
    for n,h in p['source_sha256'].items():assert sha(src/n)==h,n
    hashes=dict(p['source_sha256']);hashes.update({n:sha(src/n) for n in ['audit_shared_mode_readout.py','live_shared_mode_readout.py',Path(__file__).name]})
    configs=[c for c in live.cold.configs() if c['family'] in ['local','bp']]
    protocol=dict(source_sha256=hashes,experiment_protocol_sha256=sha(inp/'protocol.json'),audit_sha256=sha(base/'audit/summary.json'),
        timing_seeds=p['seeds'],repetition_seed=249911,order_seed=249929,memory_seeds=[5900000,5900015],configs=configs,
        scope='full cold generation + every visited-mode LP + own positive-volume geometry/interior proof +2048 sampling +257 query read; no external cache')
    dump(out/'protocol.json',protocol);q=np.linspace(0,1,257);rows=[];memory=[];proc=psutil.Process();begin=time.perf_counter();checks=0
    def check(seed,cfg,a,m):
        old=np.load(inp/f'state_{seed}_{cfg["name"]}_0.npz')
        for key in ['points','prediction','allocation']:assert old[key].tobytes()==a[key].tobytes(),(seed,cfg['name'],key)
        frozen=np.load(root/'results/cold_stagnation_switch/development'/f'{seed}_{cfg["name"]}.npz');assert frozen['selected_b'].tobytes()==a['selected_b'].tobytes()
    with live.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=observations(5900001);start=time.perf_counter()
        for cfg in configs:
            a,m=live.fit(cfg,x[:4],v[:4],q,5900001);check(5900001,cfg,a,m)
        warm=time.perf_counter()-start;jobs=[(s,c) for s in p['seeds'] for c in configs];order=np.random.default_rng(protocol['order_seed']).permutation(len(jobs))
        for index in order:
            seed,cfg=jobs[index];x,v=observations(seed);a,m=live.fit(cfg,x[:4],v[:4],q,seed);check(seed,cfg,a,m);checks+=1;rows.append(dict(seed=seed,method=cfg['name'],**m))
            if len(rows)%20==0:print(json.dumps(dict(complete=len(rows),total=len(jobs))),flush=True)
        dump(out/'timings.json',rows)
        for seed in protocol['memory_seeds']:
            x,v=observations(seed)
            for cfg in configs:
                gc.collect();rss=proc.memory_info().rss;tracemalloc.start();a,m=live.fit(cfg,x[:4],v[:4],q,seed);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();check(seed,cfg,a,m);checks+=1
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,rss_before=rss,rss_after=proc.memory_info().rss,metadata=m))
        dump(out/'memory.json',memory)
    result=dict(passed=True,timing_runs=len(rows),memory_runs=len(memory),bytewise_predictor_replays=checks,warmup_seconds=warm,total_seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),outputs_sha256={n:sha(out/n) for n in ['timings.json','memory.json']},
        methods={c['name']:dict(mean_seconds=float(np.mean([r['total_seconds'] for r in rows if r['method']==c['name']])),
            mean_discovery_seconds=float(np.mean([r['discovery_seconds'] for r in rows if r['method']==c['name']])),
            mean_geometry_seconds=float(np.mean([r['geometry_seconds'] for r in rows if r['method']==c['name']])),
            mean_sampling_read_seconds=float(np.mean([r['sampling_seconds']+r['read_seconds'] for r in rows if r['method']==c['name']])),
            max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name'])) for c in configs},
        memory_scope='one fresh process; tracemalloc excludes some native workspaces; two fixed tasks not worst-case bound; no RSS superiority claim')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

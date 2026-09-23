"""256 all33 complete fresh pipelines; timing and trace memory kept separate."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import tracemalloc
import numpy as np
import psutil
import torch
import dual_jump_resource_suite as suite
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/dual_jump_query';out=base/'resources';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert json.loads((base/'live_primitive/summary.json').read_text())['passed'];old=json.loads((base/'live_primitive/protocol.json').read_text());hashes=dict(old['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    start=time.perf_counter();loaded,manifest=suite.oldfit.meta.load(root);loading=time.perf_counter()-start;assert manifest==old['checkpoint_manifest']
    p=dict(source_sha256=hashes,primitive_summary_sha256=sha(base/'live_primitive/summary.json'),configs=suite.configs(),checkpoint_manifest=manifest,
        seeds=list(range(5900000,5900016)),memory_seeds=[5900000,5900015],order_seed=256929,repetition_seed=249911,query_points=257,
        scope='fresh no-cache complete pipelines;13 warm mechanisms include actual cold nodual16+full scan if needed+atomic+continuation;10 old solvers;10 heads',
        model_loading_seconds=loading,query_targets_accessed=False)
    dump(out/'protocol.json',p);q=np.linspace(0,1,257);rows=[];memory=[];proc=psutil.Process();begin=time.perf_counter()
    with suite.live.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=observations(5900001);start=time.perf_counter()
        for cfg in p['configs']:
            a,m=suite.fit(cfg,x[:4],v[:4],q,5900001,loaded);suite.check(root,5900001,cfg,a,m)
        warm=time.perf_counter()-start;jobs=[(s,c) for s in p['seeds'] for c in p['configs']]
        for index in np.random.default_rng(p['order_seed']).permutation(len(jobs)):
            seed,cfg=jobs[index];x,v=observations(seed);a,m=suite.fit(cfg,x[:4],v[:4],q,seed,loaded);suite.check(root,seed,cfg,a,m);rows.append(dict(seed=seed,method=cfg['name'],family=cfg['family'],metadata=m,seconds=m['complete_call_seconds']))
            if len(rows)%33==0:dump(out/'timings.json',rows);print(json.dumps(dict(timing_runs=len(rows),total=len(jobs))),flush=True)
        dump(out/'timings.json',rows)
        for seed in p['memory_seeds']:
            x,v=observations(seed)
            for cfg in p['configs']:
                gc.collect();rss=proc.memory_info().rss;tracemalloc.start();a,m=suite.fit(cfg,x[:4],v[:4],q,seed,loaded);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();suite.check(root,seed,cfg,a,m)
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,rss_before=rss,rss_after=proc.memory_info().rss,metadata=m))
                if len(memory)%11==0:dump(out/'memory.json',memory);print(json.dumps(dict(memory_runs=len(memory),total=66)),flush=True)
        dump(out/'memory.json',memory)
    for n,h in hashes.items():assert sha(src/n)==h,n
    result=dict(passed=True,timing_runs=len(rows),memory_runs=len(memory),bytewise_predictor_replays=len(rows)+len(memory),warmup_predictors=33,warmup_seconds=warm,
        seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),timings_sha256=sha(out/'timings.json'),memory_sha256=sha(out/'memory.json'),
        methods={c['name']:dict(mean_seconds=float(np.mean([r['seconds'] for r in rows if r['method']==c['name']])),
            max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name'])) for c in p['configs']},
        memory_scope='one fresh process with all frozen meta models resident; trace misses some native allocations; two fixed tasks, no isolated deployment peak or RSS superiority claim')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

"""258A paired fresh full vs short search, plus the simpler coldALM16 control."""
import argparse
import gc
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import live_dual_jump_query as full
import live_dual_jump_short as short
import live_shared_mode_readout as coldlive
from run_local_dual_jump_continuation import CONFIGS
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def configs():
    selected=[c for c in CONFIGS if c['name'] in ['dual_alm64','activity_alm64','reorder_alm64','dual_reset_alm64']]
    ans=[dict(name=c['name']+'__'+backend,method=c['name'],backend=backend,config=c) for c in selected for backend in ['full','short']]
    ans.append(dict(name='cold__alm16',method='cold__alm16',backend='cold',config=next(c for c in full.cold.configs() if c['name']=='alm16')))
    return ans

def fit(cfg,x,v,q,seed):
    start=time.perf_counter()
    if cfg['backend']=='cold':
        arrays,meta=coldlive.fit(cfg['config'],x,v,q,seed);arrays['point_prediction']=full.cold.base.forward(q,arrays['selected_b'])
    else:arrays,meta=(full if cfg['backend']=='full' else short).fit(cfg['config'],x,v,q,seed)
    meta['complete_call_seconds']=time.perf_counter()-start
    return arrays,meta

def check(root,seed,cfg,a):
    if cfg['backend']=='cold':
        with np.load(root/'results/cold_stagnation_switch/development'/f'{seed}_alm16.npz') as z:
            assert a['selected_b'].tobytes()==z['selected_b'].tobytes();assert a['point_prediction'].tobytes()==z['prediction'].tobytes()
        path=root/'results/shared_mode_readout/development'/f'state_{seed}_alm16_0.npz'
    else:
        with np.load(root/'results/dual_jump_query/development'/f'point_{seed}_{cfg["method"]}.npz') as z:
            assert a['selected_b'].tobytes()==z['b'].tobytes();assert a['point_prediction'].tobytes()==z['prediction'].tobytes()
        with np.load(root/'results/local_dual_jump/continuation'/f'{seed}_{cfg["method"]}.npz') as z:assert a['best_bank'].tobytes()==z['best_bank'].tobytes()
        path=root/'results/dual_jump_query/development'/f'mode_{seed}_{cfg["method"]}_0.npz'
    with np.load(path) as z:
        for key in ['points','allocation','prediction']:assert a[key].tobytes()==z[key].tobytes(),(seed,cfg['name'],key)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/minimum_sufficient_dual';out=base/'short_resources';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    primitive=json.loads((base/'short_primitive/summary.json').read_text());assert primitive['passed'];hashes=dict(primitive['source_sha256']);hashes.update({n:sha(src/n) for n in ['live_dual_jump_short.py',Path(__file__).name]})
    for n,h in hashes.items():assert sha(src/n)==h,n
    p=dict(source_sha256=hashes,primitive_sha256=sha(base/'short_primitive/summary.json'),configs=configs(),seeds=list(range(5900000,5900016)),memory_seeds=[5900000,5900015],
        order_seed=258929,query_targets_accessed=False,scope='fresh complete paired pipelines; same exact predicates,events,writes,geometry and predictors; only trial execution shortened')
    dump(out/'protocol.json',p);rows=[];memory=[];q=np.linspace(0,1,257);begin=time.perf_counter()
    with full.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=observations(5900001);start=time.perf_counter()
        for cfg in p['configs']:a,m=fit(cfg,x[:4],v[:4],q,5900001);check(root,5900001,cfg,a)
        warm=time.perf_counter()-start;jobs=[(seed,cfg) for seed in p['seeds'] for cfg in p['configs']]
        for i in np.random.default_rng(p['order_seed']).permutation(len(jobs)):
            seed,cfg=jobs[i];x,v=observations(seed);a,m=fit(cfg,x[:4],v[:4],q,seed);check(root,seed,cfg,a);rows.append(dict(seed=seed,method=cfg['name'],backend=cfg['backend'],metadata=m))
            if len(rows)%18==0:dump(out/'timings.json',rows);print(json.dumps(dict(timing_runs=len(rows),total=144)),flush=True)
        for seed in p['memory_seeds']:
            x,v=observations(seed)
            for cfg in p['configs']:
                gc.collect();tracemalloc.start();a,m=fit(cfg,x[:4],v[:4],q,seed);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();check(root,seed,cfg,a)
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,metadata=m))
            dump(out/'memory.json',memory);print(json.dumps(dict(memory_runs=len(memory),total=18)),flush=True)
    means={c['name']:dict(mean_seconds=float(np.mean([r['metadata']['complete_call_seconds'] for r in rows if r['method']==c['name']])),
        mean_search_seconds=float(np.mean([r['metadata'].get('search_seconds',0.) for r in rows if r['method']==c['name']])),
        max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name'])) for c in p['configs']}
    for n,h in hashes.items():assert sha(src/n)==h,n
    result=dict(passed=True,timing_runs=len(rows),memory_runs=len(memory),bytewise_predictor_replays=len(rows)+len(memory),warmup_replays=9,warmup_seconds=warm,seconds=time.perf_counter()-begin,
        methods=means,protocol_sha256=sha(out/'protocol.json'),timings_sha256=sha(out/'timings.json'),memory_sha256=sha(out/'memory.json'),
        scope='output-preserving engineering result, not new query improvement; trace excludes some native allocation and is not isolated deployment memory')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

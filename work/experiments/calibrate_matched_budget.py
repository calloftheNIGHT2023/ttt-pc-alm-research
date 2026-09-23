"""262A: time-only configuration inclusion; old development inputs only."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import tracemalloc
import numpy as np
import torch
import matched_budget_suite as suite
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/matched_budget_confirmation';inp=base/'primitive';out=base/'calibration';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    summary=json.loads((inp/'summary.json').read_text());assert summary['passed'];p=json.loads((inp/'protocol.json').read_text());assert sha(inp/'protocol.json')==summary['protocol_sha256'];hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    start=time.perf_counter();loaded,manifest=suite.oldfit.meta.load(root);loading=time.perf_counter()-start;assert manifest==p['checkpoint_manifest'];cfgs=suite.catalogue()
    protocol=dict(source_sha256=hashes,primitive_summary_sha256=sha(inp/'summary.json'),checkpoint_manifest=manifest,prior_sha256=suite.hashes_of_priors(),configs=cfgs,
        seeds=list(range(5900000,5900016)),memory_seeds=[5900000,5900015],order_seed=262929,primary=suite.PRIMARY,budget_factor=1.10,model_loading_seconds=loading,query_targets_accessed=False,
        selection='retain all original36; every new config within time cap; cheapest extra above cap for groups with no new config within cap; never select on risk')
    dump(out/'protocol.json',protocol);q=np.linspace(0,1,257);rows=[];memory=[];begin=time.perf_counter();checks=0
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=observations(5900001);start=time.perf_counter()
        for cfg in cfgs:
            a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,5900001,loaded);assert not m['execution_failed'];suite.check_existing(root,5900001,cfg,a,m)
        warm=time.perf_counter()-start;jobs=[(seed,cfg) for seed in protocol['seeds'] for cfg in cfgs]
        for i in np.random.default_rng(protocol['order_seed']).permutation(len(jobs)):
            seed,cfg=jobs[i];x,v=observations(seed);a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,seed,loaded);checks+=suite.check_existing(root,seed,cfg,a,m)
            path=out/f"{seed}_{cfg['name']}.npz";np.savez_compressed(path,x_observed=x[:4],v_observed=v[:4],q_observed=q,**a)
            rows.append(dict(seed=seed,method=cfg['name'],family=cfg['family'],metadata=m,seconds=m['charged_complete_seconds'],file=path.name,sha256=sha(path)))
            if len(rows)%25==0:dump(out/'timings.json',rows);print(json.dumps(dict(timing_runs=len(rows),total=len(jobs))),flush=True)
        dump(out/'timings.json',rows)
        for seed in protocol['memory_seeds']:
            x,v=observations(seed)
            for cfg in cfgs:
                gc.collect();tracemalloc.start();a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,seed,loaded);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                with np.load(out/f"{seed}_{cfg['name']}.npz") as ref:
                    for k,value in a.items():assert value.tobytes()==ref[k].tobytes(),(seed,cfg['name'],k)
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,metadata=m))
                if len(memory)%10==0:dump(out/'memory.json',memory);print(json.dumps(dict(memory_runs=len(memory),total=50)),flush=True)
        dump(out/'memory.json',memory)
    means={c['name']:float(np.mean([r['seconds'] for r in rows if r['method']==c['name']])) for c in cfgs};cap=means[suite.PRIMARY]*protocol['budget_factor'];original=suite.original.configs();oldnames={c['name'] for c in original};extras=[c for c in cfgs if c['name'] not in oldnames]
    selected=[c for c in extras if means[c['name']]<=cap];selected_names={c['name'] for c in selected};over=[]
    for group in sorted({c['group'] for c in extras}):
        cc=[c for c in extras if c['group']==group]
        if not any(c['name'] in selected_names for c in cc):over.append(min(cc,key=lambda c:(means[c['name']],c['name'])))
    choices=original+selected+over;assert len({c['name'] for c in choices})==len(choices)
    dump(out/'selected_configs.json',dict(configs=choices,within_budget_extras=[c['name'] for c in selected],over_budget_background_extras=[c['name'] for c in over],
        budget_seconds=cap,primary_seconds=means[suite.PRIMARY],calibration_means=means,query_targets_accessed=False))
    for n,h in hashes.items():assert sha(src/n)==h,n
    ans=dict(passed=True,timing_runs=len(rows),memory_runs=len(memory),warmup_runs=25,warmup_seconds=warm,seconds=time.perf_counter()-begin,existing_bytewise_predictors=checks,
        failures=sum(r['metadata']['execution_failed'] for r in rows),selected_configs=len(choices),within_budget_extras=[c['name'] for c in selected],over_budget_background_extras=[c['name'] for c in over],budget_seconds=cap,
        methods={c['name']:dict(mean_seconds=means[c['name']],max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name'])) for c in cfgs},
        protocol_sha256=sha(out/'protocol.json'),timings_sha256=sha(out/'timings.json'),memory_sha256=sha(out/'memory.json'),selected_configs_sha256=sha(out/'selected_configs.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()

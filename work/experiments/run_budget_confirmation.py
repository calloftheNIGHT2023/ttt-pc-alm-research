"""262B: one frozen64-task prediction batch, never imports an evaluator.

This file stops after saving every prediction and the pre-evaluation manifest.
Only declared numerical failures fall back; interruption/I/O/coding errors stop.
"""
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
    base=root/'results/matched_budget_confirmation';cal=base/'calibration';audit=base/'calibration_audit';out=base/'confirmation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert json.loads((audit/'summary.json').read_text())['passed'];hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    cs=json.loads((cal/'summary.json').read_text());selection=json.loads((cal/'selected_configs.json').read_text());assert sha(cal/'selected_configs.json')==cs['selected_configs_sha256'];cfgs=selection['configs']
    scan=base/'reserved_scan.json';unseen=json.loads(scan.read_text());assert unseen['passed'] and unseen['matching_output_files']==[] and unseen['matching_result_json_files']==[]
    start=time.perf_counter();loaded,manifest=suite.oldfit.meta.load(root);loading=time.perf_counter()-start;cp=json.loads((cal/'protocol.json').read_text());assert manifest==cp['checkpoint_manifest'];assert suite.hashes_of_priors()==cp['prior_sha256']
    p=dict(source_sha256=hashes,calibration_audit_sha256=sha(audit/'summary.json'),calibration_selection_sha256=sha(cal/'selected_configs.json'),reserved_scan_sha256=sha(scan),configs=cfgs,
        checkpoint_manifest=manifest,prior_sha256=cp['prior_sha256'],seeds=list(range(5910000,5910064)),memory_seeds=[5910000,5910063],primary=suite.PRIMARY,
        query_points=257,n_context=4,mode_particles=2048,repetition_seed=249911,order_seed=262930,model_loading_seconds=loading,
        recoverable_exceptions=[t.__name__ for t in suite.RECOVERABLE],failure_policy='charge attempt and fixed zero-bias predictor; preserve all tasks; no retries; other errors stop before evaluation',
        query_targets_accessed=False,scope='one independent synthetic task batch; primary single fixed deployment MC; no task-dependent method choice, no truth until all prediction files are committed')
    dump(out/'protocol.json',p);q=np.linspace(0,1,257);rows=[];memory=[];files={};begin=time.perf_counter()
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x,v=observations(5900001);start=time.perf_counter()
        for cfg in cfgs:
            a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,5900001,loaded);assert not m['execution_failed']
        warm=time.perf_counter()-start;jobs=[(s,c) for s in p['seeds'] for c in cfgs]
        for i in np.random.default_rng(p['order_seed']).permutation(len(jobs)):
            seed,cfg=jobs[i];x,v=observations(seed);a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,seed,loaded)
            path=out/f"{seed}_{cfg['name']}.npz";assert not path.exists();np.savez_compressed(path,x_observed=x[:4],v_observed=v[:4],q_observed=q,**a);files[path.name]=sha(path)
            rows.append(dict(seed=seed,method=cfg['name'],family=cfg['family'],readout='point' if cfg['family'] in ['head','prior_plus'] else 'mode',
                file=path.name,sha256=files[path.name],metadata=m,seconds=m['charged_complete_seconds']))
            if len(rows)%len(cfgs)==0:
                dump(out/'rows.json',rows);print(json.dumps(dict(predictions_done=len(rows),total=len(jobs),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin)),flush=True)
        dump(out/'rows.json',rows)
        for seed in p['memory_seeds']:
            x,v=observations(seed)
            for cfg in cfgs:
                gc.collect();tracemalloc.start();a,m=suite.guarded_fit(cfg,x[:4],v[:4],q,seed,loaded);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                with np.load(out/f"{seed}_{cfg['name']}.npz") as ref:
                    for k,value in a.items():assert value.tobytes()==ref[k].tobytes(),(seed,cfg['name'],k)
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,metadata=m))
                if len(memory)%10==0:dump(out/'memory.json',memory);print(json.dumps(dict(memory_done=len(memory),total=2*len(cfgs))),flush=True)
        dump(out/'memory.json',memory)
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),memory_sha256=sha(out/'memory.json'),prediction_files=files,query_targets_accessed=False)
    dump(out/'before_query_manifest.json',before)
    ans=dict(passed=True,predictions_complete=True,tasks=64,configs=len(cfgs),predictions=len(rows),memory_replays=len(memory),failures=sum(r['metadata']['execution_failed'] for r in rows),
        warmup_seconds=warm,seconds=time.perf_counter()-begin,before_query_manifest_sha256=sha(out/'before_query_manifest.json'),query_targets_accessed=False,
        next='independent pre-query prediction audit, then one evaluation; no algorithm edits based on new answers')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()

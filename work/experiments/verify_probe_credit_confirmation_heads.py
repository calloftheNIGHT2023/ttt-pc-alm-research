"""287 additional-head golden preflight and192+16 old-input resource calls."""
import argparse,gc,json,os,time,tracemalloc
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import psutil
import probe_credit_confirmation_suite as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';out=base/'head_preflight';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parent=root/'results/round_286_audit.json';old=json.loads(parent.read_text());assert old['passed'];hashes=dict(old['source_sha256'])
    for n in ['probe_credit_confirmation_suite.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/287_probe_credit_confirmation_protocol.md';assert sha(design)==old['report_sha256'][design.name]
    allcfg=suite.catalogue(root);cfgs=[c for c in allcfg if c['group']=='additional_head'];index=suite.resources.frozen_inputs(root)
    start=time.perf_counter();loaded,manifest=suite.resources.legacy.oldfit.meta.load(root);loading=time.perf_counter()-start
    original_resource=root/'results/probe_credit_resources/calibration';rr=json.loads((original_resource/'protocol.json').read_text());assert rr['checkpoint_manifest']==manifest
    plan=json.loads((base/'planning/summary.json').read_text());assert plan['passed'] and plan['planned_tasks']==8192
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),all_configurations=allcfg,additional_configurations=cfgs,checkpoint_manifest=manifest,model_loading_seconds=loading,
        primitive_seeds=[5910000,5910001,5910053,5910063],timing_seeds=rr['seeds'],timing_repeats=3,memory_seeds=[5910000,5910063],order_seed=287929,original_resource_summary_sha256=sha(original_resource/'summary.json'),
        phase_accesses_new_contexts_or_targets=False,scope='Old observations only;32 head golden checks,192 complete untraced timings,16 separate memory probes; no confirmation task generated')
    dump(out/'protocol.json',p);inputs={};rows=[];timings=[];memory=[];counts=Counter();files={};begin=time.perf_counter();process=psutil.Process()
    for seed in p['timing_seeds']:
        directory,row=index[seed,suite.PRIMARY];assert sha(directory/row['file'])==row['sha256']
        with np.load(directory/row['file']) as z:inputs[seed]=tuple(z[k].copy() for k in ['x_observed','v_observed','q_observed'])
    def invoke(cfg,seed):
        x,v,q=inputs[seed];repairs=[];start=time.perf_counter()
        with conditioned.geometry_scope(repairs):a,m=suite.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=suite.fit)
        elapsed=time.perf_counter()-start;m.update(charged_complete_seconds=elapsed,geometry_repair_log=repairs)
        return a,m,elapsed
    with discovery_box(.12):
        for seed in p['primitive_seeds']:
            for cfg in cfgs:
                a,m,elapsed=invoke(cfg,seed);assert not m['execution_failed'];counts['primitive_gold_arrays']+=suite.resources.check_frozen(root,seed,cfg,a,m,index)
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,**a);files[fn]=sha(out/fn);rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=files[fn],metadata=m));counts['primitive_predictors']+=1
        dump(out/'primitive_rows.json',rows);print(json.dumps(dict(primitive_predictors=len(rows),seconds=time.perf_counter()-begin)),flush=True)
        jobs=[(seed,cfg,rep) for seed in p['timing_seeds'] for cfg in cfgs for rep in range(3)]
        for idx in np.random.default_rng(287929).permutation(len(jobs)):
            seed,cfg,rep=jobs[int(idx)];gc.collect();a,m,elapsed=invoke(cfg,seed)
            if not m['execution_failed']:counts['timing_gold_arrays']+=suite.resources.check_frozen(root,seed,cfg,a,m,index)
            timings.append(dict(seed=seed,method=cfg['name'],repeat=rep,order=len(timings),seconds=elapsed,metadata=m));del a,m
        dump(out/'timings.json',timings)
        for seed in p['memory_seeds']:
            for cfg in cfgs:
                gc.collect();before=process.memory_info()._asdict();tracemalloc.start();a,m,elapsed=invoke(cfg,seed);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()._asdict()
                if not m['execution_failed']:counts['memory_gold_arrays']+=suite.resources.check_frozen(root,seed,cfg,a,m,index)
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,process_before=before,process_after=after,instrumented_seconds_not_benchmark=elapsed,metadata=m));del a,m
        dump(out/'memory.json',memory)
    original_summary=json.loads((original_resource/'summary.json').read_text());cap=original_summary['primary_seconds'];table=[]
    for cfg in cfgs:
        vals=np.array([r['seconds'] for r in timings if r['method']==cfg['name']]);mean=float(vals.mean());label='main_1.00' if mean<=cap else 'sensitivity_1.10' if mean<=1.1*cap else 'measured_over_budget'
        table.append(dict(method=cfg['name'],mean_seconds=mean,median_seconds=float(np.median(vals)),p90_seconds=float(np.quantile(vals,.9)),maximum_seconds=float(vals.max()),resource_class=label,
            failures=sum(r['metadata']['execution_failed'] for r in timings if r['method']==cfg['name']),max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==cfg['name'])))
    dump(out/'methods.json',table);dump(out/'files.json',files)
    for n,h in hashes.items():assert sha(src/n)==h,n
    ans=dict(passed=True,counts=counts,timing_calls=len(timings),memory_calls=len(memory),seconds=time.perf_counter()-begin,failures=sum(r['metadata']['execution_failed'] for r in timings),
        phase_accesses_new_contexts_or_targets=False,outputs_sha256={n:sha(out/n) for n in ['protocol.json','primitive_rows.json','timings.json','memory.json','methods.json','files.json']},next='Independent extra-head/resource census, then frozen resumable8192-task27-method prediction run')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()

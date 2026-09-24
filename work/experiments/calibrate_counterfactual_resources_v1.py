"""293 fixed888 full timings and74 separate memory probes, no query truth."""
import argparse
import gc
from pathlib import Path
import time
import tracemalloc
import numpy as np
import psutil
import torch
import counterfactual_resource_suite_v1 as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def run(root,out):
    hashes=suite.gate(root)
    pre=root/'results/counterfactual_resources/preflight_v1'
    ss=suite.complete(pre);pp=read(pre/'protocol.json')
    assert ss['predictors']==148 and ss['counts']['local_reference_batches']==12
    assert pp['source_sha256']==hashes
    for name,digest in read(pre/'files.json').items():assert sha(pre/name)==digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfgs=suite.catalogue(root);assert cfgs==pp['configs']
    index=suite.frozen_inputs(root)
    inputs=suite.observed_inputs(root,index,suite.SEEDS)
    environment=suite.environment_snapshot(root)
    assert not environment['other_research_or_git_pack_processes'],environment
    start=time.perf_counter();loaded,manifest=suite.resources.legacy.oldfit.meta.load(root)
    loading=time.perf_counter()-start;assert manifest==pp['checkpoint_manifest']
    p=dict(source_sha256=hashes,preflight_summary_sha256=sha(pre/'summary.json'),
        design_sha256=pp['design_sha256'],configs=cfgs,seeds=suite.SEEDS,repeats=3,
        memory_seeds=[5910000,5910063],primary=suite.PRIMARY,order_seed=293929,
        warmup_seed=5910001,budget_factor=1.0,sensitivity_factor=1.10,
        checkpoint_manifest=manifest,model_loading_seconds=loading,
        all_preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),
        threads=dict(blas=1,omp=1,torch=torch.get_num_threads()),query_targets_accessed=False,
        scope='Complete original-input untraced calls including projection; loading/import/I-O/GC excluded and separately described',
        selection='All within candidate mean time; cheapest above cap per otherwise unrepresented group; all37 retained',
        memory_scope='Separate tracemalloc plus process RSS/lifetime peak; not per-method exact native peak',
        environment_start=environment)
    save(out/'protocol.json',p)
    (out/'calls').mkdir()
    warmup=[];timings=[];memory=[];files={};environments=[environment];checks=0
    begin=time.perf_counter();process=psutil.Process()
    def check_environment():
        row=suite.environment_snapshot(root);environments.append(row)
        save(out/'calls'/f'environment_{len(environments):03d}.json',row)
        assert not row['other_research_or_git_pack_processes'],row
    def invoke(cfg,seed):
        x,v,q=inputs[seed];repairs=[];start=time.perf_counter()
        def runner(c,xx,vv,qq,ss,ll):return suite.fit(c,xx,vv,qq,ss,ll,trace=False)
        with conditioned.geometry_scope(repairs):
            a,m=suite.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
        if m['execution_failed']:
            for field in suite.FIELDS:a[field]=suite.project(a[field])
        seconds=time.perf_counter()-start
        m.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs),charged_complete_seconds=seconds)
        return a,m,seconds
    def deterministic(cfg,seed,a,m):
        nonlocal checks
        fn=f'{seed}_{cfg["name"]}.npz'
        if fn not in files:
            with (out/fn).open('xb') as stream:np.savez_compressed(stream,**a)
            files[fn]=sha(out/fn)
        else:
            assert sha(out/fn)==files[fn]
            with np.load(out/fn,allow_pickle=False) as z:
                assert set(a)==set(z.files)
                for key,value in a.items():
                    assert value.shape==z[key].shape and value.dtype==z[key].dtype
                    assert value.tobytes()==z[key].tobytes(),(seed,cfg['name'],key)
                    checks+=1
        for field in suite.FIELDS:
            assert a[field].shape==(257,) and np.all((a[field]>=0)&(a[field]<=1))
        if not m['execution_failed']:checks+=suite.check_frozen(seed,cfg,a,m,index)
        return fn
    def record_call(kind,seed,cfg,rep,a,m,seconds):
        fn=deterministic(cfg,seed,a,m)
        return dict(seed=seed,method=cfg['name'],group=cfg['group'],repeat=rep,kind=kind,
                    seconds=seconds,metadata=m,file=fn,sha256=files[fn])
    with discovery_box(.12):
        for cfg in cfgs:
            gc.collect();a,m,seconds=invoke(cfg,5910001)
            row=record_call('warmup',5910001,cfg,0,a,m,seconds)
            save(out/'calls'/f'warmup_{len(warmup):03d}.json',row);warmup.append(row);del a,m
        save(out/'warmup.json',warmup);check_environment()
        jobs=[(seed,cfg,rep) for seed in suite.SEEDS for cfg in cfgs for rep in range(3)]
        assert len(jobs)==888
        for idx in np.random.default_rng(293929).permutation(len(jobs)):
            seed,cfg,rep=jobs[int(idx)]
            gc.collect();a,m,seconds=invoke(cfg,seed)
            row=record_call('timing',seed,cfg,rep,a,m,seconds);row['order']=len(timings)
            save(out/'calls'/f'timing_{len(timings):03d}.json',row);timings.append(row);del a,m
            if len(timings)%37==0:
                check_environment()
                print(dict(timing_calls=len(timings),total=888,seconds=time.perf_counter()-begin),flush=True)
        save(out/'timings.json',timings)
        for seed in p['memory_seeds']:
            for cfg in cfgs:
                gc.collect();before=process.memory_info()._asdict();tracemalloc.start()
                try:
                    a,m,instrumented=invoke(cfg,seed)
                    current,peak=tracemalloc.get_traced_memory()
                finally:tracemalloc.stop()
                after=process.memory_info()._asdict()
                row=record_call('memory',seed,cfg,0,a,m,instrumented)
                row.update(traced_current_bytes=current,traced_peak_bytes=peak,process_before=before,process_after=after,
                    instrumented_seconds_not_benchmark=instrumented,
                    returned_array_element_bytes_subtotal=sum(z.nbytes for z in a.values() if isinstance(z,np.ndarray)),
                    array_scope='Element-byte subtotal may double-count views; not predictor or native process peak')
                save(out/'calls'/f'memory_{len(memory):03d}.json',row);memory.append(row);del a,m
                if len(memory)%10==0:
                    check_environment()
                    print(dict(memory_calls=len(memory),total=74,seconds=time.perf_counter()-begin),flush=True)
        save(out/'memory.json',memory)
    check_environment()
    save(out/'files.json',files);save(out/'environments.json',environments)
    methods=[];means={}
    for cfg in cfgs:
        rr=[r for r in timings if r['method']==cfg['name']]
        seconds=np.array([r['seconds'] for r in rr]);assert len(rr)==24
        means[cfg['name']]=float(seconds.mean())
        methods.append(dict(method=cfg['name'],group=cfg['group'],mean_seconds=means[cfg['name']],
            median_seconds=float(np.median(seconds)),p90_seconds=float(np.quantile(seconds,.9)),maximum_seconds=float(seconds.max()),
            failures=sum(r['metadata']['execution_failed'] for r in rr),
            max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==cfg['name']),
            task_repeats=[dict(seed=seed,seconds=[r['seconds'] for r in rr if r['seed']==seed]) for seed in suite.SEEDS]))
    selection=suite.select_budget(cfgs,means)
    save(out/'methods.json',methods);save(out/'selection.json',selection)
    assert suite.gate(root)==hashes
    result=dict(passed=True,timing_runs=len(timings),memory_runs=len(memory),warmup_runs=len(warmup),
        methods=37,saved_predictors=len(files),deterministic_array_checks=checks,seconds=time.perf_counter()-begin,
        failures=sum(r['metadata']['execution_failed'] for r in timings),primary_seconds=means[suite.PRIMARY],
        query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','warmup.json','timings.json','memory.json',
            'files.json','methods.json','selection.json','environments.json']},
        next='Independent all37 resource audit before interpreting or selecting a fresh query experiment')
    save(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();out=root/'results/counterfactual_resources/calibration_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise


if __name__=='__main__':main()

"""327 full untraced time and separate-process native/traced memory probes."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import tracemalloc
import numpy as np
import psutil
import torch
import online_credit_resource_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def setup(root):
    hashes=suite.gate(root)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tick=time.perf_counter();loaded,manifest=suite.old.resources.legacy.oldfit.meta.load(root)
    seconds=time.perf_counter()-tick
    assert manifest==read(root/'results/online_stasis_resources/calibration_v1/protocol.json')['checkpoint_manifest']
    return hashes,loaded,manifest,seconds


def memory_worker(root,out,name,seed,parent):
    hashes,loaded,manifest,loading=setup(root)
    environment=suite.old.environment_snapshot(root)
    assert all(p['pid']==parent for p in environment['other_research_or_git_pack_processes']),environment
    cfg=next(c for c in suite.catalogue(root) if c['name']==name)
    index=suite.old.frozen_inputs(root);x,v,q=suite.old.observed_inputs(root,index,[seed])[seed]
    gc.collect();process=psutil.Process();before=process.memory_info()._asdict()
    with discovery_box(.12):
        tracemalloc.start()
        try:
            a,m,seconds=suite.invoke(cfg,x,v,q,seed,loaded)
            current,peak=tracemalloc.get_traced_memory()
        finally:tracemalloc.stop()
    after=process.memory_info()._asdict()
    with (out/'arrays.npz').open('xb') as f:np.savez_compressed(f,**a)
    save(out/'record.json',dict(seed=seed,method=name,metadata=m,instrumented_seconds_not_benchmark=seconds,
        process_before=before,process_after=after,traced_current_bytes=current,traced_peak_bytes=peak,
        model_loading_seconds=loading,preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),
        source_sha256=hashes,checkpoint_manifest=manifest,arrays_sha256=sha(out/'arrays.npz'),pid=os.getpid(),parent_pid=parent,
        environment=environment,query_targets_accessed=False,
        memory_scope='Isolated process absolute lifetime peak includes imports/models; traced peak excludes some native allocations. No peak subtraction.'))


def run(root,out):
    hashes,loaded,manifest,loading=setup(root);configs=suite.catalogue(root)
    index=suite.old.frozen_inputs(root);inputs=suite.old.observed_inputs(root,index,suite.old.SEEDS)
    environment=suite.old.environment_snapshot(root);assert not environment['other_research_or_git_pack_processes'],environment
    protocol=dict(source_sha256=hashes,design_sha256=sha(root/'outputs/ttt-pc-alm-research/327_resource_protocol_v1.md'),
        functional_audit_sha256=sha(root/'results/online_credit_branch_search/functional_audit_v1/summary.json'),
        configs=configs,seeds=suite.old.SEEDS,repeats=2,warmup_seed=5910001,memory_seeds=[5910000,5910063],order_seed=327929,
        primary=suite.PRIMARY,secondary=suite.SECONDARY,model_loading_seconds=loading,checkpoint_manifest=manifest,
        preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),threads=dict(blas=1,omp=1,torch=1),
        trace=False,query_targets_accessed=False,environment_start=environment,
        speed_scope='Complete fit plus range projection; no imports/loading/I-O/GC/tracemalloc. Each memory probe is a separate process.')
    save(out/'protocol.json',protocol);(out/'calls').mkdir();(out/'memory_calls').mkdir()
    begin=time.perf_counter();timings=[];warmup=[];memory=[];files={};checks=0;environments=[]
    def check_environment():
        env=suite.old.environment_snapshot(root);environments.append(env)
        save(out/'calls'/f'environment_{len(environments):04d}.json',env)
        assert not env['other_research_or_git_pack_processes'],env
    def deterministic(seed,cfg,a,m):
        nonlocal checks
        checks+=suite.check_reference(root,cfg,seed,a,m,index)
        filename=f'{seed}_{cfg["name"]}.npz'
        if filename not in files:
            with (out/filename).open('xb') as f:np.savez_compressed(f,**a)
            files[filename]=sha(out/filename)
        else:
            assert sha(out/filename)==files[filename]
            with np.load(out/filename,allow_pickle=False) as z:
                assert set(z.files)==set(a)
                for n,value in a.items():assert value.shape==z[n].shape and value.dtype==z[n].dtype and value.tobytes()==z[n].tobytes(),(seed,cfg['name'],n);checks+=1
        for f in suite.old.FIELDS:assert a[f].shape==(257,) and np.isfinite(a[f]).all() and np.all((a[f]>=0)&(a[f]<=1))
        return filename
    with discovery_box(.12):
        for cfg in configs:
            gc.collect();a,m,seconds=suite.invoke(cfg,*inputs[5910001],5910001,loaded)
            filename=deterministic(5910001,cfg,a,m)
            row=dict(seed=5910001,method=cfg['name'],seconds=seconds,metadata=m,file=filename,sha256=files[filename])
            save(out/'calls'/f'warmup_{len(warmup):03d}.json',row);warmup.append(row);del a,m
        save(out/'warmup.json',warmup);check_environment()
        jobs=[(seed,cfg,rep) for seed in protocol['seeds'] for cfg in configs for rep in range(2)]
        assert len(jobs)==816
        for order,idx in enumerate(np.random.default_rng(327929).permutation(len(jobs))):
            seed,cfg,rep=jobs[int(idx)];gc.collect();a,m,seconds=suite.invoke(cfg,*inputs[seed],seed,loaded)
            filename=deterministic(seed,cfg,a,m)
            row=dict(seed=seed,method=cfg['name'],group=cfg['group'],repeat=rep,order=order,seconds=seconds,metadata=m,file=filename,sha256=files[filename])
            save(out/'calls'/f'timing_{order:04d}.json',row);timings.append(row);del a,m
            if len(timings)%51==0:
                check_environment();print(dict(timing_calls=len(timings),total=816,seconds=time.perf_counter()-begin),flush=True)
    save(out/'timings.json',timings)
    for seed in protocol['memory_seeds']:
        for cfg in configs:
            check_environment();directory=out/'memory_calls'/f'{seed}_{cfg["name"]}'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--out',str(directory),'--worker',cfg['name'],'--seed',str(seed),'--parent',str(os.getpid())]
            child=subprocess.run(cmd,capture_output=True,text=True,check=False)
            if child.returncode:
                save(out/'worker_failure.json',dict(command=cmd,returncode=child.returncode,stdout=child.stdout,stderr=child.stderr));raise RuntimeError('Memory worker failed; see worker_failure.json')
            row=read(directory/'record.json');assert row['source_sha256']==hashes and row['checkpoint_manifest']==manifest
            assert sha(directory/'arrays.npz')==row['arrays_sha256']
            with np.load(directory/'arrays.npz',allow_pickle=False) as z:a={n:z[n] for n in z.files}
            deterministic(seed,cfg,a,row['metadata']);del a
            row['record_file']=str((directory/'record.json').relative_to(out));row['record_sha256']=sha(directory/'record.json')
            memory.append(row)
            if len(memory)%6==0:print(dict(memory_calls=len(memory),total=102,seconds=time.perf_counter()-begin),flush=True)
    save(out/'memory.json',memory);save(out/'files.json',files);save(out/'environments.json',environments)
    methods=[];means={}
    for cfg in configs:
        rr=[r for r in timings if r['method']==cfg['name']];assert len(rr)==16
        mm=[r for r in memory if r['method']==cfg['name']];assert len(mm)==2
        times=np.array([r['seconds'] for r in rr]);means[cfg['name']]=float(times.mean())
        methods.append(dict(method=cfg['name'],group=cfg['group'],mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),p90_seconds=float(np.quantile(times,.9)),maximum_seconds=float(times.max()),
            failures=sum(r['metadata']['execution_failed'] for r in rr),maximum_traced_peak_bytes=max(r['traced_peak_bytes'] for r in mm),
            maximum_absolute_lifetime_peak_wset=max(r['process_after'].get('peak_wset',0) for r in mm),
            isolated_process_memory_records=[r['record_file'] for r in mm]))
    selection={}
    for name in [suite.PRIMARY,suite.SECONDARY]:
        cap=means[name]
        selection[name]=dict(mean_seconds=cap,within_budget=[c['name'] for c in configs if means[c['name']]<=cap],
            sensitivity_110_percent=[c['name'] for c in configs if means[c['name']]<=1.1*cap],
            highest_adam_still_within=means['probe_then_adam3840_33']<=cap,all_methods_retained=True,query_quality_used=False)
    save(out/'methods.json',methods);save(out/'selection.json',selection);check_environment();assert suite.gate(root)==hashes
    result=dict(passed=True,methods=51,timing_runs=816,memory_runs=102,warmup_runs=51,saved_predictors=len(files),deterministic_array_checks=checks,
        seconds=time.perf_counter()-begin,failures=sum(r['metadata']['execution_failed'] for r in timings),primary_seconds=means[suite.PRIMARY],secondary_seconds=means[suite.SECONDARY],
        query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','warmup.json','timings.json','memory.json','files.json','environments.json','methods.json','selection.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker');p.add_argument('--seed',type=int);p.add_argument('--parent',type=int)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    try:
        if args.worker:memory_worker(root,args.out,args.worker,args.seed,args.parent)
        else:run(root,args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

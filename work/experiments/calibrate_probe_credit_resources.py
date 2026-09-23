"""285 single-thread complete-cost selection:600 calls, then50 memory probes."""
import argparse,gc,json,os,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import torch
import probe_credit_resource_suite as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_resources';pre=base/'primitive';out=base/'calibration';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ss=json.loads((pre/'summary.json').read_text());assert ss['passed'];pp=json.loads((pre/'protocol.json').read_text());assert sha(pre/'protocol.json')==ss['protocol_sha256'] and sha(pre/'rows.json')==ss['rows_sha256']
    hashes=dict(pp['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert sha(root/'outputs/ttt-pc-alm-research/285_probe_credit_resource_protocol.md')==pp['design_sha256']
    start=time.perf_counter();loaded,manifest=suite.legacy.oldfit.meta.load(root);loading=time.perf_counter()-start;assert manifest==pp['checkpoint_manifest']
    cfgs=suite.catalogue();assert cfgs==pp['configs'];index=suite.frozen_inputs(root);seeds=[5910000,5910001,5910008,5910016,5910032,5910048,5910053,5910063];inputs={}
    for seed in seeds:
        directory,row=index[seed,suite.PRIMARY];assert sha(directory/row['file'])==row['sha256']
        with np.load(directory/row['file']) as z:inputs[seed]=tuple(z[k].copy() for k in ['x_observed','v_observed','q_observed'])
    p=dict(source_sha256=hashes,primitive_summary_sha256=sha(pre/'summary.json'),design_sha256=pp['design_sha256'],configs=cfgs,seeds=seeds,repeats=3,memory_seeds=[5910000,5910063],
        primary=suite.PRIMARY,order_seed=285929,budget_factor=1.0,sensitivity_factor=1.10,warmup_seed=5910001,checkpoint_manifest=manifest,
        model_loading_seconds=loading,all_preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),
        threads=dict(blas=1,omp=1,torch=torch.get_num_threads()),phase_accesses_query_targets=False,
        scope='Complete original-input calls, all online work, no trace; loading/import/I-O excluded and separate; no other research compute launched concurrently',
        selection='All within primary mean time; each group cheapest above cap if none within; quality unused; old69 remain in next quality report',
        memory_scope='Separate tracemalloc and process RSS/lifetime peak; neither is an exact per-method native worst-case peak')
    dump(out/'protocol.json',p);timings=[];warmup=[];memory=[];files={};checks=0;begin=time.perf_counter();process=psutil.Process()
    def invoke(cfg,seed):
        x,v,q=inputs[seed];repairs=[];start=time.perf_counter()
        def runner(c,x,v,q,seed,loaded):return suite.fit(c,x,v,q,seed,loaded,trace=False)
        with conditioned.geometry_scope(repairs):a,m=suite.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
        elapsed=time.perf_counter()-start;m.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs),charged_complete_seconds=elapsed)
        return a,m,elapsed
    def deterministic(cfg,seed,a,m):
        nonlocal checks
        fn=f'{seed}_{cfg["name"]}.npz'
        if fn not in files:
            x,v,q=inputs[seed];np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);files[fn]=sha(out/fn)
        else:
            assert sha(out/fn)==files[fn]
            with np.load(out/fn) as z:
                for k,val in a.items():assert val.tobytes()==z[k].tobytes(),(seed,cfg['name'],k);checks+=1
        if not m['execution_failed']:checks+=suite.check_frozen(root,seed,cfg,a,m,index)
        return fn
    with discovery_box(.12):
        for cfg in cfgs:
            gc.collect();a,m,seconds=invoke(cfg,5910001);checks+=suite.check_frozen(root,5910001,cfg,a,m,index) if not m['execution_failed'] else 0
            warmup.append(dict(seed=5910001,method=cfg['name'],seconds=seconds,metadata=m));del a,m
        dump(out/'warmup.json',warmup);jobs=[(seed,cfg,rep) for seed in seeds for cfg in cfgs for rep in range(3)]
        for idx in np.random.default_rng(285929).permutation(len(jobs)):
            seed,cfg,rep=jobs[int(idx)];gc.collect();a,m,seconds=invoke(cfg,seed);fn=deterministic(cfg,seed,a,m)
            timings.append(dict(seed=seed,method=cfg['name'],group=cfg['group'],repeat=rep,order=len(timings),seconds=seconds,metadata=m,file=fn,sha256=files[fn]));del a,m
            if len(timings)%50==0:dump(out/'timings.json',timings);print(json.dumps(dict(timing_runs=len(timings),total=600,seconds=time.perf_counter()-begin)),flush=True)
        dump(out/'timings.json',timings);dump(out/'files.json',files)
        # Memory instrumentation is deliberately not included in timing means.
        for seed in p['memory_seeds']:
            for cfg in cfgs:
                gc.collect();before=process.memory_info()._asdict();tracemalloc.start();a,m,instrumented=invoke(cfg,seed);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=process.memory_info()._asdict()
                fn=deterministic(cfg,seed,a,m);returned=sum(z.nbytes for z in a.values() if isinstance(z,np.ndarray))
                memory.append(dict(seed=seed,method=cfg['name'],traced_current_bytes=current,traced_peak_bytes=peak,process_before=before,process_after=after,
                    returned_array_element_bytes_subtotal=returned,array_subtotal_scope='Can count shared backing buffers twice; not retained predictor or process peak',instrumented_seconds_not_benchmark=instrumented,metadata=m,file=fn,sha256=files[fn]));del a,m
                if len(memory)%10==0:dump(out/'memory.json',memory);print(json.dumps(dict(memory_runs=len(memory),total=50,seconds=time.perf_counter()-begin)),flush=True)
        dump(out/'memory.json',memory)
    methods=[];means={}
    for cfg in cfgs:
        rr=[r for r in timings if r['method']==cfg['name']];seconds=np.array([r['seconds'] for r in rr]);means[cfg['name']]=float(seconds.mean())
        bytask=[dict(seed=seed,seconds=[r['seconds'] for r in rr if r['seed']==seed]) for seed in seeds]
        methods.append(dict(method=cfg['name'],group=cfg['group'],mean_seconds=means[cfg['name']],median_seconds=float(np.median(seconds)),p90_seconds=float(np.quantile(seconds,.9)),maximum_seconds=float(seconds.max()),
            failures=sum(r['metadata']['execution_failed'] for r in rr),task_repeats=bytask,max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==cfg['name'])))
    cap=means[suite.PRIMARY];within=[c for c in cfgs if means[c['name']]<=cap];above=[]
    for group in sorted({c['group'] for c in cfgs}):
        cc=[c for c in cfgs if c['group']==group]
        if not any(c in within for c in cc):above.append(min(cc,key=lambda c:(means[c['name']],c['name'])))
    selected=dict(primary=suite.PRIMARY,budget_seconds=cap,budget_factor=1.0,within_budget=[c['name'] for c in within],over_budget_background=[c['name'] for c in above],configs=within+above,
        sensitivity_110_percent=[c['name'] for c in cfgs if means[c['name']]<=cap*1.1],calibration_means=means,phase_accesses_query_targets=False)
    dump(out/'methods.json',methods);dump(out/'selected_configs.json',selected)
    for n,h in hashes.items():assert sha(src/n)==h,n
    ans=dict(passed=True,timing_runs=len(timings),memory_runs=len(memory),warmup_runs=len(warmup),saved_predictors=len(files),deterministic_array_checks=checks,seconds=time.perf_counter()-begin,
        failures=sum(r['metadata']['execution_failed'] for r in timings),primary_seconds=cap,within_budget=selected['within_budget'],over_budget_background=selected['over_budget_background'],phase_accesses_query_targets=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','warmup.json','timings.json','memory.json','files.json','methods.json','selected_configs.json']},next='Independent600/50 resource audit, then freeze selected-budget development predictions; goal active')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()

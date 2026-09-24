"""294 fixed128 new development tasks; seal ALL37 outputs before query scoring.

Separate process phases, no intermediate quality monitoring or automatic retry.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import counterfactual_resource_suite_v1 as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_independent_hybrid_memory import observations
from analyze_recovered_online_comparison import forward
import probe_confirmation_statistics as statistics
from diagnose_gradient_flat_split_states_v1 import read, save, sha

PILOT_SEEDS = list(range(294000000,294000128))
OLD_SEEDS = [5910000,5910063]
METRICS = ['mse257','mse129','point_mse257','point_mse129']
DESIGN = 'outputs/ttt-pc-alm-research/294_branch_complementarity_risk_and_pilot.md'
SOURCES = ['counterfactual_fresh_pilot_v1.py','test_counterfactual_fresh_pilot_v1.py',
           'continue_counterfactual_pilot_v1.ps1']


def exclusive(path, value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())


def gate(root):
    hashes=suite.gate(root)
    cal=root/'results/counterfactual_resources/calibration_v1'
    audit=root/'results/counterfactual_resources/audit_v1'
    cs=suite.complete(cal);a=suite.complete(audit)
    assert cs['timing_runs']==888 and a['methods']==37
    assert a['calibration_summary_sha256']==sha(cal/'summary.json')
    assert read(cal/'protocol.json')['source_sha256']==hashes
    for name in SOURCES:hashes[name]=sha(root/'work/experiments'/name)
    return hashes


def scalar_truth(seed,q):
    b=np.random.default_rng(seed).uniform(-.12,.12,4)
    ans=[]
    for xx in q:
        h=float(xx)
        for bias in b:
            t=h+float(bias)
            if t<=0. or t>=1.:h=0.
            elif t<=.5:h=2.*t
            else:h=2.-2.*t
        ans.append(h)
    return np.array(ans)


def validate_arrays(a):
    for key,value in a.items():
        if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),key
    for name in suite.FIELDS:
        assert a[name].shape==(257,) and a[name].dtype==np.float64
        assert np.all((a[name]>=0)&(a[name]<=1))
    assert a['x_observed'].shape==a['v_observed'].shape==(4,)
    assert np.array_equal(a['q_observed'],np.linspace(0,1,257))
    assert not np.isin(a['q_observed'],a['x_observed']).any()


def verify_predictions(folder, protocol, reference=None):
    counts=Counter();rows=[];files={}
    for seed in protocol['seeds']:
        task=folder/str(seed);c=read(task/'commit.json')
        assert c['seed']==seed and not c['query_targets_accessed']
        assert c['protocol_sha256']==sha(folder/'protocol.json')
        expected=[protocol['configs'][int(i)]['name'] for i in np.random.default_rng(
            np.random.SeedSequence([294929,seed])).permutation(37)]
        assert [r['method'] for r in c['rows']]==expected
        assert len(c['rows'])==37
        observed=None
        for order,r in enumerate(c['rows']):
            assert r['seed']==seed and r['order']==order and r['seconds']>0
            assert r['seconds']==r['metadata']['charged_complete_seconds']
            assert not r['metadata'].get('query_targets_accessed',False)
            assert sha(folder/r['file'])==r['sha256']
            for suffix in ['start','call']:
                path=folder/r[suffix+'_file']
                assert sha(path)==r[suffix+'_sha256']
                journal=read(path)
                assert journal['seed']==seed and journal['method']==r['method']
                assert journal['order']==order and journal['protocol_sha256']==c['protocol_sha256']
                if suffix=='call':assert journal['metadata']==r['metadata']
            with np.load(folder/r['file'],allow_pickle=False) as z:
                a={key:z[key] for key in z.files};validate_arrays(a)
                current=(a['x_observed'].tobytes(),a['v_observed'].tobytes())
                if observed is None:observed=current
                else:assert current==observed
                if reference is not None:
                    name=f'{seed}_{r["method"]}.npz'
                    assert sha(reference/name)==read(reference/'files.json')[name]
                    with np.load(reference/name,allow_pickle=False) as old:
                        for key in old.files:
                            assert a[key].dtype==old[key].dtype and a[key].shape==old[key].shape
                            assert a[key].tobytes()==old[key].tobytes(),(seed,r['method'],key)
                            counts['old_exact_arrays']+=1
                counts['numeric_arrays']+=sum(np.issubdtype(v.dtype,np.number) for v in a.values())
            files[r['file']]=r['sha256'];rows.append(r);counts['predictors']+=1
        assert c['failures']==sum(r['metadata']['execution_failed'] for r in c['rows'])
        assert c['charged_seconds']==sum(r['seconds'] for r in c['rows'])
        counts['tasks']+=1
    assert len(files)==counts['predictors']==len(protocol['seeds'])*37
    return rows,files,dict(counts)


def predict(root,out,stage):
    hashes=gate(root);cfgs=suite.catalogue(root)
    test=root/'results/counterfactual_fresh_pilot/tests_v1'
    assert suite.complete(test)['passed']
    assert read(test/'protocol.json')['source_sha256']==hashes
    cal=root/'results/counterfactual_resources/calibration_v1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    begin=time.perf_counter();loaded,manifest=suite.resources.legacy.oldfit.meta.load(root)
    loading=time.perf_counter()-begin
    assert manifest==read(cal/'protocol.json')['checkpoint_manifest']
    seeds=OLD_SEEDS if stage=='preflight' else PILOT_SEEDS
    if stage=='pilot':
        old=out.parent/'preflight_predictions_v1';oldscore=out.parent/'preflight_evaluation_v1'
        assert suite.complete(old)['predictors']==74
        assert suite.complete(oldscore)['independent_scalar_risk_passed']
        for folder in [old,oldscore]:
            assert read(folder/'protocol.json')['source_sha256']==hashes
            assert read(folder/'protocol.json')['design_sha256']==sha(root/DESIGN)
    environment=suite.environment_snapshot(root)
    assert not environment['other_research_or_git_pack_processes'],environment
    p=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/DESIGN),
        resource_summary_sha256=sha(cal/'summary.json'),resource_audit_sha256=sha(cal.parent/'audit_v1/summary.json'),
        test_summary_sha256=sha(test/'summary.json'),configs=cfgs,primary=suite.PRIMARY,seeds=seeds,
        methods=[c['name'] for c in cfgs],observations_per_task=4,query_points=257,particles=2048,
        order_seed=294929,query_targets_accessed=False,new_development_tasks=stage=='pilot',
        checkpoint_manifest=manifest,model_loading_seconds=loading,environment_start=environment,
        threads=dict(blas=1,omp=1,torch=torch.get_num_threads()),trace=False,
        failure_policy='Original charged numerical attempt and zero-bias fallback; no retries or task exclusions',
        quality_stopping=False,free_disk_floor_bytes=20*2**30,
        inference='New development pilot, not paper confirmation or general TTT evidence')
    exclusive(out/'protocol.json',p);ph=sha(out/'protocol.json')
    inputs=None
    if stage=='preflight':inputs=suite.observed_inputs(root,suite.frozen_inputs(root),seeds)
    environments=[environment];calls=0;begin=time.perf_counter()
    with discovery_box(.12):
        for seed in seeds:
            assert shutil.disk_usage(out).free>=p['free_disk_floor_bytes']
            task=out/str(seed);task.mkdir();rows=[];wall=time.perf_counter()
            if inputs is not None:x,v,q=inputs[seed]
            else:
                xx,vv=observations(seed);x=xx[:4].copy();v=vv[:4].copy();q=np.linspace(0,1,257);del xx,vv
            for idx in np.random.default_rng(np.random.SeedSequence([294929,seed])).permutation(37):
                cfg=cfgs[int(idx)];name=cfg['name'];order=len(rows);repairs=[]
                sf=task/(name+'.start.json');cf=task/(name+'.call.json')
                exclusive(sf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,unix_time=time.time()))
                def runner(c,xx,vv,qq,ss,ll):return suite.fit(c,xx,vv,qq,ss,ll,trace=False)
                start=time.perf_counter()
                with conditioned.geometry_scope(repairs):
                    a,m=suite.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
                if m['execution_failed']:
                    for field in suite.FIELDS:a[field]=suite.project(a[field])
                seconds=time.perf_counter()-start
                m.update(charged_complete_seconds=seconds,geometry_repair_log=repairs,
                         geometry_repair_count=len(repairs),query_targets_accessed=False)
                exclusive(cf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,metadata=m))
                a.update(x_observed=x,v_observed=v,q_observed=q);validate_arrays(a)
                path=task/(name+'.npz')
                with path.open('xb') as stream:
                    np.savez_compressed(stream,**a);stream.flush();os.fsync(stream.fileno())
                rows.append(dict(seed=seed,method=name,order=order,file=str(path.relative_to(out)),sha256=sha(path),
                    start_file=str(sf.relative_to(out)),start_sha256=sha(sf),call_file=str(cf.relative_to(out)),
                    call_sha256=sha(cf),seconds=seconds,metadata=m));calls+=1;del a,m
            exclusive(task/'commit.json',dict(seed=seed,protocol_sha256=ph,rows=rows,
                failures=sum(r['metadata']['execution_failed'] for r in rows),charged_seconds=sum(r['seconds'] for r in rows),
                wall_with_io_seconds=time.perf_counter()-wall,query_targets_accessed=False))
            if calls%(37*4)==0 or stage=='preflight':
                env=suite.environment_snapshot(root);environments.append(env)
                exclusive(out/f'environment_{calls//37:03d}.json',env)
                assert not env['other_research_or_git_pack_processes'],env
                print(dict(stage=stage,tasks=calls//37,total=len(seeds),predictors=calls,
                    seconds=time.perf_counter()-begin,query_targets_accessed=False),flush=True)
    rows,files,checks=verify_predictions(out,p,cal if stage=='preflight' else None)
    assert gate(root)==hashes
    for name,value in [('rows.json',rows),('files.json',files),('environments.json',environments)]:exclusive(out/name,value)
    exclusive(out/'before_query_manifest.json',dict(protocol_sha256=ph,source_sha256=hashes,
        rows_sha256=sha(out/'rows.json'),files_sha256=sha(out/'files.json'),checks=checks,query_targets_accessed=False))
    result=dict(passed=True,tasks=len(seeds),predictors=calls,checks=checks,
        failures=sum(r['metadata']['execution_failed'] for r in rows),query_targets_accessed=False,
        seconds=time.perf_counter()-begin,core_research_goal_complete=False,
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','rows.json','files.json',
            'environments.json','before_query_manifest.json']})
    exclusive(out/'summary.json',result);print(result,flush=True)


def score(root,out,stage):
    hashes=gate(root);folder=out.parent/(stage+'_predictions_v1');run=suite.complete(folder)
    p=read(folder/'protocol.json');assert p['source_sha256']==hashes and not p['query_targets_accessed']
    assert p['design_sha256']==sha(root/DESIGN)
    assert p['seeds']==(OLD_SEEDS if stage=='preflight' else PILOT_SEEDS)
    cal=root/'results/counterfactual_resources/calibration_v1'
    rows,files,checks=verify_predictions(folder,p,cal if stage=='preflight' else None)
    assert checks==run['checks'] and rows==read(folder/'rows.json') and files==read(folder/'files.json')
    assert sha(folder/'files.json')==read(folder/'before_query_manifest.json')['files_sha256']
    names=p['methods'];seeds=p['seeds'];pi=names.index(suite.PRIMARY);ci=[j for j in range(37) if j!=pi]
    assert pi==0 and len(ci)==36
    protocol=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/DESIGN),
        prediction_summary_sha256=sha(folder/'summary.json'),before_query_manifest_sha256=sha(folder/'before_query_manifest.json'),
        methods=names,seeds=seeds,primary=suite.PRIMARY,metrics=METRICS,primary_metric='mse257',
        primary_family=[names[j] for j in ci],bootstrap=dict(repetitions=20000,seed=294193,block=64,
            adjusted_quantile=1-.05/36,quantile_method='linear',resampling_unit='task'),
        query_targets_accessed=True,development_pilot=stage=='pilot',core_research_goal_complete=False,
        inference='Approximate paired percentile bootstrap; all36 retained, no interim quality selection')
    # All predictions and their durable call journals passed BEFORE this marker and query access.
    exclusive(out/'protocol.json',protocol);begin=time.perf_counter();n=len(seeds)
    risk=np.empty((n,37,4));times=np.empty((n,37));failed=np.zeros((n,37),bool)
    lookup={(r['seed'],r['method']):r for r in rows};scalar_gap=0.;truth_gap=0.
    for si,seed in enumerate(seeds):
        q=np.linspace(0,1,257);b=np.random.default_rng(seed).uniform(-.12,.12,4)
        truth=forward(q,b[None])[0];scalar=scalar_truth(seed,q)
        truth_gap=max(truth_gap,float(np.max(abs(truth-scalar))));assert truth_gap<2e-12
        for mi,name in enumerate(names):
            r=lookup[seed,name]
            with np.load(folder/r['file'],allow_pickle=False) as a:
                errors=[np.square(a[f]-truth) for f in suite.FIELDS]
                risk[si,mi]=[z.mean() for e in errors for z in [e,e[::2]]]
                independent=[]
                for field in suite.FIELDS:
                    for stride in [1,2]:
                        vals=a[field][::stride];target=scalar[::stride]
                        independent.append(math.fsum((float(u)-float(t))**2 for u,t in zip(vals,target))/len(vals))
                scalar_gap=max(scalar_gap,float(np.max(abs(risk[si,mi]-independent))))
                assert scalar_gap<2e-12
            times[si,mi]=r['seconds'];failed[si,mi]=r['metadata']['execution_failed']
    assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1))
    diff=risk[:,pi:pi+1,:]-risk[:,ci,:]
    statistics.selftest();boot,digest=statistics.bootstrap_means(diff.reshape(n,-1),20000,294193,64)
    # Independent direct indexed means, including every resample and all144 contrasts.
    rng=np.random.default_rng(294193);flat=diff.reshape(n,-1);boot_gap=0.
    for first in range(0,20000,7):
        size=min(7,20000-first);indices=rng.integers(n,size=(size,n),dtype=np.int64)
        other=flat[indices].mean(1)
        boot_gap=max(boot_gap,float(np.max(abs(other-boot[first:first+size]))))
        assert boot_gap<2e-12
    boot=boot.reshape(20000,36,4)
    with (out/'task_metrics.npz').open('xb') as stream:
        np.savez_compressed(stream,seeds=seeds,methods=names,metric_names=METRICS,risk=risk,seconds=times,failures=failed)
    with (out/'bootstrap_means.npz').open('xb') as stream:np.savez_compressed(stream,means=boot)
    comparisons=[]
    for j,mi in enumerate(ci):
        for k,metric in enumerate(METRICS):
            d=diff[:,j,k];bounds=np.quantile(boot[:,j,k],[.025,.975])
            upper=float(np.quantile(boot[:,j,k],1-.05/36)) if k==0 else None
            comparisons.append(dict(control=names[mi],metric=metric,mean_difference=float(d.mean()),
                sample_sd=float(d.std(ddof=1)),descriptive95=bounds.tolist(),bonferroni_upper=upper,
                adjusted_negative=upper<0 if upper is not None else None,
                improved=int((d<0).sum()),equal=int((d==0).sum()),worse=int((d>0).sum())))
    prior={r['method']:r for r in read(cal/'methods.json')};cap=prior[suite.PRIMARY]['mean_seconds']
    methods=[]
    for j,name in enumerate(names):
        mean=float(times[:,j].mean());ratio=prior[name]['mean_seconds']/cap
        methods.append(dict(method=name,metrics={metric:float(risk[:,j,k].mean()) for k,metric in enumerate(METRICS)},
            mean_seconds=mean,actual_ratio_to_primary=mean/float(times[:,pi].mean()),
            calibration_ratio_to_primary=ratio,calibration_class='1.00' if ratio<=1 else '1.10' if ratio<=1.1 else 'over_1.10',
            failures=int(failed[:,j].sum())))
    for name,value in [('methods.json',methods),('comparisons.json',comparisons)]:exclusive(out/name,value)
    assert gate(root)==hashes
    result=dict(passed=True,tasks=n,predictors=n*37,comparisons=144,main_family=36,
        adjusted_negative_comparisons=sum(r['adjusted_negative'] for r in comparisons if r['metric']=='mse257'),
        query_targets_accessed=True,independent_scalar_risk_passed=True,maximum_truth_gap=truth_gap,
        maximum_scalar_risk_gap=scalar_gap,all_bootstrap_means_checked=20000*144,maximum_bootstrap_gap=boot_gap,
        bootstrap_index_sha256=digest,numerical_failures=int(failed.sum()),core_research_goal_complete=False,
        stage=stage,seconds=time.perf_counter()-begin,
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','task_metrics.npz','bootstrap_means.npz',
            'methods.json','comparisons.json']})
    exclusive(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--stage',choices=['preflight','pilot'],required=True)
    p.add_argument('--phase',choices=['predict','score'],required=True);args=p.parse_args();root=args.project.resolve()
    suffix='predictions' if args.phase=='predict' else 'evaluation'
    out=root/'results/counterfactual_fresh_pilot'/f'{args.stage}_{suffix}_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:(predict if args.phase=='predict' else score)(root,out,args.stage)
    except BaseException as exc:
        exclusive(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise


if __name__=='__main__':main()

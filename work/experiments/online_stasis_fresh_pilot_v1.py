"""307: fixed256 x 46, durable prediction seal followed by separate scoring."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import online_stasis_fresh_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_independent_hybrid_memory import observations
from analyze_recovered_online_comparison import forward
from counterfactual_fresh_pilot_v1 import scalar_truth,validate_arrays
import probe_confirmation_statistics as stats
from diagnose_gradient_flat_split_states_v1 import read,save,sha

METRICS=['mse257','mse129','point_mse257','point_mse129']


def exclusive(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.flush();os.fsync(f.fileno())


def signature(x,v,q):
    h=hashlib.sha256()
    for a in [x,v,q]:h.update(a.tobytes())
    return h.hexdigest()


def verify_predictions(root,folder,p):
    rows=[];files={};counts=Counter();ref=suite.references(root) if p['stage']=='preflight' else None
    ph=sha(folder/'protocol.json')
    for seed in p['seeds']:
        commit=read(folder/str(seed)/'commit.json')
        assert commit['seed']==seed and commit['protocol_sha256']==ph and not commit['query_targets_accessed']
        expected=[p['configs'][int(i)]['name'] for i in np.random.default_rng(np.random.SeedSequence([307929,seed])).permutation(46)]
        assert [r['method'] for r in commit['rows']]==expected and len(commit['rows'])==46
        for i,r in enumerate(commit['rows']):
            assert r['seed']==seed and r['order']==i and math.isfinite(r['seconds']) and r['seconds']>0
            assert r['seconds']==r['metadata']['charged_complete_seconds']
            assert not r['metadata']['query_targets_accessed']
            assert sha(folder/r['file'])==r['sha256']
            for suffix in ['start','call']:
                path=folder/r[suffix+'_file'];assert sha(path)==r[suffix+'_sha256']
                note=read(path)
                assert note['seed']==seed and note['method']==r['method'] and note['order']==i and note['protocol_sha256']==ph
                if suffix=='call':assert note['metadata']==r['metadata']
            with np.load(folder/r['file'],allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files};validate_arrays(arrays)
                assert signature(arrays['x_observed'],arrays['v_observed'],arrays['q_observed'])==commit['observed_sha256']
                if ref is not None:counts['frozen_arrays']+=suite.check_reference(root,r,arrays,ref)
                counts['numeric_arrays']+=sum(np.issubdtype(a.dtype,np.number) for a in arrays.values())
            assert r['file'] not in files
            files[r['file']]=r['sha256'];rows.append(r);counts['predictors']+=1
        assert commit['failures']==sum(r['metadata']['execution_failed'] for r in commit['rows'])
        assert commit['charged_seconds']==sum(r['seconds'] for r in commit['rows'])
        counts['tasks']+=1
    assert len(rows)==len(files)==len(p['seeds'])*46
    return rows,files,dict(counts)


def cost_table(rows,names):
    means={name:math.fsum(r['seconds'] for r in rows if r['method']==name)/sum(r['method']==name for r in rows) for name in names}
    cap=means[suite.PRIMARY];table=[]
    for name in names:
        rr=[r for r in rows if r['method']==name];times=[r['seconds'] for r in rr];fields={}
        def collect(d,prefix=''):
            for k,v in d.items():
                if isinstance(v,dict):collect(v,prefix+k+'.')
                elif 'bytes' in k and isinstance(v,(int,float)) and not isinstance(v,bool):
                    fields[prefix+k]=max(fields.get(prefix+k,0),v)
        for r in rr:collect(r['metadata'])
        ratio=means[name]/cap
        table.append(dict(method=name,mean_seconds=means[name],median_seconds=float(np.median(times)),
            p90_seconds=float(np.quantile(times,.9)),ratio_to_primary=ratio,
            budget_class='1.00' if means[name]<=cap else '1.10' if means[name]<=1.10*cap else 'over_1.10',
            failures=sum(r['metadata']['execution_failed'] for r in rr),maximum_reported_byte_fields=fields,
            state_scope='Reported named byte subtotals; not uniform or exact native peak memory'))
    return dict(primary=suite.PRIMARY,primary_mean_seconds=cap,methods=table,query_targets_accessed=False,
        within_budget=[r['method'] for r in table if r['budget_class']=='1.00'],
        sensitivity_110_percent=[r['method'] for r in table if r['budget_class']!='over_1.10'],
        all_methods_retained=True,selection_uses_query_quality=False)


def predict(root,out,stage):
    hashes=suite.gate(root);cfgs=suite.catalogue(root)
    tests=root/suite.BASE/'tests_v1';ts=suite.old.complete(tests)
    assert ts['passed'] and read(tests/'protocol.json')['source_sha256']==hashes
    assert read(tests/'protocol.json')['design_sha256']==sha(root/suite.DESIGN)
    if stage=='pilot':
        for suffix,field,value in [('predictions','predictors',138),('evaluation','independent_scalar_risk_passed',True)]:
            folder=out.parent/f'preflight_{suffix}_v1';s=suite.old.complete(folder)
            assert s[field]==value
            assert read(folder/'protocol.json')['source_sha256']==hashes
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tick=time.perf_counter();loaded,manifest=suite.old.resources.legacy.oldfit.meta.load(root)
    load_seconds=time.perf_counter()-tick
    old_protocol=read(root/'results/online_stasis_resources/calibration_v1/protocol.json')
    assert manifest==old_protocol['checkpoint_manifest']
    seeds=suite.OLD_SEEDS if stage=='preflight' else suite.PILOT_SEEDS
    p=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),configs=cfgs,
        primary=suite.PRIMARY,methods=[c['name'] for c in cfgs],seeds=seeds,observations_per_task=4,
        query_points=257,particles=2048,order_seed=307929,query_targets_accessed=False,
        new_development_tasks=stage=='pilot',checkpoint_manifest=manifest,model_loading_seconds=load_seconds,
        all_preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),
        threads=dict(blas=1,omp=1,torch=torch.get_num_threads()),trace=False,free_disk_floor_bytes=20*2**30,
        failure_policy='Charge numerical failure and original zero-bias fallback, never retry or exclude',
        quality_stopping=False,test_summary_sha256=sha(tests/'summary.json'),
        resource_audit_sha256=sha(root/'results/online_stasis_resources/audit_v1/summary.json'),
        control_audit_sha256=sha(root/'results/online_stasis_controls/audit_v1/summary.json'),
        inference='Development pilot, not independent paper confirmation or general TTT evidence')
    exclusive(out/'protocol.json',p);ph=sha(out/'protocol.json')
    inputs=suite.old.observed_inputs(root,suite.old.frozen_inputs(root),seeds) if stage=='preflight' else None
    environments=[];calls=0;begin=time.perf_counter()
    def environment():
        while True:
            row=suite.old.environment_snapshot(root);row['paused']=bool(row['other_research_or_git_pack_processes'])
            environments.append(row);exclusive(out/f'environment_{len(environments):05d}.json',row)
            if not row['paused']:return
            print(dict(phase='interference_pause',environment=row),flush=True);time.sleep(10)
    with discovery_box(.12):
        for seed in seeds:
            environment()
            assert shutil.disk_usage(out).free>=p['free_disk_floor_bytes']
            task=out/str(seed);task.mkdir();rows=[];wall=time.perf_counter()
            if inputs is not None:x,v,q=inputs[seed]
            else:
                xx,vv=observations(seed);x=xx[:4].copy();v=vv[:4].copy();q=np.linspace(0,1,257);del xx,vv
            observed=signature(x,v,q)
            for i in np.random.default_rng(np.random.SeedSequence([307929,seed])).permutation(46):
                cfg=cfgs[int(i)];name=cfg['name'];order=len(rows)
                sf=task/(name+'.start.json');cf=task/(name+'.call.json')
                exclusive(sf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,unix_time=time.time()))
                a,m=suite.invoke(cfg,x,v,q,seed,loaded)
                exclusive(cf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,metadata=m))
                assert signature(x,v,q)==observed
                a.update(x_observed=x,v_observed=v,q_observed=q);validate_arrays(a)
                path=task/(name+'.npz')
                with path.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
                rows.append(dict(seed=seed,method=name,order=order,file=str(path.relative_to(out)),sha256=sha(path),
                    start_file=str(sf.relative_to(out)),start_sha256=sha(sf),call_file=str(cf.relative_to(out)),call_sha256=sha(cf),
                    seconds=m['charged_complete_seconds'],metadata=m));calls+=1
                del a,m
            exclusive(task/'commit.json',dict(seed=seed,protocol_sha256=ph,rows=rows,observed_sha256=observed,
                failures=sum(r['metadata']['execution_failed'] for r in rows),charged_seconds=sum(r['seconds'] for r in rows),
                wall_with_io_seconds=time.perf_counter()-wall,query_targets_accessed=False))
            print(dict(stage=stage,tasks=calls//46,total=len(seeds),predictors=calls,seconds=time.perf_counter()-begin,
                       query_targets_accessed=False),flush=True)
    environment()
    rows,files,checks=verify_predictions(root,out,p)
    assert suite.gate(root)==hashes
    for name,value in [('rows.json',rows),('files.json',files),('environments.json',environments),('costs.json',cost_table(rows,p['methods']))]:
        exclusive(out/name,value)
    exclusive(out/'before_query_manifest.json',dict(protocol_sha256=ph,source_sha256=hashes,rows_sha256=sha(out/'rows.json'),
        files_sha256=sha(out/'files.json'),costs_sha256=sha(out/'costs.json'),checks=checks,query_targets_accessed=False))
    result=dict(passed=True,tasks=len(seeds),predictors=calls,checks=checks,failures=sum(r['metadata']['execution_failed'] for r in rows),
        query_targets_accessed=False,seconds=time.perf_counter()-begin,core_research_goal_complete=False,
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','rows.json','files.json','environments.json','costs.json','before_query_manifest.json']})
    exclusive(out/'summary.json',result);print(result,flush=True)


def score(root,out,stage):
    hashes=suite.gate(root);folder=out.parent/f'{stage}_predictions_v1';s=suite.old.complete(folder)
    p=read(folder/'protocol.json');assert p['source_sha256']==hashes and p['design_sha256']==sha(root/suite.DESIGN)
    assert p['seeds']==(suite.OLD_SEEDS if stage=='preflight' else suite.PILOT_SEEDS)
    rows,files,checks=verify_predictions(root,folder,p)
    assert rows==read(folder/'rows.json') and files==read(folder/'files.json') and checks==s['checks']
    seal=read(folder/'before_query_manifest.json');assert not seal['query_targets_accessed']
    for key,name in [('files_sha256','files.json'),('rows_sha256','rows.json'),('costs_sha256','costs.json')]:assert sha(folder/name)==seal[key]
    costs=read(folder/'costs.json');assert costs==cost_table(rows,p['methods'])
    names=p['methods'];seeds=p['seeds'];pi=names.index(suite.PRIMARY);ci=[j for j in range(46) if j!=pi]
    assert pi==37 and len(ci)==45
    protocol=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),
        prediction_summary_sha256=sha(folder/'summary.json'),before_query_manifest_sha256=sha(folder/'before_query_manifest.json'),
        costs_sha256=sha(folder/'costs.json'),methods=names,seeds=seeds,primary=suite.PRIMARY,metrics=METRICS,primary_metric='mse257',
        primary_family=[names[j] for j in ci],bootstrap=dict(repetitions=20000,seed=307193,block=64,adjusted_quantile=1-.05/45,
            quantile_method='linear',resampling_unit='task'),query_targets_accessed=True,development_pilot=stage=='pilot',
        core_research_goal_complete=False,inference='Approximate paired percentile bootstrap; all45 main contrasts retained')
    exclusive(out/'protocol.json',protocol)
    begin=time.perf_counter();n=len(seeds);risk=np.empty((n,46,4));times=np.empty((n,46));failed=np.zeros((n,46),bool)
    lookup={(r['seed'],r['method']):r for r in rows};scalar_gap=truth_gap=0.
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
                scalar_gap=max(scalar_gap,float(np.max(abs(risk[si,mi]-independent))));assert scalar_gap<2e-12
            times[si,mi]=r['seconds'];failed[si,mi]=r['metadata']['execution_failed']
    assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1))
    diff=risk[:,pi:pi+1,:]-risk[:,ci,:];flat=diff.reshape(n,-1)
    stats.selftest();boot,digest=stats.bootstrap_means(flat,20000,307193,64)
    rng=np.random.default_rng(307193);boot_gap=0.
    for first in range(0,20000,7):
        size=min(7,20000-first);indices=rng.integers(n,size=(size,n),dtype=np.int64)
        other=flat[indices].mean(1)
        boot_gap=max(boot_gap,float(np.max(abs(other-boot[first:first+size]))));assert boot_gap<2e-12
    boot=boot.reshape(20000,45,4)
    with (out/'task_metrics.npz').open('xb') as f:np.savez_compressed(f,seeds=seeds,methods=names,metric_names=METRICS,risk=risk,seconds=times,failures=failed)
    with (out/'bootstrap_means.npz').open('xb') as f:np.savez_compressed(f,means=boot)
    comparisons=[]
    for j,mi in enumerate(ci):
        for k,metric in enumerate(METRICS):
            d=diff[:,j,k];bounds=np.quantile(boot[:,j,k],[.025,.975])
            upper=float(np.quantile(boot[:,j,k],1-.05/45)) if k==0 else None
            comparisons.append(dict(control=names[mi],metric=metric,mean_difference=float(d.mean()),sample_sd=float(d.std(ddof=1)),
                descriptive95=bounds.tolist(),bonferroni_upper=upper,adjusted_negative=upper<0 if upper is not None else None,
                improved=int((d<0).sum()),equal=int((d==0).sum()),worse=int((d>0).sum())))
    cost_by_name={r['method']:r for r in costs['methods']};methods=[]
    for j,name in enumerate(names):
        c=cost_by_name[name]
        assert math.isclose(c['mean_seconds'],float(times[:,j].mean()),rel_tol=1e-13,abs_tol=1e-14)
        methods.append(dict(method=name,metrics={metric:float(risk[:,j,k].mean()) for k,metric in enumerate(METRICS)},
            mean_seconds=c['mean_seconds'],actual_ratio_to_primary=c['ratio_to_primary'],budget_class=c['budget_class'],
            failures=int(failed[:,j].sum()),maximum_reported_byte_fields=c['maximum_reported_byte_fields']))
    for name,value in [('methods.json',methods),('comparisons.json',comparisons)]:exclusive(out/name,value)
    assert suite.gate(root)==hashes
    result=dict(passed=True,tasks=n,predictors=n*46,comparisons=180,main_family=45,
        adjusted_negative_comparisons=sum(r['adjusted_negative'] for r in comparisons if r['metric']=='mse257'),
        query_targets_accessed=True,independent_scalar_risk_passed=True,maximum_truth_gap=truth_gap,maximum_scalar_risk_gap=scalar_gap,
        all_bootstrap_means_checked=20000*180,maximum_bootstrap_gap=boot_gap,bootstrap_index_sha256=digest,
        numerical_failures=int(failed.sum()),core_research_goal_complete=False,stage=stage,seconds=time.perf_counter()-begin,
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','task_metrics.npz','bootstrap_means.npz','methods.json','comparisons.json']})
    exclusive(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','pilot'],required=True);ap.add_argument('--phase',choices=['predict','score'],required=True)
    args=ap.parse_args();root=args.project.resolve();suffix='predictions' if args.phase=='predict' else 'evaluation'
    out=root/suite.BASE/f'{args.stage}_{suffix}_v1';out.mkdir(parents=True,exist_ok=False)
    try:(predict if args.phase=='predict' else score)(root,out,args.stage)
    except BaseException as exc:
        exclusive(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

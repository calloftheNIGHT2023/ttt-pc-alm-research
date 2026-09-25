"""328 support-only prediction phase; compact index and durable full call logs."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import online_credit_fresh_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_independent_hybrid_memory import observations
from evaluate_complete_credit_mode_geometry_v1 import read,sha


def exclusive(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())


def signature(x,v,q):
    h=hashlib.sha256()
    for a in [x,v,q]:h.update(a.tobytes())
    return h.hexdigest()


def validate(arrays):
    for n,a in arrays.items():
        if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all(),n
    for n in suite.FIELDS:
        a=arrays[n];assert a.shape==(257,) and a.dtype==np.float64 and np.all((a>=0)&(a<=1))
    assert arrays['x_observed'].shape==arrays['v_observed'].shape==(4,)
    assert np.array_equal(arrays['q_observed'],np.linspace(0,1,257))
    assert not np.isin(arrays['q_observed'],arrays['x_observed']).any()


def byte_fields(meta,prefix=''):
    result={}
    for n,value in meta.items():
        if isinstance(value,dict):result.update(byte_fields(value,prefix+n+'.'))
        elif 'bytes' in n and isinstance(value,(int,float)) and not isinstance(value,bool):result[prefix+n]=value
    return result


def verify(root,folder,p):
    rows=[];files={};counts=Counter();ph=sha(folder/'protocol.json');byname={c['name']:c for c in p['configs']}
    for seed in p['seeds']:
        commit=read(folder/str(seed)/'commit.json')
        assert commit['seed']==seed and commit['protocol_sha256']==ph and not commit['query_targets_accessed']
        expected=[p['configs'][int(i)]['name'] for i in np.random.default_rng(np.random.SeedSequence([328929,seed])).permutation(51)]
        assert [r['method'] for r in commit['rows']]==expected
        for i,r in enumerate(commit['rows']):
            assert r['seed']==seed and r['order']==i and r['seconds']>0 and math.isfinite(r['seconds'])
            assert sha(folder/r['file'])==r['sha256'] and sha(folder/r['metadata_file'])==r['metadata_sha256']
            log=read(folder/r['metadata_file']);meta=log['metadata']
            assert log['seed']==seed and log['method']==r['method'] and log['order']==i and log['protocol_sha256']==ph
            assert not meta['query_targets_accessed'] and meta['charged_complete_seconds']==r['seconds'] and meta['execution_failed']==r['execution_failed']
            assert byte_fields(meta)==r['named_byte_fields']
            assert sha(folder/r['start_file'])==r['start_sha256']
            start=read(folder/r['start_file']);assert start['seed']==seed and start['method']==r['method'] and start['protocol_sha256']==ph and start['order']==i
            with np.load(folder/r['file'],allow_pickle=False) as z:
                a={n:z[n] for n in z.files};validate(a)
                assert signature(a['x_observed'],a['v_observed'],a['q_observed'])==commit['observed_sha256']
                if p['stage']=='preflight':counts['frozen_arrays']+=suite.reference(root,byname[r['method']],seed,a,meta)
                counts['numeric_arrays']+=sum(np.issubdtype(v.dtype,np.number) for v in a.values())
            assert r['file'] not in files;files[r['file']]=r['sha256'];rows.append(r);counts['predictors']+=1
        assert commit['failures']==sum(r['execution_failed'] for r in commit['rows'])
        assert commit['charged_seconds']==math.fsum(r['seconds'] for r in commit['rows']);counts['tasks']+=1
    assert len(rows)==len(files)==len(p['seeds'])*51
    return rows,files,dict(counts)


def cost_table(rows,names):
    means={n:math.fsum(r['seconds'] for r in rows if r['method']==n)/sum(r['method']==n for r in rows) for n in names}
    table=[]
    for n in names:
        rr=[r for r in rows if r['method']==n];tt=[r['seconds'] for r in rr];fields={}
        for r in rr:
            for k,v in r['named_byte_fields'].items():fields[k]=max(fields.get(k,0),v)
        table.append(dict(method=n,mean_seconds=means[n],median_seconds=float(np.median(tt)),p90_seconds=float(np.quantile(tt,.9)),maximum_seconds=max(tt),
            failures=sum(r['execution_failed'] for r in rr),maximum_named_byte_fields=fields,
            ratios_to_candidates={c:means[n]/means[c] for c in suite.CANDIDATES}))
    return dict(methods=table,candidates={c:dict(mean_seconds=means[c],within_budget=[n for n in names if means[n]<=means[c]],
        sensitivity_110_percent=[n for n in names if means[n]<=1.1*means[c]]) for c in suite.CANDIDATES},
        all_methods_retained=True,query_quality_used=False,state_scope='Named byte fields only; separate resource calibration measures process/traced peak')


def run(root,out,stage):
    hashes=suite.gate(root);cfgs=suite.resources.catalogue(root);tested=root/suite.BASE/'tests_v1'
    tests=suite.resources.old.complete(tested)
    assert read(tested/'protocol.json')['source_sha256']==hashes
    if stage=='pilot':
        ps=suite.resources.old.complete(out.parent/'preflight_predictions_v1');es=suite.resources.old.complete(out.parent/'preflight_evaluation_v1')
        assert ps['predictors']==153 and es['independent_scalar_risk_passed']
        audited=read(out.parent/'preflight_audit_v1/summary.json')
        assert audited['passed'] and audited['counts']['predictors']==153
        assert audited['prediction_summary_sha256']==sha(out.parent/'preflight_predictions_v1/summary.json')
        assert audited['evaluation_summary_sha256']==sha(out.parent/'preflight_evaluation_v1/summary.json')
        assert read(out.parent/'preflight_predictions_v1/protocol.json')['source_sha256']==hashes
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tick=time.perf_counter();loaded,checkpoint=suite.resources.old.resources.legacy.oldfit.meta.load(root);loading=time.perf_counter()-tick
    cal=root/suite.resources.BASE/'calibration_v1';assert checkpoint==read(cal/'protocol.json')['checkpoint_manifest']
    seeds=suite.OLD_SEEDS if stage=='preflight' else suite.PILOT_SEEDS
    p=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),configs=cfgs,seeds=seeds,
        methods=[c['name'] for c in cfgs],candidates=suite.CANDIDATES,query_points=257,observations_per_task=4,order_seed=328929,
        query_targets_accessed=False,posterior_reference_accessed=False,checkpoint_manifest=checkpoint,model_loading_seconds=loading,
        preloaded_model_bytes=sum(v['shared_model_bytes'] for v in checkpoint.values()),threads=dict(blas=1,omp=1,torch=1),trace=False,
        test_summary_sha256=sha(tested/'summary.json'),resource_audit_sha256=sha(root/suite.resources.BASE/'audit_v1/summary.json'),
        frozen_resource_selection=read(cal/'selection.json'),free_disk_floor_bytes=20*2**30,quality_stopping=False,
        failure_policy='Existing guarded numerical failure: charge failed attempt and fixed zero-bias prediction; no retry or task exclusion',
        inference='Fresh controlled development replication, not final paper confirmation or general TTT/downstream claim')
    exclusive(out/'protocol.json',p);ph=sha(out/'protocol.json');begin=time.perf_counter();calls=0;environments=[]
    inputs=suite.resources.old.observed_inputs(root,suite.resources.old.frozen_inputs(root),seeds) if stage=='preflight' else None
    def environment():
        while True:
            e=suite.resources.old.environment_snapshot(root);e['waiting_for_other_research']=bool(e['other_research_or_git_pack_processes'])
            environments.append(e);exclusive(out/f'environment_{len(environments):05d}.json',e)
            if not e['waiting_for_other_research']:return
            print(dict(phase='interference_wait',processes=e['other_research_or_git_pack_processes']),flush=True);time.sleep(10)
    with discovery_box(.12):
        for seed in seeds:
            environment();assert shutil.disk_usage(out).free>=p['free_disk_floor_bytes'];directory=out/str(seed);directory.mkdir();rows=[];wall=time.perf_counter()
            if inputs is not None:x,v,q=inputs[seed]
            else:
                xx,vv=observations(seed);x,v=xx[:4].copy(),vv[:4].copy();q=np.linspace(0,1,257);del xx,vv
            observed=signature(x,v,q)
            for i in np.random.default_rng(np.random.SeedSequence([328929,seed])).permutation(51):
                cfg=cfgs[int(i)];name=cfg['name'];order=len(rows);sf=directory/(name+'.start.json');mf=directory/(name+'.call.json')
                exclusive(sf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,unix_time=time.time()))
                a,m,seconds=suite.resources.invoke(cfg,x,v,q,seed,loaded)
                if m['execution_failed'] and any(t in m.get('failure_message','') for t in ['Global BP called by local-only candidate','Global solver or BP entered branch selector']):
                    raise RuntimeError('Forbidden-credit/solver gate violation is not a recoverable numerical failure')
                assert signature(x,v,q)==observed
                a.update(x_observed=x.copy(),v_observed=v.copy(),q_observed=q.copy());validate(a)
                exclusive(mf,dict(seed=seed,method=name,order=order,protocol_sha256=ph,metadata=m))
                path=directory/(name+'.npz')
                with path.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
                rows.append(dict(seed=seed,method=name,order=order,file=str(path.relative_to(out)),sha256=sha(path),
                    metadata_file=str(mf.relative_to(out)),metadata_sha256=sha(mf),start_file=str(sf.relative_to(out)),start_sha256=sha(sf),
                    seconds=seconds,execution_failed=m['execution_failed'],named_byte_fields=byte_fields(m)))
                calls+=1;del a,m
            exclusive(directory/'commit.json',dict(seed=seed,protocol_sha256=ph,rows=rows,observed_sha256=observed,
                failures=sum(r['execution_failed'] for r in rows),charged_seconds=math.fsum(r['seconds'] for r in rows),
                wall_with_io_seconds=time.perf_counter()-wall,query_targets_accessed=False))
            print(dict(stage=stage,tasks=calls//51,total=len(seeds),predictors=calls,seconds=time.perf_counter()-begin,query_targets_accessed=False),flush=True)
    environment();rows,files,counts=verify(root,out,p);assert suite.gate(root)==hashes
    for n,value in [('rows.json',rows),('files.json',files),('environments.json',environments),('costs.json',cost_table(rows,p['methods']))]:exclusive(out/n,value)
    exclusive(out/'before_query_manifest.json',dict(protocol_sha256=ph,source_sha256=hashes,query_targets_accessed=False,
        rows_sha256=sha(out/'rows.json'),files_sha256=sha(out/'files.json'),costs_sha256=sha(out/'costs.json'),checks=counts))
    result=dict(passed=True,tasks=len(seeds),predictors=calls,checks=counts,seconds=time.perf_counter()-begin,query_targets_accessed=False,
        failures=sum(r['execution_failed'] for r in rows),core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','files.json','environments.json','costs.json','before_query_manifest.json']})
    exclusive(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','pilot'],required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2]
    out=root/suite.BASE/f'{a.stage}_predictions_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,a.stage)
    except Exception:exclusive(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

"""288: independent scalar projection, piecewise teacher and all bootstrap checks."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import time

import numpy as np
from known_range_projection_pipeline_v1 import (FIELDS,METRICS,complete,gate,now,paths,read,save,sha,source_hashes)
from audit_probe_credit_confirmation_evaluation_v6 import independent_truth


def run(root,stage,out,hashes):
    loc = paths(root,stage)
    pred, ev = loc['projected'], loc['evaluation']
    ps, es = complete(pred), complete(ev)
    p, ep = read(pred/'protocol.json'), read(ev/'protocol.json')
    assert p['source_sha256']==ep['source_sha256']==hashes
    assert ep['prediction_summary_sha256']==sha(pred/'summary.json')
    assert p['original_before_query_manifest_sha256']==sha(loc['pred']/'before_query_manifest.json')
    assert p['original_protocol_sha256']==sha(loc['pred']/'protocol.json')
    assert p['original_summary_sha256']==sha(loc['pred']/'summary.json')
    assert p['final_original_seal_sha256']==sha(root/'results/round_287_audit_v6.json')
    assert ep['raw_evaluation_summary_sha256']==sha(loc['raw_eval']/'summary.json')
    assert ep['supplementary_plan_sha256']==p['supplementary_plan_sha256']==sha(root/'outputs/ttt-pc-alm-research/288_full_projection_protocol.md')
    names,seeds,controls = ep['methods'],ep['seeds'],ep['controls']
    n=len(seeds)
    assert n==(2 if stage=='preflight' else 8192) and ps['predictors']==n*27
    assert names==p['methods'] and seeds==p['seeds'] and ep['metrics']==METRICS
    assert ep['bootstrap']==dict(repetitions=100000,seed=288193,block=64,quantile_method='linear',descriptive_quantiles=[.025,.975],adjusted_upper_quantile=.998)
    save(out/'protocol.json',dict(stage=stage,created_utc=now(),source_sha256=hashes,supplementary_analysis=True,
         prediction_summary_sha256=sha(pred/'summary.json'),evaluation_summary_sha256=sha(ev/'summary.json'),
         query_targets_accessed=True,scope='All original arrays, independent scalar projection, piecewise teacher, every risk/summary/bootstrap value',
         risk_tolerance=dict(atol=1e-12,rtol=1e-11),bootstrap_tolerance=dict(atol=1e-12,rtol=1e-11)))
    begin=time.perf_counter()
    with np.load(ev/'task_metrics.npz',allow_pickle=False) as z:
        assert z['seeds'].tolist()==seeds and z['methods'].tolist()==names and z['metric_names'].tolist()==METRICS
        risk,raw_risk,times,projection_times,failures=[z[k].copy() for k in ['risk','raw_risk','original_seconds','projection_seconds','failures']]
    with np.load(loc['raw_eval']/'task_metrics.npz',allow_pickle=False) as z:
        for key,actual in [('risk',raw_risk),('seconds',times),('failures',failures)]:
            np.testing.assert_array_equal(actual,z[key])
    before=read(loc['pred']/'before_query_manifest.json')
    assert len(before['task_commits'])==n
    records=defaultdict(list)
    calls=0
    maxrisk=0.
    maxraw=0.
    maxdominance=0.
    numeric_arrays=0
    expected_first=0
    for block in read(pred/'blocks.json'):
        first,count=block['first'],block['tasks']
        assert first==expected_first
        expected_first+=count
        with np.load(pred/block['data_file'],allow_pickle=False) as z:
            values=z['projected'].copy()
            assert z['seeds'].tolist()==seeds[first:first+count] and z['methods'].tolist()==names and z['fields'].tolist()==FIELDS
            np.testing.assert_array_equal(z['projection_seconds'],projection_times[first:first+count])
        local=read(pred/block['records_file'])
        assert len(local)==count and values.shape==(count,27,2,257)
        for i,record in enumerate(local):
            seed=seeds[first+i]
            assert record['seed']==seed and record['original_commit']==before['task_commits'][first+i]
            original_commit=loc['pred']/record['original_commit']['file']
            assert sha(original_commit)==record['original_commit']['sha256']
            commit=read(original_commit)
            assert commit['seed']==seed and commit['methods']==names
            assert commit['rows_file']==record['original_rows_file']
            assert sha(loc['pred']/commit['rows_file'])==record['original_rows_sha256']==commit['files'][commit['rows_file']]
            rows={r['method']:r for r in read(loc['pred']/commit['rows_file'])}
            truth=independent_truth(np.linspace(0,1,257),np.random.default_rng(seed).uniform(-.12,.12,4))
            assert np.all((truth>=0)&(truth<=1))
            assert len(record['methods'])==27
            for j,method in enumerate(record['methods']):
                name=names[j]
                assert method['method']==name
                row=rows[name]
                assert row['file']==method['original_file']
                assert sha(loc['pred']/row['file'])==method['original_sha256']==row['sha256']==commit['files'][row['file']]
                assert times[first+i,j]==row['metadata']['charged_complete_seconds']
                assert failures[first+i,j]==row['metadata']['execution_failed']==method['original_execution_failed']
                expected_risk=[]
                expected_raw=[]
                with np.load(loc['pred']/row['file'],allow_pickle=False) as z:
                    for f,field in enumerate(FIELDS):
                        original=z[field]
                        expected=np.fromiter((0. if v<0 else 1. if v>1 else float(v) for v in original),dtype=np.float64,count=257)
                        actual=values[i,j,f]
                        assert expected.tobytes()==actual.tobytes(),(seed,name,field)
                        diagnostic=dict(below_zero=sum(float(v)<0 for v in original),above_one=sum(float(v)>1 for v in original),
                            changed_values=sum(a!=b for a,b in zip(original.tolist(),expected.tolist())),
                            bitwise_unchanged=original.tobytes()==expected.tobytes(),original_min=float(min(original)),
                            original_max=float(max(original)),maximum_absolute_change=float(max(abs(a-b) for a,b in zip(original,expected))))
                        assert diagnostic==method['fields'][field]
                        records[name,field].append(diagnostic)
                        delta=expected-truth
                        rawdelta=original-truth
                        expected_risk.extend([float(np.dot(delta,delta)/257),float(np.dot(delta[::2],delta[::2])/129)])
                        expected_raw.extend([float(np.dot(rawdelta,rawdelta)/257),float(np.dot(rawdelta[::2],rawdelta[::2])/129)])
                        strong_gap=rawdelta**2-delta**2-(original-expected)**2
                        tolerance=1e-12+1e-11*abs(rawdelta**2)
                        assert np.all(strong_gap>=-tolerance)
                        maxdominance=max(maxdominance,float(np.max(delta**2-rawdelta**2)))
                        numeric_arrays+=1
                np.testing.assert_allclose(expected_risk,risk[first+i,j],rtol=1e-11,atol=1e-12)
                np.testing.assert_allclose(expected_raw,raw_risk[first+i,j],rtol=1e-11,atol=1e-12)
                maxrisk=max(maxrisk,float(np.max(abs(np.array(expected_risk)-risk[first+i,j]))))
                maxraw=max(maxraw,float(np.max(abs(np.array(expected_raw)-raw_risk[first+i,j]))))
                calls+=1
        print(dict(stage=stage,phase='independent_projection_risk',tasks=first+count,total=n,seconds=time.perf_counter()-begin),flush=True)
    assert expected_first==n and calls==n*27 and numeric_arrays==n*54
    primary=names.index(ep['primary'])
    indices=[names.index(name) for name in controls]
    differences=risk[:,primary:primary+1]-risk[:,indices]
    with np.load(ev/'bootstrap_means.npz',allow_pickle=False) as z:
        boot=z['means'].copy()
        assert z['controls'].tolist()==controls and z['metrics'].tolist()==METRICS
    assert boot.shape==(100000,26,4)
    rng=np.random.default_rng(288193)
    digest=hashlib.sha256()
    flat=boot.reshape(100000,-1)
    d=differences.reshape(n,-1)
    maxbootstrap=0.
    for first in range(0,100000,37):
        count=min(37,100000-first)
        indices=rng.integers(n,size=(count,n),dtype=np.int64)
        digest.update(indices.tobytes())
        weights=np.zeros((count,n),dtype=np.float64)
        for row in range(count):
            np.add.at(weights[row],indices[row],1./n)
        expected=weights@d
        np.testing.assert_allclose(expected,flat[first:first+count],rtol=1e-11,atol=1e-12)
        maxbootstrap=max(maxbootstrap,float(np.max(abs(expected-flat[first:first+count]))))
    assert digest.hexdigest()==es['bootstrap_index_sha256']
    comparisons=read(ev/'comparisons.json')
    lookup={(r['control'],r['metric']):r for r in comparisons}
    assert len(lookup)==len(comparisons)==104
    passed=0
    for j,name in enumerate(controls):
        for k,metric in enumerate(METRICS):
            r=lookup[name,metric]
            dd=differences[:,j,k]
            assert r['tasks']==n and r['mean_difference']==float(dd.mean()) and r['paired_sample_sd']==float(dd.std(ddof=1))
            assert r['descriptive95']==np.quantile(boot[:,j,k],[.025,.975],method='linear').tolist()
            assert [r['improved_tasks'],r['equal_tasks'],r['worse_tasks']]==[int(np.sum(dd<0)),int(np.sum(dd==0)),int(np.sum(dd>0))]
            main=name!='probe_all_alm64' and k==0
            assert r['supplementary_main_comparison']==main
            if main:
                upper=float(np.quantile(boot[:,j,k],.998,method='linear'))
                assert r['adjusted_upper']==upper and r['upper_below_zero']==(upper<0)
                passed+=upper<0
            else:
                assert r['adjusted_upper'] is None and r['upper_below_zero'] is None
    methods=read(ev/'methods.json')
    assert [r['method'] for r in methods]==names
    for j,r in enumerate(methods):
        for k,metric in enumerate(METRICS):
            assert r['metrics'][metric]==dict(raw_mean=float(raw_risk[:,j,k].mean()),projected_mean=float(risk[:,j,k].mean()),
                mean_change=float((risk[:,j,k]-raw_risk[:,j,k]).mean()),projected_task_sd=float(risk[:,j,k].std(ddof=1)))
        assert r['original_complete_mean_seconds']==float(times[:,j].mean())
        assert r['isolated_projection_pair_mean_seconds']==float(projection_times[:,j].mean())
        assert r['projected_output_numeric_bytes']==4112 and r['failures']==int(failures[:,j].sum())
        for field in FIELDS:
            rr=records[names[j],field]
            expected=dict(below_zero=sum(t['below_zero'] for t in rr),above_one=sum(t['above_one'] for t in rr),
                changed_values=sum(t['changed_values'] for t in rr),changed_tasks=sum(t['changed_values']>0 for t in rr),
                all_tasks_bitwise_unchanged=all(t['bitwise_unchanged'] for t in rr),
                original_min=min(t['original_min'] for t in rr),original_max=max(t['original_max'] for t in rr),
                maximum_absolute_change=max(t['maximum_absolute_change'] for t in rr))
            assert expected==r['projection'][field]
    assert es['supplementary_negative_upper_bounds']==passed and not es['core_research_goal_complete']
    raw_es=read(loc['raw_eval']/'summary.json')
    assert es['original_main_negative_upper_bounds']==raw_es['main_adjusted_negative_upper_bounds']
    assert source_hashes(root)==hashes
    result=dict(passed=True,stage=stage,supplementary_analysis=True,tasks=n,predictors=calls,arrays=numeric_arrays,
         comparisons=104,method_tables=27,bootstrap_replicates=100000,bootstrap_scalar_means=10400000,
         maximum_independent_risk_gap=maxrisk,maximum_independent_original_risk_gap=maxraw,
         maximum_independent_bootstrap_gap=maxbootstrap,maximum_pointwise_risk_increase=maxdominance,
         supplementary_negative_upper_bounds=int(passed),original_main_negative_upper_bounds=es['original_main_negative_upper_bounds'],
         query_targets_accessed=True,core_research_goal_complete=False,seconds=time.perf_counter()-begin,
         evaluation_summary_sha256=sha(ev/'summary.json'),outputs_sha256={'protocol.json':sha(out/'protocol.json')},
         next='Report all original/projected results, limitations and resource separation; scientific goal remains open')
    save(out/'summary.json',result)
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--stage',choices=['preflight','confirmation'],required=True)
    args=parser.parse_args()
    root=args.project.resolve()
    hashes=gate(root,args.stage)
    out=paths(root,args.stage)['audit']
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,args.stage,out,hashes)
    except BaseException as exc:
        save(out/'failure.json',dict(utc=now(),error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__': main()

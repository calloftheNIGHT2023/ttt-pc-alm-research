"""307 independent full-result audit; run only after all pilot predictions/score."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import statistics
import numpy as np
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def close(a,b,tol=2e-12):
    assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)


def quantile(values,p):
    a=sorted(float(x) for x in values);z=(len(a)-1)*p;i=math.floor(z)
    return a[i]+(z-i)*(a[min(i+1,len(a)-1)]-a[i])


def run(root,out):
    base=root/'results/online_stasis_fresh_pilot'
    predictions=base/'pilot_predictions_v1';evaluation=base/'pilot_evaluation_v1'
    ss=read(predictions/'summary.json');es=read(evaluation/'summary.json')
    assert ss['passed'] and es['passed'] and ss['tasks']==es['tasks']==256
    assert ss['predictors']==es['predictors']==11776 and es['main_family']==45
    for folder,summary in [(predictions,ss),(evaluation,es)]:
        for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    p=read(predictions/'protocol.json');ep=read(evaluation/'protocol.json')
    assert p['seeds']==list(range(307000000,307000256)) and p['methods']==ep['methods']
    assert ep['prediction_summary_sha256']==sha(predictions/'summary.json')
    assert ep['before_query_manifest_sha256']==sha(predictions/'before_query_manifest.json')
    assert ep['costs_sha256']==sha(predictions/'costs.json')
    assert p['design_sha256']==ep['design_sha256']==sha(root/'outputs/ttt-pc-alm-research/307_online_stasis_fresh_pilot_protocol.md')
    for name,digest in p['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    assert ep['source_sha256']==p['source_sha256']
    assert ep['bootstrap']==dict(repetitions=20000,seed=307193,block=64,adjusted_quantile=1-.05/45,quantile_method='linear',resampling_unit='task')
    rows=read(predictions/'rows.json');files=read(predictions/'files.json');seal=read(predictions/'before_query_manifest.json')
    assert not seal['query_targets_accessed'] and not p['query_targets_accessed']
    for key,name in [('rows_sha256','rows.json'),('files_sha256','files.json'),('costs_sha256','costs.json')]:assert seal[key]==sha(predictions/name)
    names=p['methods'];pi=names.index(p['primary']);ci=[i for i in range(46) if i!=pi]
    assert pi==37 and len(names)==46
    index={(r['seed'],r['method']):r for r in rows};assert len(index)==len(rows)==len(files)==11776
    with np.load(evaluation/'task_metrics.npz',allow_pickle=False) as z:
        risk=z['risk'];times=z['seconds'];failures=z['failures']
        assert z['seeds'].tolist()==p['seeds'] and z['methods'].tolist()==names
        assert z['metric_names'].tolist()==['mse257','mse129','point_mse257','point_mse129']
    assert risk.shape==(256,46,4) and times.shape==failures.shape==(256,46)
    assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1))
    assert np.isfinite(times).all() and np.all(times>0) and failures.dtype==np.bool_
    assert ss['failures']==es['numerical_failures']==int(failures.sum())
    assert es['comparisons']==180 and es['all_bootstrap_means_checked']==3600000
    assert p['threads']==dict(blas=1,omp=1,torch=1) and p['trace'] is False
    counts=Counter();scalar_gap=0.;support_gap=0.
    def scalar_forward(value,biases):
        for bias in biases:
            arg=float(value)+float(bias)
            value=max(0.,min(2.*arg,2.-2.*arg))
        return value
    # Independent scalar fold and sum; no fitting or tuning in this audit.
    for si,seed in enumerate(p['seeds']):
        commit=read(predictions/str(seed)/'commit.json')
        assert commit['seed']==seed and not commit['query_targets_accessed']
        assert commit['protocol_sha256']==sha(predictions/'protocol.json')
        assert commit['failures']==sum(r['metadata']['execution_failed'] for r in commit['rows'])
        assert commit['charged_seconds']==sum(r['seconds'] for r in commit['rows'])
        expected=[names[int(i)] for i in np.random.default_rng(np.random.SeedSequence([307929,seed])).permutation(46)]
        assert [r['method'] for r in commit['rows']]==expected
        generator=np.random.default_rng(seed)
        biases=generator.uniform(-.12,.12,4)
        expected_x=generator.uniform(0,1,24)[:4]
        noise=np.random.default_rng(seed+19000000).uniform(-.001,.001,24)[:4]
        expected_v=np.array([scalar_forward(x,biases) for x in expected_x])+noise
        truth=[scalar_forward(i/256,biases) for i in range(257)]
        for mi,name in enumerate(names):
            r=index[seed,name]
            assert sha(predictions/r['file'])==r['sha256']==files[r['file']]
            assert r['seconds']==times[si,mi] and r['metadata']['execution_failed']==bool(failures[si,mi])
            assert r['seconds']==r['metadata']['charged_complete_seconds']
            assert not r['metadata']['query_targets_accessed']
            assert r==commit['rows'][r['order']]
            for suffix in ['start','call']:
                file=predictions/r[suffix+'_file'];assert sha(file)==r[suffix+'_sha256']
                j=read(file);assert j['seed']==seed and j['method']==name and j['order']==r['order']
                assert j['protocol_sha256']==sha(predictions/'protocol.json')
                if suffix=='call':assert j['metadata']==r['metadata']
            with np.load(predictions/r['file'],allow_pickle=False) as z:
                for key in z.files:
                    value=z[key]
                    if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),(seed,name,key)
                observed=hashlib.sha256()
                for key in ['x_observed','v_observed','q_observed']:
                    value=z[key];assert np.isfinite(value).all()
                    observed.update(value.tobytes())
                assert observed.hexdigest()==commit['observed_sha256']
                assert z['x_observed'].shape==z['v_observed'].shape==(4,)
                assert np.array_equal(z['x_observed'],expected_x)
                support_gap=max(support_gap,float(np.max(abs(z['v_observed']-expected_v))))
                assert support_gap<2e-12
                counts['support_generators']+=1
                assert np.array_equal(z['q_observed'],np.linspace(0,1,257))
                assert not np.isin(z['q_observed'],z['x_observed']).any()
                for fi,field in enumerate(['prediction','point_prediction']):
                    a=z[field];assert a.shape==(257,) and np.all((a>=0)&(a<=1))
                    for stride in [1,2]:
                        expected_risk=math.fsum((float(a[i])-truth[i])**2 for i in range(0,257,stride))/len(range(0,257,stride))
                        scalar_gap=max(scalar_gap,abs(expected_risk-risk[si,mi,2*fi+stride-1]))
                        assert scalar_gap<2e-12
                        counts['scalar_risks']+=1
            counts['predictors']+=1
    costs=read(predictions/'costs.json');costs_by_name={r['method']:r for r in costs['methods']}
    cap=statistics.fmean(times[:,pi])
    for i,name in enumerate(names):
        c=costs_by_name[name];mean=statistics.fmean(times[:,i]);ratio=mean/cap
        close(mean,c['mean_seconds']);close(ratio,c['ratio_to_primary'])
        close(statistics.median(times[:,i]),c['median_seconds']);close(quantile(times[:,i],.9),c['p90_seconds'])
        assert c['budget_class']==('1.00' if mean<=cap else '1.10' if mean<=1.10*cap else 'over_1.10')
        assert c['failures']==int(failures[:,i].sum())
    close(costs['primary_mean_seconds'],cap)
    assert costs['within_budget']==[n for n in names if costs_by_name[n]['budget_class']=='1.00']
    assert costs['sensitivity_110_percent']==[n for n in names if costs_by_name[n]['budget_class']!='over_1.10']
    assert costs['all_methods_retained'] and not costs['selection_uses_query_quality']
    table=read(evaluation/'methods.json');comparisons=read(evaluation/'comparisons.json')
    assert [r['method'] for r in table]==names and len(comparisons)==180
    for i,r in enumerate(table):
        for j,metric in enumerate(['mse257','mse129','point_mse257','point_mse129']):
            close(r['metrics'][metric],statistics.fmean(risk[:,i,j]))
        assert r['maximum_reported_byte_fields']==costs_by_name[r['method']]['maximum_reported_byte_fields']
    diffs=risk[:,pi:pi+1,:]-risk[:,ci,:]
    with np.load(evaluation/'bootstrap_means.npz',allow_pickle=False) as z:boot=z['means']
    assert boot.shape==(20000,45,4)
    rng=np.random.default_rng(307193);boot_gap=0.
    for first in range(0,20000,13):
        size=min(13,20000-first);ids=rng.integers(256,size=(size,256),dtype=np.int64)
        recomputed=diffs[ids].mean(axis=1)
        boot_gap=max(boot_gap,float(np.max(abs(recomputed-boot[first:first+size]))));assert boot_gap<2e-12
    for j,mi in enumerate(ci):
        for k,metric in enumerate(['mse257','mse129','point_mse257','point_mse129']):
            r=comparisons[j*4+k];d=diffs[:,j,k]
            assert r['control']==names[mi] and r['metric']==metric
            close(r['mean_difference'],statistics.fmean(d));close(r['sample_sd'],statistics.stdev(d))
            for t,percent in enumerate([.025,.975]):close(r['descriptive95'][t],quantile(boot[:,j,k],percent))
            if k==0:
                upper=quantile(boot[:,j,k],1-.05/45);close(r['bonferroni_upper'],upper)
                assert r['adjusted_negative']==(upper<0)
            else:assert r['bonferroni_upper'] is None and r['adjusted_negative'] is None
            assert [r['improved'],r['equal'],r['worse']]==[int((d<0).sum()),int((d==0).sum()),int((d>0).sum())]
            counts['comparisons']+=1
    assert es['adjusted_negative_comparisons']==sum(r['adjusted_negative'] for r in comparisons if r['metric']=='mse257')
    result=dict(passed=True,counts=dict(counts),maximum_scalar_risk_gap=scalar_gap,maximum_bootstrap_gap=boot_gap,
        maximum_scalar_support_gap=support_gap,
        all_bootstrap_means_checked=3600000,query_targets_accessed=True,core_research_goal_complete=False,
        prediction_summary_sha256=sha(predictions/'summary.json'),evaluation_summary_sha256=sha(evaluation/'summary.json'),
        source_sha256={Path(__file__).name:sha(Path(__file__))})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_fresh_pilot/audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

"""328 independent support generator, raw predictions, risks, costs and statistics."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

METRICS=['mse257','mse129','point_mse257','point_mse129']
CANDIDATES=['online_first_fit_dual','online_uniform_state_dual_plus_residual']


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12),(a,b)


def quantile(values,p):
    values=sorted(map(float,values));t=(len(values)-1)*p;i=math.floor(t)
    return values[i]+(t-i)*(values[min(i+1,len(values)-1)]-values[i])


def scalar_forward(x,biases):
    for b in biases:
        z=float(x)+float(b);x=max(0.,min(2*z,2-2*z))
    return x


def particle_forward(points,q):
    h=np.broadcast_to(q,(len(points),len(q))).copy()
    for j in range(4):
        h=h+points[:,j,None];h=np.maximum(0.,np.minimum(2*h,2-2*h))
    return h.mean(axis=0)


def run(root,out,stage):
    start=time.perf_counter();base=root/'results/online_credit_fresh_pilot';pred=base/f'{stage}_predictions_v2';ev=base/f'{stage}_evaluation_v2'
    ps,es=read(pred/'summary.json'),read(ev/'summary.json');assert ps['passed'] and es['passed']
    for folder,s in [(pred,ps),(ev,es)]:
        for n,d in s['outputs_sha256'].items():assert sha(folder/n)==d
    p,ep=read(pred/'protocol.json'),read(ev/'protocol.json');seeds=p['seeds'];names=p['methods'];n=len(seeds)
    assert seeds==(list(range(328000000,328000512)) if stage=='pilot' else [5910000,5910001,5910063])
    assert p['methods']==ep['methods'] and p['candidates']==ep['candidates']==CANDIDATES and p['seeds']==ep['seeds']
    assert ps['tasks']==es['tasks']==n and ps['predictors']==es['predictors']==51*n
    assert es['comparisons']==400 and es['main_family']==100
    assert ep['prediction_summary_sha256']==sha(pred/'summary.json') and ep['before_query_manifest_sha256']==sha(pred/'before_query_manifest.json')
    assert p['design_sha256']==ep['design_sha256']==sha(root/'outputs/ttt-pc-alm-research/328_fresh_online_credit_protocol_v1.md')
    assert p['serialization_revision_sha256']==ep['serialization_revision_sha256']==sha(root/'outputs/ttt-pc-alm-research/328_serialization_correction_v2.md')
    for fn,d in p['source_sha256'].items():assert sha(root/'work/experiments'/fn)==d
    assert ep['source_sha256']==p['source_sha256'];reps=20000 if stage=='pilot' else 103
    assert ep['bootstrap']==dict(repetitions=reps,seed=328193,block=64,adjusted_quantile=1-.05/100,quantile_method='linear',resampling_unit='task')
    rows=read(pred/'rows.json');files=read(pred/'files.json');seal=read(pred/'before_query_manifest.json')
    assert not p['query_targets_accessed'] and not seal['query_targets_accessed'];ph=sha(pred/'protocol.json')
    for key,fn in [('rows_sha256','rows.json'),('files_sha256','files.json'),('costs_sha256','costs.json')]:assert seal[key]==sha(pred/fn)
    assert seal['protocol_sha256']==ph
    index={(r['seed'],r['method']):r for r in rows};assert len(index)==len(rows)==len(files)==51*n
    with np.load(ev/'task_metrics.npz',allow_pickle=False) as z:
        risk,times,failed=z['risk'],z['seconds'],z['failures']
        assert z['seeds'].tolist()==seeds and z['methods'].tolist()==names and z['metric_names'].tolist()==METRICS
    assert risk.shape==(n,51,4) and times.shape==failed.shape==(n,51)
    assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1)) and np.all(times>0)
    assert int(failed.sum())==ps['failures']==es['numerical_failures']
    counts=Counter();gaps=Counter();groups={label:[] for label in ['support_fit','fallback','execution_failed']}
    for si,seed in enumerate(seeds):
        commit=read(pred/str(seed)/'commit.json');assert commit['seed']==seed and commit['protocol_sha256']==ph and not commit['query_targets_accessed']
        expected=[names[int(i)] for i in np.random.default_rng(np.random.SeedSequence([328929,seed])).permutation(51)]
        assert [r['method'] for r in commit['rows']]==expected
        assert commit['failures']==sum(r['execution_failed'] for r in commit['rows'])
        assert commit['charged_seconds']==math.fsum(r['seconds'] for r in commit['rows'])
        rng=np.random.default_rng(seed);latent=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)[:4]
        noise=np.random.default_rng(seed+19000000).uniform(-.001,.001,24)[:4]
        support=np.array([scalar_forward(t,latent) for t in x])+noise;truth=[scalar_forward(i/256,latent) for i in range(257)]
        for mi,name in enumerate(names):
            r=index[seed,name];assert r==commit['rows'][r['order']]
            assert sha(pred/r['file'])==r['sha256']==files[r['file']]
            assert sha(pred/r['metadata_file'])==r['metadata_sha256'] and sha(pred/r['start_file'])==r['start_sha256']
            log=read(pred/r['metadata_file']);m=log['metadata']
            for note in [log,read(pred/r['start_file'])]:assert note['seed']==seed and note['method']==name and note['order']==r['order'] and note['protocol_sha256']==ph
            assert r['seconds']==m['charged_complete_seconds']==times[si,mi] and r['execution_failed']==m['execution_failed']==bool(failed[si,mi])
            assert not m['query_targets_accessed']
            if name==CANDIDATES[0]:
                label='execution_failed' if m['execution_failed'] else 'support_fit' if m['selected_state']['support_fit'] else 'fallback'
                groups[label].append(si)
            with np.load(pred/r['file'],allow_pickle=False) as z:
                observed=hashlib.sha256()
                for field in ['x_observed','v_observed','q_observed']:observed.update(z[field].tobytes())
                assert observed.hexdigest()==commit['observed_sha256'] and np.array_equal(z['x_observed'],x)
                gaps['support_generator']=max(gaps['support_generator'],float(np.max(abs(z['v_observed']-support))));assert gaps['support_generator']<2e-12
                assert np.array_equal(z['q_observed'],np.linspace(0,1,257)) and not np.isin(z['q_observed'],x).any()
                for field_index,field in enumerate(['prediction','point_prediction']):
                    a=z[field];assert a.shape==(257,) and np.all((a>=0)&(a<=1))
                    for stride in [1,2]:
                        value=math.fsum((float(a[i])-truth[i])**2 for i in range(0,257,stride))/len(range(0,257,stride))
                        gaps['scalar_risk']=max(gaps['scalar_risk'],abs(value-risk[si,mi,2*field_index+stride-1]));assert gaps['scalar_risk']<2e-12;counts['scalar_risks']+=1
                if name.startswith('online_') and not m['execution_failed']:
                    recomputed=particle_forward(z['points'],z['q_observed'])
                    gaps['particle_readout']=max(gaps['particle_readout'],float(np.max(abs(recomputed-z['prediction']))));assert gaps['particle_readout']<2e-12
                    assert not m['trace_enabled'] and not m['uses_complete_posterior_reference']
                    if not name.endswith('_bp'):assert m['no_global_bp_guard_enabled'];counts['no_bp_candidate_checks']+=1
                    counts['independent_particle_predictors']+=1
            counts['support_generators']+=1;counts['predictors']+=1
        if (si+1)%32==0:print(dict(audited_tasks=si+1,total=n,seconds=time.perf_counter()-start),flush=True)
    costs=read(pred/'costs.json');assert costs==read(ev/'actual_costs.json');cby={c['method']:c for c in costs['methods']}
    means={name:statistics.fmean(times[:,j]) for j,name in enumerate(names)}
    for j,name in enumerate(names):
        c=cby[name];close(c['mean_seconds'],means[name]);close(c['median_seconds'],statistics.median(times[:,j]));close(c['p90_seconds'],quantile(times[:,j],.9));close(c['maximum_seconds'],max(times[:,j]))
        assert c['failures']==int(failed[:,j].sum())
        for candidate in CANDIDATES:close(c['ratios_to_candidates'][candidate],means[name]/means[candidate])
        counts['independent_cost_methods']+=1
    for candidate in CANDIDATES:
        c=costs['candidates'][candidate];close(c['mean_seconds'],means[candidate])
        assert c['within_budget']==[name for name in names if means[name]<=means[candidate]]
        assert c['sensitivity_110_percent']==[name for name in names if means[name]<=means[candidate]*1.1]
    assert costs['all_methods_retained'] and not costs['query_quality_used']
    table=read(ev/'methods.json');assert [r['method'] for r in table]==names
    for j,row in enumerate(table):
        for k,metric in enumerate(METRICS):close(row['metrics'][metric],statistics.fmean(risk[:,j,k]))
        close(row['mean_seconds'],means[row['method']]);assert row['maximum_named_byte_fields']==cby[row['method']]['maximum_named_byte_fields']
    pairs=[(c,b) for c in CANDIDATES for b in names if c!=b]
    differences=np.stack([risk[:,names.index(c)]-risk[:,names.index(b)] for c,b in pairs],axis=1)
    with np.load(ev/'bootstrap_means.npz',allow_pickle=False) as z:boot=z['means']
    assert boot.shape==(reps,100,4)
    rng=np.random.default_rng(328193);digest=hashlib.sha256()
    for first in range(0,reps,13):
        size=min(13,reps-first);ids=rng.integers(n,size=(size,n),dtype=np.int64);digest.update(ids.tobytes())
        calculated=differences[ids].mean(axis=1);gaps['bootstrap']=max(gaps['bootstrap'],float(np.max(abs(calculated-boot[first:first+size]))));assert gaps['bootstrap']<2e-12
    assert digest.hexdigest()==es['bootstrap_index_sha256']
    comparisons=read(ev/'comparisons.json');assert len(comparisons)==400
    for j,(candidate,control) in enumerate(pairs):
        for k,metric in enumerate(METRICS):
            row=comparisons[4*j+k];d=differences[:,j,k];assert row['candidate']==candidate and row['control']==control and row['metric']==metric
            close(row['mean_difference'],statistics.fmean(d));close(row['sample_sd'],statistics.stdev(d))
            for ii,percent in enumerate([.025,.975]):close(row['descriptive95'][ii],quantile(boot[:,j,k],percent))
            if k==0:
                upper=quantile(boot[:,j,k],1-.05/100);close(row['bonferroni_upper'],upper);assert row['adjusted_negative']==(upper<0)
            else:assert row['bonferroni_upper'] is None and row['adjusted_negative'] is None
            absolute=[abs(float(v)) for v in d];largest=max(range(n),key=lambda i:absolute[i]);total=math.fsum(absolute)
            assert row['largest_absolute_task_seed']==seeds[largest]
            close(row['largest_absolute_task_delta'],d[largest]);close(row['largest_absolute_share'],absolute[largest]/total if total else 0.)
            leave=[math.fsum(float(v) for t,v in enumerate(d) if t!=i)/(n-1) for i in range(n)]
            close(row['leave_largest_absolute_out_mean'],leave[largest]);close(row['worst_leave_one_out_mean'],max(leave))
            assert [row['improved'],row['equal'],row['worse']]==[int(sum(d<0)),int(sum(d==0)),int(sum(d>0))];counts['comparisons']+=1
    assert es['all_bootstrap_means_checked']==reps*400 and es['adjusted_negative_main_comparisons']==sum(r['adjusted_negative'] for r in comparisons if r['metric']=='mse257')
    grouped=read(ev/'mechanism_groups.json');assert not grouped['query_quality_used_for_grouping']
    assert len(grouped['groups'])==3 and sum(g['tasks'] for g in grouped['groups'])==n
    for group in grouped['groups']:
        indices=groups[group['group']];assert group['tasks']==len(indices) and group['seeds']==[seeds[i] for i in indices]
        for j,item in enumerate(group['metrics']):
            assert item['method']==names[j]
            for k,metric in enumerate(METRICS):
                if indices:close(item['means'][metric],statistics.fmean(risk[i,j,k] for i in indices))
                else:assert item['means'][metric] is None
                counts['group_metric_fields']+=1
        for j,(candidate,control) in enumerate(pairs):
            for k,metric in enumerate(METRICS):
                row=group['comparisons'][4*j+k];assert row['candidate']==candidate and row['control']==control and row['metric']==metric
                values=[float(differences[i,j,k]) for i in indices]
                if indices:close(row['mean_difference'],statistics.fmean(values))
                else:assert row['mean_difference'] is None
                assert [row['improved'],row['equal'],row['worse']]==[sum(v<0 for v in values),sum(v==0 for v in values),sum(v>0 for v in values)]
                counts['group_comparisons']+=1
    result=dict(passed=True,stage=stage,counts=dict(counts),maximum_gaps=dict(gaps),all_bootstrap_means_checked=reps*400,
        seconds=time.perf_counter()-start,query_targets_accessed=True,core_research_goal_complete=False,
        prediction_summary_sha256=sha(pred/'summary.json'),evaluation_summary_sha256=sha(ev/'summary.json'),source_sha256={Path(__file__).name:sha(Path(__file__))})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','pilot'],required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2]
    out=root/'results/online_credit_fresh_pilot'/f'{a.stage}_audit_v2';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,a.stage)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

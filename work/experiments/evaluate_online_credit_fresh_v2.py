"""328 evaluation only after the complete durable prediction seal."""
import argparse
import math
from pathlib import Path
import time
import traceback
import numpy as np
import online_credit_fresh_suite_v2 as suite
import run_online_credit_fresh_v2 as pipeline
import probe_confirmation_statistics as statistics
from counterfactual_fresh_pilot_v1 import scalar_truth
from analyze_recovered_online_comparison import forward
from evaluate_complete_credit_mode_geometry_v1 import read,sha

METRICS=['mse257','mse129','point_mse257','point_mse129']


def paired_comparisons(risk,names):
    pairs=[(candidate,control) for candidate in suite.CANDIDATES for control in names if control!=candidate]
    assert len(pairs)==100
    delta=np.stack([risk[:,names.index(c)]-risk[:,names.index(b)] for c,b in pairs],axis=1)
    return pairs,delta


def concentration(delta,seeds):
    d=np.asarray(delta);i=int(np.argmax(abs(d)));total=math.fsum(map(float,abs(d)));mean=math.fsum(map(float,d))/len(d)
    leave=[(len(d)*mean-float(v))/(len(d)-1) for v in d]
    return dict(largest_absolute_task_seed=int(seeds[i]),largest_absolute_task_delta=float(d[i]),
        largest_absolute_share=float(abs(d[i])/total) if total else 0.,
        leave_largest_absolute_out_mean=leave[i],worst_leave_one_out_mean=max(leave),
        improved=int(np.sum(d<0)),equal=int(np.sum(d==0)),worse=int(np.sum(d>0)))


def run(root,out,stage):
    hashes=suite.gate(root);folder=out.parent/f'{stage}_predictions_v2';summary=suite.resources.old.complete(folder)
    p=read(folder/'protocol.json');assert p['source_sha256']==hashes and p['design_sha256']==sha(root/suite.DESIGN)
    assert p['serialization_revision_sha256']==sha(root/suite.REVISION)
    assert p['seeds']==(suite.OLD_SEEDS if stage=='preflight' else suite.PILOT_SEEDS)
    rows,files,checks=pipeline.verify(root,folder,p)
    assert rows==read(folder/'rows.json') and files==read(folder/'files.json') and checks==summary['checks']
    seal=read(folder/'before_query_manifest.json');assert not seal['query_targets_accessed'] and seal['protocol_sha256']==sha(folder/'protocol.json')
    for k,n in [('rows_sha256','rows.json'),('files_sha256','files.json'),('costs_sha256','costs.json')]:assert sha(folder/n)==seal[k]
    costs=read(folder/'costs.json');assert costs==pipeline.cost_table(rows,p['methods'])
    names,seeds=p['methods'],p['seeds'];n=len(seeds);reps=103 if stage=='preflight' else 20000
    protocol=dict(stage=stage,source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),serialization_revision_sha256=sha(root/suite.REVISION),methods=names,seeds=seeds,
        candidates=suite.CANDIDATES,metrics=METRICS,primary_metric='mse257',main_comparison_family_size=100,
        bootstrap=dict(repetitions=reps,seed=328193,block=64,adjusted_quantile=1-.05/100,quantile_method='linear',resampling_unit='task'),
        prediction_summary_sha256=sha(folder/'summary.json'),before_query_manifest_sha256=sha(folder/'before_query_manifest.json'),
        frozen_resource_selection=p['frozen_resource_selection'],query_targets_accessed=True,
        inference='Approximate paired percentile bootstrap; preflight statistics are code checks only, not evidence of superiority',
        core_research_goal_complete=False)
    pipeline.exclusive(out/'protocol.json',protocol);begin=time.perf_counter()
    risk=np.empty((n,51,4));times=np.empty((n,51));failed=np.zeros((n,51),bool);lookup={(r['seed'],r['method']):r for r in rows}
    truth_gap=scalar_gap=0.
    for i,seed in enumerate(seeds):
        q=np.linspace(0,1,257);b=np.random.default_rng(seed).uniform(-.12,.12,4)
        truth=forward(q,b[None])[0];independent=scalar_truth(seed,q)
        truth_gap=max(truth_gap,float(np.max(abs(truth-independent))));assert truth_gap<2e-12
        for j,name in enumerate(names):
            row=lookup[seed,name]
            with np.load(folder/row['file'],allow_pickle=False) as a:
                values=[];separate=[]
                for field in suite.FIELDS:
                    for stride in [1,2]:
                        prediction=a[field][::stride];target=truth[::stride]
                        values.append(float(np.mean((prediction-target)**2)))
                        separate.append(math.fsum((float(v)-float(t))**2 for v,t in zip(prediction,independent[::stride]))/len(prediction))
                risk[i,j]=values;scalar_gap=max(scalar_gap,float(np.max(abs(np.array(values)-separate))));assert scalar_gap<2e-12
            times[i,j]=row['seconds'];failed[i,j]=row['execution_failed']
    assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1));pairs,delta=paired_comparisons(risk,names)
    flat=delta.reshape(n,400);statistics.selftest();boot,digest=statistics.bootstrap_means(flat,reps,328193,64)
    rng=np.random.default_rng(328193);boot_gap=0.
    for first in range(0,reps,7):
        size=min(7,reps-first);indices=rng.integers(n,size=(size,n),dtype=np.int64)
        independent=flat[indices].mean(axis=1);boot_gap=max(boot_gap,float(np.max(abs(independent-boot[first:first+size]))));assert boot_gap<2e-12
    boot=boot.reshape(reps,100,4)
    with (out/'task_metrics.npz').open('xb') as f:np.savez_compressed(f,seeds=seeds,methods=names,metric_names=METRICS,risk=risk,seconds=times,failures=failed)
    with (out/'bootstrap_means.npz').open('xb') as f:np.savez_compressed(f,means=boot)
    comparisons=[]
    for j,(candidate,control) in enumerate(pairs):
        for k,metric in enumerate(METRICS):
            values=delta[:,j,k];bounds=np.quantile(boot[:,j,k],[.025,.975]);upper=float(np.quantile(boot[:,j,k],1-.05/100)) if k==0 else None
            comparisons.append(dict(candidate=candidate,control=control,metric=metric,mean_difference=math.fsum(map(float,values))/n,
                sample_sd=float(values.std(ddof=1)),descriptive95=bounds.tolist(),bonferroni_upper=upper,
                adjusted_negative=upper<0 if upper is not None else None,**concentration(values,seeds)))
    byname={m['method']:m for m in costs['methods']};methods=[]
    for j,name in enumerate(names):
        c=byname[name];assert abs(float(times[:,j].mean())-c['mean_seconds'])<1e-12
        methods.append(dict(method=name,metrics={metric:math.fsum(map(float,risk[:,j,k]))/n for k,metric in enumerate(METRICS)},
            mean_seconds=c['mean_seconds'],median_seconds=c['median_seconds'],p90_seconds=c['p90_seconds'],maximum_seconds=c['maximum_seconds'],
            ratios_to_candidates=c['ratios_to_candidates'],failures=int(failed[:,j].sum()),maximum_named_byte_fields=c['maximum_named_byte_fields']))
    for fn,value in [('methods.json',methods),('comparisons.json',comparisons),('actual_costs.json',costs)]:pipeline.exclusive(out/fn,value)
    groups={name:[] for name in ['support_fit','fallback','execution_failed']}
    for i,seed in enumerate(seeds):
        row=lookup[seed,suite.CANDIDATES[0]];meta=read(folder/row['metadata_file'])['metadata']
        label='execution_failed' if meta['execution_failed'] else 'support_fit' if meta['selected_state']['support_fit'] else 'fallback'
        groups[label].append(i)
    grouped=[]
    for label,indices in groups.items():
        grouped.append(dict(group=label,seeds=[seeds[i] for i in indices],tasks=len(indices),
            metrics=[dict(method=name,means={metric:math.fsum(float(risk[i,j,k]) for i in indices)/len(indices) if indices else None for k,metric in enumerate(METRICS)}) for j,name in enumerate(names)],
            comparisons=[dict(candidate=c,control=b,metric=metric,mean_difference=math.fsum(float(delta[i,j,k]) for i in indices)/len(indices) if indices else None,
                improved=int(sum(delta[i,j,k]<0 for i in indices)),
                equal=int(sum(delta[i,j,k]==0 for i in indices)),
                worse=int(sum(delta[i,j,k]>0 for i in indices)))
                for j,(c,b) in enumerate(pairs) for k,metric in enumerate(METRICS)]))
    pipeline.exclusive(out/'mechanism_groups.json',dict(groups=grouped,query_quality_used_for_grouping=False,inference='Descriptive mechanism strata only; no subgroup significance or replacement of all-task result'))
    assert suite.gate(root)==hashes
    result=dict(passed=True,tasks=n,predictors=n*51,comparisons=400,main_family=100,
        adjusted_negative_main_comparisons=sum(r['adjusted_negative'] for r in comparisons if r['metric']=='mse257'),
        independent_scalar_risk_passed=True,maximum_truth_gap=truth_gap,maximum_scalar_risk_gap=scalar_gap,
        all_bootstrap_means_checked=reps*400,maximum_bootstrap_gap=boot_gap,bootstrap_index_sha256=digest,
        numerical_failures=int(failed.sum()),seconds=time.perf_counter()-begin,query_targets_accessed=True,
        stage=stage,core_research_goal_complete=False,
        outputs_sha256={fn:sha(out/fn) for fn in ['protocol.json','task_metrics.npz','bootstrap_means.npz','methods.json','comparisons.json','actual_costs.json','mechanism_groups.json']})
    pipeline.exclusive(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','pilot'],required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2]
    out=root/suite.BASE/f'{a.stage}_evaluation_v2';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,a.stage)
    except Exception:pipeline.exclusive(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

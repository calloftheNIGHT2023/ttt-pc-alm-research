"""287 frozen evaluation, only after all predictions and both audit gates.

Defaults to old-task functional preflight. No early fresh-query inspection.
All25 planned comparisons retained; no subgroup/metric replacement rule.
"""
import argparse,json,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import probe_confirmation_statistics as statistics
from analyze_recovered_online_comparison import forward
from run_probe_credit_confirmation_v2 import verify_commit,exclusive_json,acquire_lock
from run_multiplier_fixed_point_screen import sha,dump


def load_gate(folder):
    summary=json.loads((folder/'summary.json').read_text());assert summary['passed']
    for name,h in summary.get('outputs_sha256',{}).items():assert sha(folder/name)==h,(folder,name)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight');args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';assert not (base/'predictions/RUNNING.lock').exists(),'Do not evaluate during prediction timing'
    inp=base/('runner_preflight_v2' if args.stage=='preflight' else 'predictions')
    audit=base/('prediction_audit_preflight_v2' if args.stage=='preflight' else 'prediction_audit_v2')
    pathaudit=base/('path_audit_preflight_v4' if args.stage=='preflight' else 'path_audit_v4');out=base/('evaluation_preflight_v4' if args.stage=='preflight' else 'evaluation_v4')
    run=load_gate(inp);apass=load_gate(audit);ppass=load_gate(pathaudit);p=json.loads((inp/'protocol.json').read_text());names=p['methods'];seeds=p['seeds']
    n=2 if args.stage=='preflight' else 8192;assert len(seeds)==n and run['predictors']==apass['counts']['predictors']==n*27
    assert not run['query_targets_accessed'] and not apass['query_targets_accessed'] and not ppass['query_targets_accessed']
    assert ppass['counts']['predictors']==min(n,64)*27
    if args.stage=='confirmation':assert ppass['may_evaluate_confirmation'] and seeds==list(range(5500000,5508192))
    assert json.loads((pathaudit/'protocol.json').read_text())['prediction_summary_sha256']==sha(inp/'summary.json')
    assert json.loads((pathaudit/'protocol.json').read_text())['all_prediction_audit_sha256']==sha(audit/'summary.json')
    assert json.loads((audit/'protocol.json').read_text())['prediction_summary_sha256']==sha(inp/'summary.json')
    hashes=dict(json.loads((pathaudit/'protocol.json').read_text())['source_sha256'])
    for name in [Path(__file__).name,'probe_confirmation_statistics.py']:hashes[name]=sha(src/name)
    for name,h in hashes.items():assert sha(src/name)==h,name
    if args.stage=='confirmation':
        old=base/'evaluation_preflight_v4';gate=load_gate(old)
        assert gate['tasks']==2 and gate['functional_preflight_only'] and json.loads((old/'protocol.json').read_text())['source_sha256']==hashes
        old_audit=load_gate(base/'evaluation_audit_preflight_v4');assert old_audit['evaluation_summary_sha256']==sha(old/'summary.json') and old_audit['predictors']==54
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    check=statistics.selftest();primary=p['primary'];pi=names.index(primary);controls=[name for name in names if name!=primary];family=[name for name in controls if name!='probe_all_alm64']
    assert len(names)==27 and len(controls)==26 and len(family)==25
    calibration=root/'results/probe_credit_resources/calibration';heads=base/'head_preflight'
    for directory in [calibration,heads]:
        ss=json.loads((directory/'summary.json').read_text());assert ss['passed']
        for rel,h in ss.get('outputs_sha256',{}).items():assert sha(directory/rel)==h
    old_resources={r['method']:r for d in [calibration,heads] for r in json.loads((d/'methods.json').read_text())}
    old_cap=old_resources[primary]['mean_seconds'];assert set(names)<=set(old_resources)
    # Saved audit records form a complete gate, not only a boolean summary.
    for directory in [audit,pathaudit]:
        for rel,h in json.loads((directory/'files.json').read_text()).items():
            assert sha(directory/rel)==h
            record=json.loads((directory/rel).read_text())
            for fname,hh in record.get('files',{}).items():assert sha(directory/fname)==hh
    before=json.loads((inp/'before_query_manifest.json').read_text());assert len(before['task_commits'])==n
    for seed,commit in zip(seeds,before['task_commits']):assert verify_commit(inp,seed,names,sha(inp/'protocol.json'))==commit
    # The first query teacher access occurs strictly after this full census.
    metric_names=['mse257','mse129','point_mse257','point_mse129'];repetitions=100000
    protocol=dict(stage=args.stage,source_sha256=hashes,prediction_summary_sha256=sha(inp/'summary.json'),prediction_audit_sha256=sha(audit/'summary.json'),path_audit_sha256=sha(pathaudit/'summary.json'),
        seeds=seeds,methods=names,primary=primary,controls=controls,primary_family=family,metrics=metric_names,primary_metric='mse257',
        bootstrap=dict(repetitions=repetitions,seed=287193,block=64,indices_dtype='int64',shared_resamples_for_all_methods_metrics=True,quantile_method='linear',
            descriptive_quantiles=[.025,.975],primary_adjusted_upper_quantile=1-.05/25),selftest=check,
        old_calibration_methods_sha256=sha(calibration/'methods.json'),old_extra_head_methods_sha256=sha(heads/'methods.json'),
        query_targets_accessed=True,functional_preflight_only=args.stage=='preflight',inference='Approximate paired percentile bootstrap, task unit; not finite-sample distribution-free guarantee',
        full_family_success_rule='All25 fixed main257 one-sided Bonferroni upper bounds <0; scientific generalization and resource fairness still separate')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists(),'Preserve existing evaluator evidence; inspect before any retry'
    lock,identity=acquire_lock(out)
    try:
        exclusive_json(out/'protocol.json',protocol);begin=time.perf_counter();risk=np.empty((n,27,4));times=np.empty((n,27));failures=np.zeros((n,27),dtype=bool)
        state_subtotals=defaultdict(list);extra_known=0.;extra_calls=0;extra_unknown=0;task_wall=0.
        for si,seed in enumerate(seeds):
            commit=json.loads((inp/'tasks'/str(seed)/'commit.json').read_text());rows={r['method']:r for r in json.loads((inp/commit['rows_file']).read_text())}
            teacher=np.random.default_rng(seed).uniform(-.12,.12,4);q=np.linspace(0,1,257);truth=forward(q,teacher[None])[0]
            for mi,name in enumerate(names):
                row=rows[name];assert sha(inp/row['file'])==row['sha256']
                with np.load(inp/row['file']) as a:
                    error=np.square(a['prediction']-truth);point_error=np.square(a['point_prediction']-truth)
                    risk[si,mi]=[error.mean(),error[::2].mean(),point_error.mean(),point_error[::2].mean()]
                meta=row['metadata'];times[si,mi]=meta['charged_complete_seconds'];failures[si,mi]=meta['execution_failed']
                for prefix,values in [('',meta),('metadata.',meta.get('metadata',{}))]:
                    for key,value in values.items():
                        if 'bytes' in key and isinstance(value,(int,float)) and not isinstance(value,bool):state_subtotals[name,prefix+key].append(value)
            extra=commit['interrupted_attempt_costs'];extra_known+=extra['known_completed_call_seconds'];extra_calls+=extra['completed_calls'];extra_unknown+=extra['unresolved_started_calls']
            task_wall+=commit['task_wall_including_io_seconds']
            if (si+1)%512==0:print(json.dumps(dict(evaluated_tasks=si+1,total=n,seconds=time.perf_counter()-begin)),flush=True)
        assert np.isfinite(risk).all() and np.all(times>0) and int(failures.sum())==run['numerical_failures']
        with (out/'task_metrics.npz').open('xb') as f:np.savez_compressed(f,seeds=np.array(seeds),methods=np.array(names),metric_names=np.array(metric_names),risk=risk,seconds=times,failures=failures)
        ci=[names.index(name) for name in controls];differences=risk[:,pi:pi+1,:]-risk[:,ci,:]
        boot,digest=statistics.bootstrap_means(differences.reshape(n,-1),repetitions,287193,64);boot=boot.reshape(repetitions,26,4)
        with (out/'bootstrap_means.npz').open('xb') as f:np.savez_compressed(f,controls=np.array(controls),metrics=np.array(metric_names),means=boot)
        comparisons=[]
        for j,name in enumerate(controls):
            for k,metric in enumerate(metric_names):
                values=differences[:,j,k];bounds=np.quantile(boot[:,j,k],[.025,.975],method='linear');upper=float(np.quantile(boot[:,j,k],1-.05/25,method='linear')) if name in family and k==0 else None
                comparisons.append(dict(primary=primary,control=name,metric=metric,tasks=n,mean_difference=float(values.mean()),paired_sample_sd=float(values.std(ddof=1)),
                    descriptive95=bounds.tolist(),bonferroni_one_sided_upper=upper,predeclared_main_comparison=name in family and k==0,
                    adjusted_upper_below_zero=upper<0 if upper is not None else None,improved_tasks=int(np.sum(values<0)),equal_tasks=int(np.sum(values==0)),worse_tasks=int(np.sum(values>0))))
        cap=float(times[:,pi].mean());methods=[]
        for j,name in enumerate(names):
            mean=float(times[:,j].mean());methods.append(dict(method=name,metrics={metric:dict(mean=float(risk[:,j,k].mean()),task_sample_sd=float(risk[:,j,k].std(ddof=1))) for k,metric in enumerate(metric_names)},
                complete_time=dict(mean_seconds=mean,median_seconds=float(np.median(times[:,j])),p90_seconds=float(np.quantile(times[:,j],.9)),p99_seconds=float(np.quantile(times[:,j],.99)),maximum_seconds=float(times[:,j].max())),
                actual_mean_time_ratio_to_primary=mean/cap,actual_mean_resource_class='main_1.00' if mean<=cap else 'sensitivity_1.10' if mean<=1.1*cap else 'measured_over_1.10',
                old_calibration_mean_seconds=old_resources[name]['mean_seconds'],old_calibration_resource_class='main_1.00' if old_resources[name]['mean_seconds']<=old_cap else 'sensitivity_1.10' if old_resources[name]['mean_seconds']<=old_cap*1.1 else 'measured_over_budget',
                predeclared_primary_comparison=name in family,failures=int(failures[:,j].sum()),failure_rate=float(failures[:,j].mean()),
                named_numeric_state_subtotals={key:dict(observed_rows=len(v),mean_bytes=float(np.mean(v)),maximum_bytes=max(v)) for (method,key),v in state_subtotals.items() if method==name}))
        for name,value in [('comparisons.json',comparisons),('methods.json',methods)]:exclusive_json(out/name,value)
        invocations=[json.loads(path.read_text()) for path in sorted(inp.glob('invocation_*.json'))]
        for name,h in hashes.items():assert sha(src/name)==h,name
        primary_rows=[r for r in comparisons if r['predeclared_main_comparison']];assert len(primary_rows)==25
        passed=sum(r['adjusted_upper_below_zero'] for r in primary_rows)
        ans=dict(passed=True,stage=args.stage,tasks=n,predictors=n*27,functional_preflight_only=args.stage=='preflight',query_targets_accessed=True,
            comparisons=len(comparisons),main_family_size=25,main_adjusted_negative_upper_bounds=passed,all25_adjusted_negative_upper_bounds=passed==25,
            core_research_goal_complete=False,bootstrap_index_sha256=digest,bootstrap_selftest=check,numerical_failures=int(failures.sum()),
            resource_notes=dict(successful_and_fallback_calls_seconds=float(times.sum()),extra_interrupted_completed_calls=extra_calls,extra_known_interrupted_seconds=extra_known,
                completed_task_wall_including_io_seconds=task_wall,
                unresolved_started_calls=extra_unknown,model_loading_seconds_all_invocations=sum(i['model_loading_seconds'] for i in invocations),
                shared_checkpoint_manifest=p['checkpoint_manifest'],scope='Named numeric subtotals are not peak native memory; per-method call times exclude I/O/loading, task wall and loading shown separately; faster controls may require further fair budget enhancement'),
            seconds=time.perf_counter()-begin,outputs_sha256={name:sha(out/name) for name in ['protocol.json','task_metrics.npz','bootstrap_means.npz','comparisons.json','methods.json']},
            next='Independent full risk and bootstrap audit, then report every comparator and resource gap; do not substitute secondary metrics')
        exclusive_json(out/'summary.json',ans);print(json.dumps({k:v for k,v in ans.items() if k!='resource_notes'}),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())==identity:lock.unlink()


if __name__=='__main__':main()

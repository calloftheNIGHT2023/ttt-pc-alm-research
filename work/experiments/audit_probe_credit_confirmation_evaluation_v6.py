"""Independent287 risk/summary/full-bootstrap replay; old preflight by default."""
import argparse,hashlib,json,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def independent_truth(q,b):
    h=q.copy()
    for bias in b:
        z=h+bias;h=np.zeros_like(z)
        left=(z>=0)&(z<.5);right=(z>=.5)&(z<1)
        h[left]=2*z[left];h[right]=2-2*z[right]
    return h


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight');args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';assert not (base/'predictions/RUNNING.lock').exists()
    inp=base/('evaluation_preflight_v6' if args.stage=='preflight' else 'evaluation_v6');pred=base/('runner_preflight_v2' if args.stage=='preflight' else 'predictions')
    out=base/('evaluation_audit_preflight_v6' if args.stage=='preflight' else 'evaluation_audit_v6');out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    s=json.loads((inp/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text());assert s['passed'] and s['query_targets_accessed']
    for name,h in s['outputs_sha256'].items():assert sha(inp/name)==h
    hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for name,h in hashes.items():assert sha(src/name)==h,name
    if args.stage=='confirmation':
        old=base/'evaluation_audit_preflight_v6';old_summary=json.loads((old/'summary.json').read_text());assert old_summary['passed']
        assert json.loads((old/'protocol.json').read_text())['source_sha256']==hashes
        for name,h in old_summary['outputs_sha256'].items():assert sha(old/name)==h
    exclusive_json(out/'protocol.json',dict(stage=args.stage,source_sha256=hashes,evaluation_summary_sha256=sha(inp/'summary.json'),
        scope='Independent piecewise teacher, all task losses/times/failures, every bootstrap replicate with different block layout, every interval and decision',
        risk_tolerance=dict(atol=1e-12,rtol=1e-11),bootstrap_tolerance=dict(atol=1e-12,rtol=1e-11)))
    begin=time.perf_counter();names=p['methods'];seeds=p['seeds'];metrics=p['metrics'];n=len(seeds)
    assert n==(2 if args.stage=='preflight' else 8192) and p['bootstrap']['repetitions']==100000
    with np.load(inp/'task_metrics.npz') as z:
        risk=z['risk'].copy();times=z['seconds'].copy();failures=z['failures'].copy()
        assert z['seeds'].tolist()==seeds and z['methods'].tolist()==names and z['metric_names'].tolist()==metrics
    assert risk.shape==(n,27,4) and times.shape==failures.shape==(n,27)
    maxrisk=0.;calls=0;taskwall=0.;extra_calls=0;extra_sec=0.;extra_unknown=0;state_values=defaultdict(list)
    calibration=root/'results/probe_credit_resources/calibration/methods.json';heads=base/'head_preflight/methods.json'
    assert sha(calibration)==p['old_calibration_methods_sha256'] and sha(heads)==p['old_extra_head_methods_sha256']
    old_resources={r['method']:r for path in [calibration,heads] for r in json.loads(path.read_text())}
    q=np.linspace(0,1,257)
    for i,seed in enumerate(seeds):
        commit=json.loads((pred/'tasks'/str(seed)/'commit.json').read_text());rows={r['method']:r for r in json.loads((pred/commit['rows_file']).read_text())}
        for rel,h in commit['files'].items():assert sha(pred/rel)==h
        truth=independent_truth(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for j,name in enumerate(names):
            row=rows[name]
            with np.load(pred/row['file']) as z:
                delta=z['prediction']-truth;point=z['point_prediction']-truth
                expected=np.array([np.dot(delta,delta)/257,np.dot(delta[::2],delta[::2])/129,np.dot(point,point)/257,np.dot(point[::2],point[::2])/129])
            np.testing.assert_allclose(expected,risk[i,j],rtol=1e-11,atol=1e-12);maxrisk=max(maxrisk,float(np.max(abs(expected-risk[i,j]))))
            assert times[i,j]==row['metadata']['charged_complete_seconds'] and failures[i,j]==row['metadata']['execution_failed'];calls+=1
            for prefix,values in [('',row['metadata']),('metadata.',row['metadata'].get('metadata',{}))]:
                for key,value in values.items():
                    if 'bytes' in key and isinstance(value,(int,float)) and not isinstance(value,bool):state_values[name,prefix+key].append(value)
        extra=commit['interrupted_attempt_costs'];extra_calls+=extra['completed_calls'];extra_sec+=extra['known_completed_call_seconds'];extra_unknown+=extra['unresolved_started_calls'];taskwall+=commit['task_wall_including_io_seconds']
        if (i+1)%512==0:print(json.dumps(dict(risk_tasks=i+1,total=n,seconds=time.perf_counter()-begin)),flush=True)
    primary=names.index(p['primary']);controls=p['controls'];indices=[names.index(name) for name in controls]
    d=(risk[:,primary:primary+1]-risk[:,indices]).reshape(n,-1)
    with np.load(inp/'bootstrap_means.npz') as z:
        boot=z['means'].copy();assert z['controls'].tolist()==controls and z['metrics'].tolist()==metrics
    assert boot.shape==(100000,26,4);rng=np.random.default_rng(287193);digest=hashlib.sha256();maxboot=0.
    # Different block size, same integer draw stream, independently expanded counts.
    flat=boot.reshape(100000,-1)
    for first in range(0,100000,37):
        count=min(37,100000-first);draw=rng.integers(n,size=(count,n),dtype=np.int64);digest.update(draw.tobytes())
        weights=np.zeros((count,n),dtype=np.float64)
        for row in range(count):np.add.at(weights[row],draw[row],1./n)
        expected=weights@d;np.testing.assert_allclose(expected,flat[first:first+count],rtol=1e-11,atol=1e-12)
        maxboot=max(maxboot,float(np.max(abs(expected-flat[first:first+count]))))
        if first==0:
            np.testing.assert_allclose(d[draw].mean(1),expected,rtol=1e-11,atol=1e-12)
    assert digest.hexdigest()==s['bootstrap_index_sha256']
    comparisons=json.loads((inp/'comparisons.json').read_text());assert len(comparisons)==104
    lookup={(r['control'],r['metric']):r for r in comparisons};assert len(lookup)==104
    d=d.reshape(n,26,4);passed=0
    for j,name in enumerate(controls):
        for k,metric in enumerate(metrics):
            r=lookup[name,metric];dd=d[:,j,k];ci=np.quantile(boot[:,j,k],[.025,.975],method='linear')
            assert r['tasks']==n and r['mean_difference']==float(dd.mean()) and r['paired_sample_sd']==float(dd.std(ddof=1))
            assert r['descriptive95']==ci.tolist() and r['improved_tasks']==int(np.sum(dd<0)) and r['equal_tasks']==int(np.sum(dd==0)) and r['worse_tasks']==int(np.sum(dd>0))
            main=name!='probe_all_alm64' and k==0;assert r['predeclared_main_comparison']==main
            if main:
                upper=float(np.quantile(boot[:,j,k],.998,method='linear'));assert upper==r['bonferroni_one_sided_upper'] and r['adjusted_upper_below_zero']==(upper<0);passed+=upper<0
            else:assert r['bonferroni_one_sided_upper'] is None and r['adjusted_upper_below_zero'] is None
    methods=json.loads((inp/'methods.json').read_text());assert len(methods)==27
    for j,row in enumerate(methods):
        assert row['method']==names[j] and row['failures']==int(failures[:,j].sum()) and row['failure_rate']==float(failures[:,j].mean())
        for k,metric in enumerate(metrics):assert row['metrics'][metric]==dict(mean=float(risk[:,j,k].mean()),task_sample_sd=float(risk[:,j,k].std(ddof=1)))
        t=times[:,j];assert row['complete_time']==dict(mean_seconds=float(t.mean()),median_seconds=float(np.median(t)),p90_seconds=float(np.quantile(t,.9)),p99_seconds=float(np.quantile(t,.99)),maximum_seconds=float(t.max()))
        cap=float(times[:,primary].mean());label='main_1.00' if t.mean()<=cap else 'sensitivity_1.10' if t.mean()<=1.1*cap else 'measured_over_1.10'
        assert row['actual_mean_resource_class']==label and row['actual_mean_time_ratio_to_primary']==float(t.mean())/cap
        old=old_resources[names[j]]['mean_seconds'];old_cap=old_resources[p['primary']]['mean_seconds']
        old_label='main_1.00' if old<=old_cap else 'sensitivity_1.10' if old<=1.1*old_cap else 'measured_over_budget'
        assert row['old_calibration_mean_seconds']==old and row['old_calibration_resource_class']==old_label
        expected_state={key:dict(observed_rows=len(values),mean_bytes=float(np.mean(values)),maximum_bytes=max(values)) for (method,key),values in state_values.items() if method==names[j]}
        assert row['named_numeric_state_subtotals']==expected_state
    assert s['main_adjusted_negative_upper_bounds']==passed and s['all25_adjusted_negative_upper_bounds']==(passed==25) and not s['core_research_goal_complete']
    resources=s['resource_notes'];assert resources['successful_and_fallback_calls_seconds']==float(times.sum()) and resources['completed_task_wall_including_io_seconds']==taskwall
    assert resources['extra_interrupted_completed_calls']==extra_calls and resources['extra_known_interrupted_seconds']==extra_sec and resources['unresolved_started_calls']==extra_unknown
    invocations=[json.loads(path.read_text()) for path in sorted(pred.glob('invocation_*.json'))]
    assert resources['model_loading_seconds_all_invocations']==sum(row['model_loading_seconds'] for row in invocations)
    assert resources['shared_checkpoint_manifest']==json.loads((pred/'protocol.json').read_text())['checkpoint_manifest']
    for name,h in hashes.items():assert sha(src/name)==h,name
    result=dict(passed=True,stage=args.stage,tasks=n,predictors=calls,independent_task_losses=calls*4,bootstrap_replicates=100000,bootstrap_scalar_means=100000*104,
        comparisons=104,method_tables=27,named_state_subtotals=len(state_values),old_and_actual_resource_classes=54,maximum_independent_risk_gap=maxrisk,maximum_independent_bootstrap_gap=maxboot,main_adjusted_negative_upper_bounds=int(passed),
        core_research_goal_complete=False,seconds=time.perf_counter()-begin,evaluation_summary_sha256=sha(inp/'summary.json'),outputs_sha256={'protocol.json':sha(out/'protocol.json')})
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

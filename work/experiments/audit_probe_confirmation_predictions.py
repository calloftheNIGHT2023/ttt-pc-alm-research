"""287 all-predictor gate; defaults to OLD runner fixture, not fresh data.

Full confirmation cannot be opened until all221184 predictions are committed.
No query truth is reconstructed; only observed supports and model readouts.
"""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_independent_heads as heads
from run_probe_credit_confirmation_v2 import verify_commit,exclusive_json,acquire_lock
from audit_local_dual_jump_modes import modes
from analyze_recovered_online_comparison import forward
from posterior_confirmation_pipeline import discovery_box
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight');args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/('runner_preflight_v2' if args.stage=='preflight' else 'predictions')
    out=base/('prediction_audit_preflight' if args.stage=='preflight' else 'prediction_audit')
    assert not (base/'predictions/RUNNING.lock').exists(),'Wait for confirmation timing before even the old audit preflight'
    assert not (inp/'RUNNING.lock').exists(),'Do not audit while prediction timing is active'
    summary=json.loads((inp/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text())
    assert summary['passed'] and not summary['query_targets_accessed'] and summary['checkpoint_replay_verified']
    expected=2 if args.stage=='preflight' else 8192;assert summary['tasks']==expected and summary['predictors']==27*expected
    assert p['seeds']==([5910000,5910063] if args.stage=='preflight' else list(range(5500000,5508192)))
    for n,h in summary['outputs_sha256'].items():assert sha(inp/n)==h,n
    before=json.loads((inp/'before_query_manifest.json').read_text());assert before['tasks']==expected and before['predictors']==27*expected
    assert [c['seed'] for c in before['task_commits']]==p['seeds'] and before['protocol_sha256']==sha(inp/'protocol.json')
    hashes=dict(p['source_sha256'])
    for name in [Path(__file__).name,'probe_confirmation_independent_heads.py']:hashes[name]=sha(src/name)
    for n,h in hashes.items():assert sha(src/n)==h,n
    if args.stage=='confirmation':
        old=base/'prediction_audit_preflight';gate=json.loads((old/'summary.json').read_text())
        assert gate['passed'] and gate['counts']['predictors']==54
        assert json.loads((old/'protocol.json').read_text())['source_sha256']==hashes
        for n,h in gate['outputs_sha256'].items():assert sha(old/n)==h,n
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    loaded,model_manifest=suite.resources.legacy.oldfit.meta.load(root);assert model_manifest==p['checkpoint_manifest']
    cfgs={c['name']:c for c in suite.catalogue(root)};assert list(cfgs)==p['methods']
    audit_protocol=dict(stage=args.stage,source_sha256=hashes,prediction_summary_sha256=sha(inp/'summary.json'),before_query_manifest_sha256=sha(inp/'before_query_manifest.json'),
        seeds=p['seeds'],query_targets_accessed=False,scope='Every committed input/file/state, independent four-layer forward and support membership; NumPy regression/meta adaptation and readout',
        tolerances=dict(mode_forward_atol=1e-12,head_atol=1e-9,head_rtol=1e-10,support_max=.001+1e-7),no_online_resource_claim=True)
    out.mkdir(parents=True,exist_ok=True);lock,identity=acquire_lock(out)
    try:
        if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==audit_protocol
        else:exclusive_json(out/'protocol.json',audit_protocol)
        protocol_hash=sha(out/'protocol.json');(out/'tasks').mkdir(exist_ok=True)
        if (out/'summary.json').exists():
            result=json.loads((out/'summary.json').read_text());assert result['passed']
            for n,h in result['outputs_sha256'].items():assert sha(out/n)==h
            print(json.dumps(dict(passed=True,already_complete=True,stage=args.stage)));return
        counts=Counter();files={};rows=[];maximum=0.;maxsupport=0.;begin=time.perf_counter()
        with discovery_box(.12):
            for seed,committed in zip(p['seeds'],before['task_commits']):
                assert verify_commit(inp,seed,p['methods'],sha(inp/'protocol.json'))==committed
                taskfile=out/'tasks'/f'{seed}.json'
                if taskfile.exists():
                    record=json.loads(taskfile.read_text());assert record['seed']==seed and record['protocol_sha256']==protocol_hash and record['prediction_commit_sha256']==committed['sha256']
                else:
                    c=json.loads((inp/committed['file']).read_text());taskrows=json.loads((inp/c['rows_file']).read_text());cc=Counter();gaps=[]
                    xx,vv=observations(seed);x=xx[:4];v=vv[:4];q=np.linspace(0,1,257)
                    for row in taskrows:
                        name=row['method'];cfg=cfgs[name];m=row['metadata'];assert sha(inp/row['file'])==row['sha256']
                        with np.load(inp/row['file']) as z:a={k:z[k].copy() for k in z.files}
                        assert a['x_observed'].tobytes()==x.tobytes() and a['v_observed'].tobytes()==v.tobytes() and a['q_observed'].tobytes()==q.tobytes()
                        for k,value in a.items():
                            if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),(seed,name,k)
                        assert a['prediction'].shape==a['point_prediction'].shape==(257,) and m['charged_complete_seconds']>0
                        assert m['geometry_repair_count']==len(m['geometry_repair_log'])
                        expected_kind='head' if cfg['family']=='prior' or cfg['group'] in ['additional_head','shallow'] else 'mode_pool'
                        assert m['method_kind']==expected_kind;support=None;state_checks=0
                        if m['execution_failed']:
                            assert not a['selected_b'].any() and m['fallback'] and not m['positive_modes']
                            assert m['failure_type'] in ['AssertionError','ArithmeticError','FloatingPointError','OverflowError','ZeroDivisionError','ValueError','RuntimeError','LinAlgError','QhullError']
                            expected_prediction=forward(q,np.zeros((1,4)))[0]
                            assert a['point_prediction'].tobytes()==a['prediction'].tobytes();cc['fixed_failure_fallbacks']+=1
                        elif expected_kind=='mode_pool':
                            points=a['points'];expected_prediction=forward(q,points).mean(0)
                            assert points.ndim==2 and points.shape[1]==4 and np.max(abs(points))<=.12+1e-9
                            keys=m['positive_modes'];assert keys==sorted(set(keys)) and set(keys)<=set(m['feasible_modes'])
                            if keys:
                                assert len(points)==2048 and len(a['allocation'])==len(keys) and a['allocation'].sum()==2048 and np.all(a['allocation']>=0)
                                support=float(np.max(abs(forward(x,points)-v)));assert support<=.001+1e-7
                                assert set(modes(x,points))<=set(keys);cc['support_particles']+=len(points)
                            else:
                                assert len(points)==1 and points[0].tobytes()==a['selected_b'].tobytes() and len(a['allocation'])==0;cc['empty_pools']+=1
                            if 'visited_modes' in m:assert set(m['feasible_modes'])<=set(m['visited_modes']) and m['lp_calls']==len(m['visited_modes'])
                            else:assert len(m['feasible_modes'])<=m['observed_modes']==m['lp_calls']
                            assert np.max(abs(forward(q,a['selected_b'][None])[0]-a['point_prediction']))<1e-12
                            assert m['retained_predictor_bytes']==points.nbytes;cc['independent_mode_readouts']+=1
                        else:
                            expected_prediction,states=heads.replay(cfg,x,v,q,a,m,loaded)
                            for k,value in states.items():np.testing.assert_allclose(value,a[k],rtol=1e-10,atol=1e-9);state_checks+=1
                            np.testing.assert_allclose(a['point_prediction'],a['prediction'],rtol=0,atol=0)
                            cc['independent_head_readouts']+=1;cc['independent_fast_state_arrays']+=state_checks
                        gap=float(np.max(abs(expected_prediction-a['prediction'])))
                        if expected_kind=='head' and not m['execution_failed']:np.testing.assert_allclose(expected_prediction,a['prediction'],rtol=1e-10,atol=1e-9)
                        else:assert gap<1e-12,(seed,name,gap)
                        gaps.append(dict(seed=seed,method=name,independent_forward_gap=gap,support_particle_max_error=support,
                            numerical_failure=m['execution_failed'],saved_numeric_array_bytes=sum(v.nbytes for v in a.values() if np.issubdtype(v.dtype,np.number))))
                        cc['predictors']+=1
                    cc['tasks']+=1;record=dict(seed=seed,protocol_sha256=protocol_hash,prediction_commit_sha256=committed['sha256'],counts=cc,rows=gaps,query_targets_accessed=False)
                    exclusive_json(taskfile,record)
                assert not record['query_targets_accessed'];counts.update(record['counts']);rows.extend(record['rows']);files[str(taskfile.relative_to(out))]=sha(taskfile)
                maximum=max(maximum,max(r['independent_forward_gap'] for r in record['rows']))
                maxsupport=max(maxsupport,max((r['support_particle_max_error'] or 0) for r in record['rows']))
                if counts['tasks']%64==0 or args.stage=='preflight':print(json.dumps(dict(tasks=counts['tasks'],total=expected,seconds=time.perf_counter()-begin,query_targets_accessed=False)),flush=True)
        assert counts['predictors']==expected*27 and counts['tasks']==expected and counts['fixed_failure_fallbacks']==summary['numerical_failures']
        for n,h in hashes.items():assert sha(src/n)==h,n
        exclusive_json(out/'files.json',files)
        ans=dict(passed=True,stage=args.stage,counts=counts,maximum_independent_forward_gap=maximum,maximum_support_particle_error=maxsupport,
            seconds_this_invocation=time.perf_counter()-begin,query_targets_accessed=False,may_evaluate=False,
            next='Fixed first64 original-path and geometry audit also required before query evaluation',outputs_sha256={n:sha(out/n) for n in ['protocol.json','files.json']})
        exclusive_json(out/'summary.json',ans);print(json.dumps(ans),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())==identity:lock.unlink()


if __name__=='__main__':main()

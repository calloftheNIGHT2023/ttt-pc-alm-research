"""397 independent prefix, receipt, optimizer-union, certificate and resource audit."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import math
import os
import time
import traceback
import numpy as np
import prefix_deadline_pilot_io_v1 as io


def audit(root,source,out):
    begin=time.perf_counter();summary=io.read(source/'summary.json')
    assert summary['passed'] and summary['all_predictions_sealed'] and not (source/'failure.json').exists()
    for f,h in summary['outputs_sha256'].items():assert io.sha(source/f)==h
    protocol=io.read(source/'protocol.json')
    io.old.verify_hashes(root,protocol['source_sha256']);io.old.verify_hashes(root,protocol['pretrained_sha256'])
    configurations=io.configs(root);assert configurations==protocol['configs']
    cfgs={c['name']:c for c in configurations};rows=io.read(source/'rows.json')
    expected=[(ci,c['name'],seed,n,budget) for ci,c,jobs in io.jobs(configurations,protocol['seeds'],protocol['stages'],protocol['budgets']) for seed,n,budget in jobs]
    assert [(r['config_index'],r['method'],r['seed'],r['n'],r['budget']) for r in rows]==expected
    assert len(rows)==protocol['expected_calls']==summary['calls']
    inputs={}
    for name,digest in io.read(source/'inputs_manifest.json').items():
        path=source/'inputs'/name;assert io.sha(path)==digest
        aa=io.load_arrays(path);seed=int(path.stem);x,v=io.observations(seed)
        np.testing.assert_array_equal(aa['x'],x);np.testing.assert_array_equal(aa['v'],v)
        np.testing.assert_array_equal(aa['q'],np.linspace(0.,1.,257));inputs[seed]=aa
    import torch
    import independent_hybrid_memory as regression
    import n24_optimizer_controls_v1 as optimizer
    import candidate_set_readout_v1 as reader
    import audit_contiguous_regional_repeat_v1 as whole
    from audit_new_task_deadline_risk_v1 import readout_check
    from test_deadline_risk_session_v1 import kernel_reference
    torch.set_num_threads(1)
    fallback={};bank_cache={};model_cache={};prediction_cache={};portfolio_cache={}
    counts=Counter();per_call=[];by_session=defaultdict(list)

    def bank_for(seed,n,family,steps,restarts):
        key=(seed,n,family,steps,restarts)
        if key not in bank_cache:
            x,v=inputs[seed]['x'][:n],inputs[seed]['v'][:n]
            with optimizer.prior_bounds():
                routine=optimizer.old.run_bp if family in ['adam','gauss_newton'] else optimizer.old.run_local
                bank,_,_=routine(optimizer.starts(restarts),x,v,family,steps,trace=False)
            bank_cache[key]=bank;counts['independent_optimizer_banks']+=1
        return bank_cache[key]

    def portfolio_for(seed,n,name):
        key=(seed,n,name)
        if key in portfolio_cache:return portfolio_cache[key]
        x,v,q=inputs[seed]['x'][:n],inputs[seed]['v'][:n],inputs[seed]['q']
        from prefix_matched_controls_v1 import PORTFOLIOS
        all_keys=set();events=[];expected_arrays={};best=None;best_error=float('inf')
        for j,(family,steps,restarts) in enumerate(PORTFOLIOS[name]):
            bank=bank_for(seed,n,family,steps,restarts);_,point=optimizer.old.select(bank,x,v)
            point_pred=np.clip(optimizer.old.base.forward(q,point),0.,1.)
            error=float(np.max(abs(optimizer.old.base.forward(x,point)-v)))
            if error<best_error:best_error=error;best=point.copy()
            all_keys.update(optimizer.old.base.pattern(x,p).astype(np.uint8).tobytes().hex() for p in bank)
            regions=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,n) for k in sorted(all_keys)])
            # Deliberately rebuild the entire union, without the candidate cache.
            ra,rm=reader.fit(x,v,q,regions,seed=seed,search_completed=False)
            counts.update(readout_check(ra,rm))
            prediction=ra['prediction'] if rm['readout_available'] else np.clip(optimizer.old.base.forward(q,best),0.,1.)
            events.append(prediction)
            raw=dict(x=x,v=v,q=q,starts=optimizer.starts(restarts),best_bank=bank,selected_point=point,
                point_prediction=point_pred,prediction=prediction,points=ra['points'],allocation=ra['allocation'])
            expected_arrays.update({f'stage_{j}_'+k:value for k,value in raw.items()})
        portfolio_cache[key]=(events,expected_arrays)
        return portfolio_cache[key]

    for ri,row in enumerate(rows):
        seed,n=row['seed'],row['n'];cfg=cfgs[row['method']];inp=inputs[seed]
        x,v,q=inp['x'][:n],inp['v'][:n],inp['q'];directory=root/row['directory']
        for f,h in row['files'].items():assert io.sha(directory/f)==h
        output=io.load_arrays(directory/'outputs.npz');meta=io.read(directory/'metadata.json');events=io.read(directory/'events.json')
        assert meta['config']==cfg and meta['budget_seconds']==row['budget'] and meta['error'] is None
        assert not meta['query_targets_accessed'] and not meta['shared_scientific_task_state_reused']
        assert meta['controller_overrun_seconds']==max(0.,meta['decision_seconds']-row['budget'])
        for k,mk in [('online_seconds','decision_seconds'),('received_final','received_final'),('selected','selected'),
            *[(k,k) for k in ['setup_seconds','cleanup_seconds','task_cleanup_seconds','archive_seconds','controller_overrun_seconds','worker_max_sampled_rss','controller_max_sampled_rss','worker_pid','generation']]]:
            assert row[k]==meta[mk]
        assert [e['ordinal'] for e in events]==list(range(len(events)))
        assert all(e['generation']==meta['generation'] for e in events)
        assert all(a['received_seconds']<=b['received_seconds'] for a,b in zip(events,events[1:]))
        assert all(0<=e['worker_seconds']<=e['received_seconds']+1e-3 for e in events)
        chosen=[(i,e) for i,e in enumerate(events) if e['kind'] in ['fallback','final'] and e['received_seconds']<=row['budget']]
        if chosen:
            i,event=chosen[-1];pred=output[f'event_{i}_prediction'];kind=event['kind']
        else:pred=np.full_like(q,.5);kind='constant'
        np.testing.assert_array_equal(pred,output['prediction']);assert kind==row['selected']
        for pred in output.values():
            assert pred.shape==q.shape and np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
        key=seed,n
        if key not in fallback:
            predictor,_,_=regression.regression(x,v,'prior4096_ridge');fallback[key]=np.clip(predictor(q),0.,1.)
        for i,event in enumerate(events):
            if event['kind']=='fallback':
                if cfg['kind']=='portfolio' and event['ordinal']>0:
                    expected_prediction=portfolio_for(seed,n,cfg['name'])[0][event['ordinal']-1]
                    counts['intermediate_portfolio_packet_replays']+=1
                else:
                    expected_prediction=fallback[key];counts['paid_prior_fallback_replays']+=1
                np.testing.assert_array_equal(output[f'event_{i}_prediction'],expected_prediction)
        samples=meta['resource_samples']
        assert all(s['worker_rss']>0 and s['controller_rss']>0 and s['worker_cpu_seconds']>=-1e-9 for s in samples)
        assert all(a['seconds']<=b['seconds'] for a,b in zip(samples,samples[1:]))
        assert meta['worker_max_sampled_rss']==max((s['worker_rss'] for s in samples),default=meta['setup_worker_memory']['rss'])
        cc=Counter()
        if meta['received_final']:
            assert events[-1]['kind']=='final' and meta['model_fingerprint_verified'] and not meta['terminated_owned_worker']
            aa=io.load_arrays(directory/'state.npz');mm=io.read(directory/'state.json')
            for k,value in [('x',x),('v',v),('q',q)]:np.testing.assert_array_equal(aa[k],value)
            np.testing.assert_array_equal(aa['prediction'],output[f'event_{len(events)-1}_prediction'])
            if cfg['kind']!='regression' or cfg['method'] in ['prior16384_ridge','prior65536_ridge']:
                np.testing.assert_array_equal(aa['fallback_prediction'],fallback[key]);assert mm['fallback_seconds']>0
            if cfg['kind']=='regional':
                scientific={k:value for k,value in aa.items() if k.startswith(('search_','readout_'))}
                cc.update(whole.audit_canonical(scientific,mm))
                full=mm['full_candidate_set_resolved'] and mm['readout_available']
                np.testing.assert_array_equal(aa['prediction'],aa['readout_prediction'] if full else fallback[key])
            elif cfg['kind']=='optimizer':
                bank=bank_for(seed,n,cfg['family'],cfg['steps'],cfg['restarts'])
                np.testing.assert_array_equal(bank,aa['best_bank']);np.testing.assert_array_equal(aa['starts'],optimizer.starts(cfg['restarts']))
                index,point=optimizer.old.select(bank,x,v);assert index==mm['selected_index']
                np.testing.assert_array_equal(point,aa['selected_point'])
                pred=np.clip(optimizer.old.base.forward(q,point),0.,1.)
                np.testing.assert_array_equal(pred,aa['point_prediction'])
                if cfg['readout']=='posterior_union':
                    ra={k[8:]:value for k,value in aa.items() if k.startswith('readout_')};cc.update(readout_check(ra,mm['readout_metadata']))
                    if mm['readout_metadata']['readout_available']:pred=ra['prediction']
                np.testing.assert_array_equal(pred,aa['prediction']);cc['optimizer_final_replays']+=1
            elif cfg['kind']=='portfolio':
                pp,expected_arrays=portfolio_for(seed,n,cfg['name'])
                for k,value in expected_arrays.items():np.testing.assert_array_equal(aa[k],value)
                np.testing.assert_array_equal(pp[-1],aa['prediction'])
                assert mm['task_local_geometry_reuse_only'] and not mm['cross_method_or_task_cache']
                cc['portfolio_archive_array_replays']+=len(expected_arrays)
            else:
                ck=seed,n,cfg['name']
                if ck not in prediction_cache:
                    if cfg['kind']=='cohort':
                        from prefix_matched_controls_v1 import load_model
                        if cfg['name'] not in model_cache:model_cache[cfg['name']]=load_model(cfg,root)[0]
                        model=model_cache[cfg['name']]
                        tx,tv,tq=[torch.from_numpy(value.copy())[None] for value in (x,v,q)]
                        with torch.no_grad():
                            state=None
                            for prefix in [p for p in io.STAGES if p<=n]:state=model.adapt(tx[:,:prefix],tv[:,:prefix],state)
                            raw=model.predict(state,tq)[0].numpy()
                        prediction_cache[ck]=np.clip(raw,0.,1.)
                        assert mm['stages']==[p for p in io.STAGES if p<=n] and mm['support_exposures']==sum(mm['stages'])
                    elif cfg['method'].startswith('prior'):
                        prediction_cache[ck]=kernel_reference(x,v,q,int(cfg['method'].split('_')[0][5:]))
                    elif cfg['method']=='residual_linear_ridge':
                        phi=np.column_stack([np.ones(n),x]);target=v-regression.base.forward(x,np.zeros(4))
                        w=np.linalg.solve(phi.T@phi+1e-4*np.eye(2),phi.T@target)
                        prediction_cache[ck]=np.clip(w[0]+q*w[1]+regression.base.forward(q,np.zeros(4)),0.,1.)
                    else:
                        predictor,_,_=regression.regression(x,v,cfg['method']);prediction_cache[ck]=np.clip(predictor(q),0.,1.)
                np.testing.assert_allclose(aa['prediction'],prediction_cache[ck],rtol=1e-10,atol=1e-10)
                cc['cohort_or_regression_prediction_replays']+=1
        else:
            assert meta['terminated_owned_worker'] and 'state.npz' not in row['files']
        counts.update(cc);counts['independent_deadline_choices']+=1;counts['resource_records']+=1
        by_session[row['method']].append(row);per_call.append(dict(directory=row['directory'],checks=dict(cc)))
        if (ri+1)%64==0:print(dict(stage='prefix_audit',calls=ri+1,seconds=time.perf_counter()-begin),flush=True)
    sessions=io.read(source/'sessions.json');assert len(sessions)==len(by_session)==42
    for ss in sessions:
        rr=by_session[ss['method']];assert len(rr)==ss['calls']
        assert rr[0]['generation']==0 and rr[0]['setup_seconds']>0
        for previous,current in zip(rr,rr[1:]):
            if previous['received_final']:
                assert current['generation']==previous['generation']+1 and current['worker_pid']==previous['worker_pid'] and current['setup_seconds']==0
            else:assert current['generation']==0 and current['setup_seconds']>0
        assert math.isclose(ss['total_setup_seconds'],math.fsum(r['setup_seconds'] for r in rr),rel_tol=0,abs_tol=1e-8)
        assert ss['total_closure_seconds']+1e-8>=math.fsum(r['cleanup_seconds'] for r in rr)
        counts['session_lifecycles']+=1
    io.old.verify_hashes(root,protocol['source_sha256']);io.old.verify_hashes(root,protocol['pretrained_sha256'])
    io.save(out/'checks.json',per_call)
    io.save(out/'summary.json',dict(passed=True,calls=len(rows),checks=dict(counts),query_targets_accessed=False,
        prediction_summary_sha256=io.sha(source/'summary.json'),seconds=time.perf_counter()-begin,
        outputs_sha256={'checks.json':io.sha(out/'checks.json')}))
    print(io.read(out/'summary.json'),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','run'],required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];source=root/io.BASE/('preflight_v1' if args.stage=='preflight' else 'development_v1')
    assert io.read(source/'summary.json')['passed']
    out=root/io.BASE/('audit_preflight_v1' if args.stage=='preflight' else 'audit_v1');out.mkdir(parents=True,exist_ok=False)
    try:audit(root,source,out)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

"""Cold context-block scaling, same algorithms and charged common recovery.

Each n starts with state=None. This is NOT the old four-stage warm-start stream.
All labels for queries remain outside adaptation and are evaluated last per task.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import run_recovered_online_comparison as prior

memory=prior.memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    x,v=prior.olddriver.observations(5900001);cfgs=[c for c in prior.configs() if c['family']!='regression']
    checks=0;bp_checks=0;empty_checks=0
    for n in [4,8,16,24]:
        states={};metas={}
        for cfg in cfgs:
            try:_,state,meta=prior.recovery.fit(x[:n],v[:n],None,cfg,np.random.default_rng(483001+n),64)
            except prior.recovery.RecoveryExhausted as exc:state=None;meta=dict(exhausted=True,attempts=exc.events)
            states[cfg['name']]=state;metas[cfg['name']]=meta
        for cfg in cfgs:
            a=states[cfg['name']];b=states[cfg['learner']+'_c5']
            assert (a is None)==(b is None)
            if a is None:
                assert all(not e['success'] for e in metas[cfg['name']]['attempts']);empty_checks+=1
            else:
                assert a.samples.tobytes()==b.samples.tobytes() and a.anchor.tobytes()==b.anchor.tobytes()
                assert metas[cfg['name']]['positive_mode_keys']==metas[cfg['learner']+'_c5']['positive_mode_keys'];checks+=1
        core=memory.neighbor.previous.core;jac=memory.base.forward_jacobian;bp=core.batched.refine
        def forbidden(*args,**kwargs):raise AssertionError('Global BP entered local cold-block candidate')
        try:
            memory.base.forward_jacobian=forbidden;core.batched.refine=forbidden
            cfg=next(c for c in cfgs if c['name']==prior.PRIMARY)
            try:_,state,_=prior.recovery.fit(x[:n],v[:n],None,cfg,np.random.default_rng(483001+n),64)
            except prior.recovery.RecoveryExhausted:state=None
            assert (state is None)==(states[prior.PRIMARY] is None)
            if state is not None:assert state.samples.tobytes()==states[prior.PRIMARY].samples.tobytes()
            bp_checks+=1
        finally:memory.base.forward_jacobian=jac;core.batched.refine=bp
        print(json.dumps(dict(preflight_n=n,equivalence_checks=checks,no_global_bp=bp_checks)),flush=True)
    return dict(passed=True,within_learner_cold_block_equivalence=checks,matched_recovery_exhaustion=empty_checks,no_global_bp_cold_blocks=bp_checks)


def trial(xx,vv,q,cfg,seed,rep,n,p,out):
    rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,n,p['posterior_samples']]))
    row=dict(seed=seed,method=cfg['name'],family=cfg['family'],learner=cfg.get('learner'),repetition=rep,n=n,complete=False)
    start=time.perf_counter()
    try:
        if cfg['family']=='regression':pred,state,meta=prior.regression(xx[:n],vv[:n],cfg['name'],None)
        else:pred,state,meta=prior.recovery.fit(xx[:n],vv[:n],None,cfg,rng,p['posterior_samples'])
    except Exception as exc:
        row.update(charged_seconds=time.perf_counter()-start,traceback=traceback.format_exc(),recovery_attempts=getattr(exc,'events',None))
        return row
    write=time.perf_counter()-start;start=time.perf_counter();prediction=pred(q);read=time.perf_counter()-start
    assert np.all(np.isfinite(prediction));arrays=dict(x=xx[:n],v=vv[:n],q=q,prediction=prediction)
    if cfg['family']!='regression':
        assert meta['previous_state_digest'] is None
        arrays.update(points=state.samples,anchor=state.anchor)
        _,h=memory.prior.capture.model.light.forward_many(xx[:n],state.samples)
        assert np.max(abs(h[:,-1]-vv[:n]))<=.001+1e-8
        assert sum(e['seconds'] for e in meta['recovery']['attempts'])<=write+1e-6
    elif state is not None:arrays.update(rls_w=state['w'],rls_p=state['p'])
    path=out/f'state_{seed}_{cfg["name"]}_{rep}_{n}.npz';np.savez_compressed(path,**arrays)
    row.update(complete=True,write_seconds=write,read_seconds=read,charged_seconds=write+read,state_bytes=meta['persistent_state_bytes'],
        state_file=path.name,state_sha256=sha(path),state_digest=meta.get('new_state_digest'),previous_state_digest=meta.get('previous_state_digest'),
        recovery=meta.get('recovery'),geometry_calls=meta.get('geometry_calls'),screen_calls=meta.get('screen_calls',[]),
        positive_mode_keys=meta.get('positive_mode_keys'),discovery_bank_sha256=meta.get('discovery_bank_sha256'),
        completion_proposals=meta.get('completion_proposals'),selected_length=meta.get('selected_length'),
        selected_ridge=meta.get('selected_ridge'),support_press=meta.get('support_press'))
    if rep==0 and cfg['family']!='regression':
        path=out/f'detail_{seed}_{cfg["name"]}_{n}.json';path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        row.update(detail_file=path.name,detail_sha256=sha(path))
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/recovered_online_comparison/development';conditional=root/'results/recovered_conditional_risk/development'
    parent=json.loads((inp/'protocol.json').read_text());hashes=dict(json.loads((conditional/'protocol.json').read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=json.loads((root/'results/round_208_audit.json').read_text());assert audit['passed']
    assert audit['source_sha256']==sha(Path(__file__).with_name('audit_research_round_208.py'))
    verification=verify()
    if args.preflight_only:print(json.dumps(verification),flush=True);return
    hashes[Path(__file__).name]=sha(Path(__file__))
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),risk_audit_sha256=sha(root/'results/round_208_audit.json'),
        configs=prior.configs(),seeds=parent['seeds'],block_sizes=[4,8,16,24],repetitions=2,posterior_samples=2048,
        rng_seed=parent['rng_seed'],order_seed=483019,query_points=257,primary=prior.PRIMARY,primary_comparator='alm_c5',
        recovery_feature_budgets=list(prior.recovery.FEATURE_BUDGETS),verification=verification,
        scope='Cold initial context blocks; all 16 reused development tasks; NOT four-stage warm-start streams or confirmation',
        failure_policy='Every failure retained and charged; no successful-subset quality average')
    out=root/'results/context_block_scaling/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    originals={(r['seed'],r['method'],r['repetition']):r for r in json.loads((inp/'rows.json').read_text()) if r['n']==4}
    order=np.random.default_rng(p['order_seed']);q=np.linspace(0,1,p['query_points']);rows=[];done=0
    checks=dict(n4_matches_original203=0,within_learner_states_bytewise=0,terminal_equivalence=0)
    jobs=[(cfg,rep,n) for cfg in p['configs'] for rep in range(p['repetitions']) for n in p['block_sizes']]
    for seed in p['seeds']:
        xx,vv=prior.olddriver.observations(seed);seedrows=[]
        for index in order.permutation(len(jobs)):
            cfg,rep,n=jobs[index];row=trial(xx,vv,q,cfg,seed,rep,n,p,out);seedrows.append(row);done+=1
            if n==4:
                assert row['complete'];ref=originals[seed,cfg['name'],rep];assert sha(inp/ref['state_file'])==ref['state_sha256']
                with np.load(out/row['state_file']) as a,np.load(inp/ref['state_file']) as b:
                    assert set(a.files)==set(b.files)
                    assert all(a[k].tobytes()==b[k].tobytes() for k in a.files),(seed,cfg['name'],rep)
                checks['n4_matches_original203']+=1
            print(json.dumps(dict(completed=done,total=len(jobs)*len(p['seeds']),seed=seed,method=cfg['name'],rep=rep,n=n,
                complete=row['complete'],recovered=bool(row.get('recovery',{}).get('triggered')) if row.get('recovery') else False,
                seconds=row['charged_seconds'])),flush=True)
        lookup={(r['method'],r['repetition'],r['n']):r for r in seedrows}
        for row in seedrows:
            if row['family']=='regression':continue
            ref=lookup[row['learner']+'_c5',row['repetition'],row['n']]
            assert ref['complete']==row['complete'];checks['terminal_equivalence']+=1
            if row['complete']:
                with np.load(out/row['state_file']) as a,np.load(out/ref['state_file']) as b:
                    assert all(a[k].tobytes()==b[k].tobytes() for k in a.files)
                assert row['positive_mode_keys']==ref['positive_mode_keys'];checks['within_learner_states_bytewise']+=1
        # Teacher/query answers revealed only after all 144 adaptation states and checks.
        truth=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            if row['complete']:
                with np.load(out/row['state_file']) as z:row['query_mse']=float(np.trapezoid((z['prediction']-truth)**2,x=q))
        rows.extend(seedrows);(out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,adaptations=done,successes=sum(r['complete'] for r in rows),failures=sum(not r['complete'] for r in rows),
        recovery_events=sum(bool(r.get('recovery') and r['recovery']['triggered']) for r in rows),checks=checks,sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

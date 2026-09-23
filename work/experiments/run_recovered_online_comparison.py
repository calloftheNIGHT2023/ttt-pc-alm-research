"""Full, charged own-state online comparison with a common bounded recovery.

Replays all tasks, not patched historical timings. Successful states must
match the frozen no-recovery run until the first actual recovery event.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import common_prior_recovery as recovery
import run_light_h2_credit as olddriver

memory=recovery.memory
PRIMARY='alm_native_full_c5_interval_endpoints'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def configs():
    names=['alm_c5','alm_c20','adam60_c5','pc_c5','nodual_c5','direct4096_c5',PRIMARY,
        'adam60_native_full_c5_interval_endpoints','pc_native_full_c5_interval_endpoints','nodual_native_full_c5_interval_endpoints',
        'alm_residual_full_c5_interval_endpoints','adam60_residual_full_c5_interval_endpoints']
    pool={c['name']:c for c in memory.configs()}
    return [pool[n] for n in names]+[dict(name=n,family='regression') for n in
        ['linear_ls','residual_linear_ls','residual_linear_ridge','residual_linear_rls','prior4096_ridge','rbf_loocv']]


def regression(x,v,name,state):
    if name!='residual_linear_ridge':return memory.prior.regression(x,v,name,state)
    phi=np.column_stack([np.ones(len(x)),x]);y=v-memory.base.forward(x,np.zeros(4))
    w=np.linalg.solve(phi.T@phi+1e-4*np.eye(2),phi.T@y)
    def predict(q):return w[0]+q*w[1]+memory.base.forward(q,np.zeros(4))
    return predict,None,dict(persistent_state_bytes=w.nbytes,regression_name=name,ridge_lambda=1e-4)


def verify():
    x,v=olddriver.observations(5900001);state=None;checks=0
    for n in [4,8,16,24]:
        batch,_,_=regression(x[:n],v[:n],'residual_linear_ridge',None)
        stream,state,_=regression(x[:n],v[:n],'residual_linear_rls',state)
        assert np.max(abs(batch(x)-stream(x)))<1e-9;checks+=1
    return dict(recovery=recovery.verify(),ridge_rls_four_stage_equivalence=checks)


def trial(xx,vv,q,cfg,seed,rep,p,out=None):
    state=None;rows=[];artifacts=[];failure=None;ever_recovered=False
    for n in p['stages']:
        rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,n,p['posterior_samples']]))
        start=time.perf_counter()
        try:
            if cfg['family']=='regression':pred,new,meta=regression(xx[:n],vv[:n],cfg['name'],state)
            else:pred,new,meta=recovery.fit(xx[:n],vv[:n],state,cfg,rng,p['posterior_samples'])
        except Exception as exc:
            failure=dict(seed=seed,method=cfg['name'],repetition=rep,failed_n=n,failed_seconds=time.perf_counter()-start,
                traceback=traceback.format_exc(),recovery_attempts=getattr(exc,'events',None));break
        write=time.perf_counter()-start;start=time.perf_counter();prediction=pred(q);read=time.perf_counter()-start
        assert np.all(np.isfinite(prediction));arrays=dict(x=xx[:n],v=vv[:n],q=q,prediction=prediction)
        if cfg['family']!='regression':
            assert memory.prior.state_digest(state)==meta['previous_state_digest']
            arrays.update(points=new.samples,anchor=new.anchor)
            _,h=memory.prior.capture.model.light.forward_many(xx[:n],new.samples)
            assert np.max(abs(h[:,-1]-vv[:n]))<=.001+1e-8
            ever_recovered|=meta['recovery']['triggered']
            assert sum(e['seconds'] for e in meta['recovery']['attempts'])<=write+1e-6
        elif new is not None:arrays.update(rls_w=new['w'],rls_p=new['p'])
        row=dict(seed=seed,method=cfg['name'],family=cfg['family'],learner=cfg.get('learner'),repetition=rep,n=n,
            write_seconds=write,read_seconds=read,total_seconds=write+read,state_bytes=meta['persistent_state_bytes'],
            state_digest=meta.get('new_state_digest'),previous_state_digest=meta.get('previous_state_digest'),
            recovery=meta.get('recovery'),ever_recovered=ever_recovered,geometry_calls=meta.get('geometry_calls'),
            selected_length=meta.get('selected_length'),selected_ridge=meta.get('selected_ridge'),
            support_press=meta.get('support_press'),screen_calls=meta.get('screen_calls',[]))
        if out is not None:
            file=out/f'state_{seed}_{cfg["name"]}_{rep}_{n}.npz';np.savez_compressed(file,**arrays)
            row.update(state_file=file.name,state_sha256=sha(file))
            if rep==0 and n==4 and cfg['family']!='regression':
                detail=out/f'detail_{seed}_{cfg["name"]}.json';detail.write_text(json.dumps(meta,indent=2),encoding='utf-8')
                row.update(detail_file=detail.name,detail_sha256=sha(detail))
        rows.append(row);artifacts.append(arrays);state=new
    return rows,artifacts,failure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_endpoint_h2/development';diagnostic=root/'results/common_prior_recovery/development'
    assert json.loads((root/'results/round_201_audit.json').read_text())['passed']
    assert json.loads((diagnostic/'independent_audit.json').read_text())['passed']
    parent=json.loads((inp/'protocol.json').read_text());hashes=dict(json.loads((diagnostic/'protocol.json').read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),diagnostic_protocol_sha256=sha(diagnostic/'protocol.json'),
        configs=configs(),seeds=parent['seeds'],repetitions=2,stages=parent['stages'],posterior_samples=2048,rng_seed=parent['rng_seed'],
        order_seed=482701,query_points=257,primary=PRIMARY,recovery_feature_budgets=list(recovery.FEATURE_BUDGETS),verification=verify(),
        scope='18 methods with common bounded recovery, same 16 development tasks; all online timings rerun; not confirmation or official TTT',
        failure_policy='Preserve partial rows and failed attempt cost; do not drop tasks or impute successful-only means')
    out=root/'results/recovered_online_comparison/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    originals={(r['seed'],r['method'],r['repetition'],r['n']):r for r in json.loads((inp/'rows.json').read_text())}
    rows=[];episodes=[];failures=[];q=np.linspace(0,1,p['query_points']);order=np.random.default_rng(p['order_seed']);done=0
    jobs=[(cfg,rep) for cfg in p['configs'] for rep in range(2)];checks=dict(unchanged_no_recovery_states_bytewise=0)
    for seed in p['seeds']:
        xx,vv=olddriver.observations(seed);seedrows=[]
        for index in order.permutation(len(jobs)):
            cfg,rep=jobs[index];rr,artifacts,failure=trial(xx,vv,q,cfg,seed,rep,p,out)
            if failure:failures.append(failure)
            for row,arrays in zip(rr,artifacts):
                if cfg['family']!='regression' and not row['ever_recovered']:
                    ref=originals[seed,cfg['name'],rep,row['n']];assert sha(inp/ref['state_file'])==ref['state_sha256']
                    with np.load(inp/ref['state_file']) as z:assert all(arrays[k].tobytes()==z[k].tobytes() for k in arrays)
                    checks['unchanged_no_recovery_states_bytewise']+=1
            ep=dict(seed=seed,method=cfg['name'],repetition=rep,complete=failure is None,stages=len(rr),
                failed_n=None if failure is None else failure['failed_n'],
                charged_seconds=sum(r['total_seconds'] for r in rr)+(0 if failure is None else failure['failed_seconds']),
                recovery_stages=sum(bool(r['recovery'] and r['recovery']['triggered']) for r in rr))
            episodes.append(ep);seedrows.extend(rr);done+=1
            print(json.dumps(dict(completed=done,total=len(jobs)*len(p['seeds']),seed=seed,method=cfg['name'],rep=rep,
                complete=ep['complete'],recoveries=ep['recovery_stages'],seconds=ep['charged_seconds'])),flush=True)
        truth=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            with np.load(out/row['state_file']) as z:row['query_mse']=float(np.trapezoid((z['prediction']-truth)**2,x=q))
        rows.extend(seedrows)
        for name,value in [('rows.json',rows),('episodes.json',episodes),('failures.json',failures)]:
            (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,episodes=done,stages=len(rows),failures=len(failures),
        recovery_events=sum(e['recovery_stages'] for e in episodes),checks=checks,sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

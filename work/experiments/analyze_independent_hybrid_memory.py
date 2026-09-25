"""Independent arithmetic replay, own-state audit, and task-level contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(x,b):
    h=np.broadcast_to(x,(len(b),len(x)))
    for j in range(4):h=np.maximum(0.,1.-np.abs(2.*(h+b[:,j,None])-1.))
    return h


def ci(values):
    a=np.array(values);rng=np.random.default_rng(481861)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';out=root/'results/independent_hybrid/analysis'
    p=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'rows.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    assert run['execution_complete'] and len(rows)==run['stages']
    failures=json.loads((inp/'failures.json').read_text()) if (inp/'failures.json').exists() else []
    assert len(failures)==run['failures'] and len(rows)+4*len(failures)==run['expected_stages']
    failed={(r['seed'],r['method'],r['repetition']) for r in failures}
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    lookup={(r['seed'],r['method'],r['repetition'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    checks=dict(state_hashes=0,posterior_prediction_replays=0,regression_prediction_replays=0,
        risk_replays=0,own_state_links=0,support_particles=0,h2_details=0,first_write_credit_families=0,
        same_discovery_safe_positive_set_checks=0)
    max_error=0.;max_support=0.;tasks=[]
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);latent=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
        vv=forward(xx,latent[None])[0]+np.random.default_rng(seed+19000000).uniform(-.001,.001,24)
        q=np.linspace(0,1,p['query_points']);truth=forward(q,latent[None])[0]
        for prefix in ['alm','adam60','pc','nodual']:
            geometry=json.loads((inp/f'detail_{seed}_{prefix}_geometry.json').read_text())
            hybrid=json.loads((inp/f'detail_{seed}_{prefix}_hybrid.json').read_text())
            assert set(geometry['positive_mode_keys'])<=set(hybrid['surviving_pattern_keys'])
            assert geometry['pool']==hybrid['pool']
            checks['same_discovery_safe_positive_set_checks']+=1
        for cfg in p['configs']:
            episode_metrics=[];incomplete=False
            for rep in range(p['repetitions']):
                if (seed,cfg['name'],rep) in failed:
                    assert not any((seed,cfg['name'],rep,n) in lookup for n in p['stages'])
                    incomplete=True;continue
                previous=None;stage_rows=[]
                for n in p['stages']:
                    row=lookup[seed,cfg['name'],rep,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256'];checks['state_hashes']+=1
                    with np.load(path) as z:
                        assert np.array_equal(z['x'],xx[:n]) and np.array_equal(z['v'],vv[:n]) and np.array_equal(z['q'],q)
                        prediction=z['prediction'].copy()
                        if cfg['family']!='regression':
                            points=z['points'];anchor=z['anchor']
                            assert len(points)==p['posterior_samples'] and np.array_equal(anchor,points[0])
                            assert np.max(abs(points))<=.12+1e-12
                            error=float(np.max(abs(forward(xx[:n],points)-vv[:n])))
                            assert error<=.001+1e-8 and abs(error-row['max_support_error'])<1e-13
                            max_support=max(max_support,error);checks['support_particles']+=len(points)
                            digest=hashlib.sha256(anchor.tobytes()+points.tobytes()).hexdigest()
                            assert row['new_state_digest']==digest and row['previous_state_digest']==previous
                            if n>4:
                                assert row['effective_generator']=='direct' and row['previous_posterior_samples']==p['posterior_samples']
                                checks['own_state_links']+=1
                            previous=digest
                            replay=np.zeros(len(q))
                            for start in range(0,len(points),193):
                                pp=points[::-1][start:start+193]
                                for begin in range(0,len(q),67):replay[begin:begin+67]+=forward(q[begin:begin+67],pp).sum(0)/len(points)
                            checks['posterior_prediction_replays']+=1
                            if n==4 and cfg['family']=='typed':
                                assert row['effective_generator']==cfg['generator'] and row['collector_kinds']==cfg['credits'];checks['first_write_credit_families']+=1
                        else:
                            # Independently reconstruct fixed features/batch
                            # regression. Streaming RLS is checked by batch.
                            phi=np.column_stack([np.ones(n),xx[:n]])
                            if cfg['name'] in ['linear_ls','residual_linear_ls','residual_linear_rls']:
                                residual=cfg['name']!='linear_ls';y=vv[:n]-forward(xx[:n],np.zeros((1,4)))[0] if residual else vv[:n]
                                if cfg['name']=='residual_linear_rls':w=np.linalg.solve(phi.T@phi+1e-4*np.eye(2),phi.T@y);assert np.max(abs(w-z['rls_w']))<1e-8
                                else:w=np.linalg.lstsq(phi,y,rcond=None)[0]
                                replay=np.column_stack([np.ones(len(q)),q])@w
                                if residual:replay+=forward(q,np.zeros((1,4)))[0]
                            elif cfg['name']=='prior4096_ridge':
                                bank=np.random.default_rng(731).uniform(-.12,.12,(4096,4));features=forward(xx[:n],bank)
                                mu=features.mean(0);centered=features-mu
                                alpha=np.linalg.solve(centered.T@centered/4096+np.eye(n)*.001**2/3,vv[:n]-mu)
                                weights=(1+centered@alpha)/4096;replay=weights@forward(q,bank)
                            else:
                                best=None;mean=vv[:n].mean();y=vv[:n]-mean
                                for length in [.01,.02,.04,.08,.16,.32]:
                                    gram=np.exp(-.5*((xx[:n,None]-xx[None,:n])/length)**2)
                                    for ridge in [1e-6,1e-4,.01,.1,1.]:
                                        inv=np.linalg.solve(gram+ridge*np.eye(n),np.eye(n));alpha=inv@y;press=float(np.mean((alpha/np.diag(inv))**2))
                                        if best is None or press<best[0]:best=(press,length,alpha)
                                replay=mean+np.exp(-.5*((q[:,None]-xx[None,:n])/best[1])**2)@best[2]
                            checks['regression_prediction_replays']+=1
                    error=float(np.max(abs(replay-prediction)));max_error=max(max_error,error);assert error<1e-8,(cfg['name'],n,error)
                    mse=float(np.trapezoid((replay-truth)**2,x=q));assert abs(mse-row['query_mse'])<1e-9;checks['risk_replays']+=1
                    if 'detail_file' in row:
                        detail=inp/row['detail_file'];assert sha(detail)==row['detail_sha256']
                        if cfg['family']=='h2':assert json.loads(detail.read_text())['geometry_reused'];checks['h2_details']+=1
                    stage_rows.append(row)
                episode_metrics.append(dict(time=sum(r['total_seconds'] for r in stage_rows),mse=float(np.mean([r['query_mse'] for r in stage_rows])),
                    first_mse=stage_rows[0]['query_mse'],last_mse=stage_rows[-1]['query_mse'],first_time=stage_rows[0]['total_seconds'],
                    first_fallback=float(stage_rows[0]['sampling'].get('fallback',False)),
                    final_state_bytes=stage_rows[-1]['persistent_state_bytes']))
            if not incomplete:
                keys=list(episode_metrics[0]);tasks.append(dict(seed=seed,method=cfg['name'],**{k:float(np.mean([r[k] for r in episode_metrics])) for k in keys}))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    lookup={(r['seed'],r['method']):r for r in tasks};summaries=[];paired=[];stages=[]
    for cfg in p['configs']:
        name=cfg['name'];rr=[r for r in tasks if r['method']==name]
        if len(rr)!=len(p['seeds']):
            summaries.append(dict(method=name,family=cfg['family'],complete_tasks=len(rr),failed_trials=sum(r['method']==name for r in failures),
                full_task_mean_available=False,time=None,mse=None))
            paired.append(dict(comparator=name,available=False,reason='Failed episodes retained; no successful-only or imputed full-task contrast'))
            continue
        summaries.append(dict(method=name,family=cfg['family'],complete_tasks=len(rr),failed_trials=0,full_task_mean_available=True,
            **{k:float(np.mean([r[k] for r in rr])) for k in keys}))
        for n in p['stages']:
            rr=[r for r in rows if r['method']==name and r['n_context']==n]
            stages.append(dict(method=name,n=n,time=float(np.mean([r['total_seconds'] for r in rr])),mse=float(np.mean([r['query_mse'] for r in rr]))))
        if name=='alm_hybrid':continue
        delta={k:np.array([lookup[s,'alm_hybrid'][k]-lookup[s,name][k] for s in p['seeds']]) for k in ['time','mse','first_time','first_mse','last_mse']}
        paired.append(dict(comparator=name,delta={k:float(v.mean()) for k,v in delta.items()},descriptive_ci95={k:ci(v) for k,v in delta.items()},
            faster_tasks=int(np.sum(delta['time']<0)),lower_risk_tasks=int(np.sum(delta['mse']<0))))
    result=dict(complete=True,all_methods_completed=not failures,failures=failures,checks=checks,max_prediction_replay_error=max_error,max_support_error=max_support,
        source_sha256=sha(Path(__file__)),input_sha256={n:sha(inp/n) for n in ['protocol.json','rows.json','run_audit.json']},
        summaries=summaries,paired=paired,scope='16 fresh development tasks, task-level averaging, no multiplicity correction or confirmation claim')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('tasks.json',tasks),('stages.json',stages)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    names=['alm_hybrid','alm_geometry','adam60_hybrid','adam60_h2','direct4096_h2','pc_hybrid','nodual_hybrid','prior4096_ridge']
    colors=plt.get_cmap('tab10').colors
    for i,name in enumerate(names):
        row=next(r for r in summaries if r['method']==name)
        if not row['full_task_mean_available']:continue
        axes[0].scatter(row['time'],row['mse'],color=colors[i],s=65,label=name)
        rr=[r for r in stages if r['method']==name]
        axes[1].plot([r['n'] for r in rr],[r['mse'] for r in rr],marker='o',label=name,color=colors[i])
    axes[0].set(xlabel='Full four-stage time (s; writes + reads)',ylabel='Mean unseen-query MSE',title='Own discovery and own online state')
    axes[1].set(xlabel='Observed support count',ylabel='Unseen-query MSE',title='Sample efficiency on new development tasks',yscale='log')
    for ax in axes:ax.grid(alpha=.22)
    axes[0].legend(fontsize=8,loc='best');fig.suptitle('16 new tasks, two repetitions; descriptive, not confirmation',fontsize=12)
    fig.tight_layout();fig.savefig(out/'independent_hybrid.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(complete=True,checks=checks,summary=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

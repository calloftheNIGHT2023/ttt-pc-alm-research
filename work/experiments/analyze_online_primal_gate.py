"""Independent arithmetic, complete-task contrasts and recovery accounting."""
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
    a=np.array(values);rng=np.random.default_rng(482709)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def regression_replay(x,v,q,name,z,row):
    phi=np.column_stack([np.ones(len(x)),x]);n=len(x)
    if name in ['linear_ls','residual_linear_ls','residual_linear_ridge','residual_linear_rls']:
        residual=name!='linear_ls';y=v-forward(x,np.zeros((1,4)))[0] if residual else v
        ridge=name in ['residual_linear_rls','residual_linear_ridge']
        w=np.linalg.solve(phi.T@phi+1e-4*np.eye(2),phi.T@y) if ridge else np.linalg.lstsq(phi,y,rcond=None)[0]
        if name=='residual_linear_rls':
            assert np.max(abs(w-z['rls_w']))<1e-8
            assert np.max(abs(np.linalg.inv(phi.T@phi+1e-4*np.eye(2))-z['rls_p']))<1e-8
        replay=np.column_stack([np.ones(len(q)),q])@w
        if residual:replay+=forward(q,np.zeros((1,4)))[0]
        return replay
    if name=='prior4096_ridge':
        bank=np.random.default_rng(731).uniform(-.12,.12,(4096,4));features=forward(x,bank);mu=features.mean(0);centered=features-mu
        alpha=np.linalg.solve(centered.T@centered/4096+np.eye(n)*.001**2/3,v-mu)
        return ((1+centered@alpha)/4096)@forward(q,bank)
    assert name=='rbf_loocv';best=None;mean=v.mean();y=v-mean
    for length in [.01,.02,.04,.08,.16,.32]:
        gram=np.exp(-.5*((x[:,None]-x[None,:])/length)**2)
        for ridge in [1e-6,1e-4,.01,.1,1.]:
            inv=np.linalg.solve(gram+ridge*np.eye(n),np.eye(n));alpha=inv@y;press=float(np.mean((alpha/np.diag(inv))**2))
            if best is None or press<best[0]:best=(press,length,ridge,alpha)
    assert best[1]==row['selected_length'] and best[2]==row['selected_ridge'] and abs(best[0]-row['support_press'])<1e-10
    return mean+np.exp(-.5*((q[:,None]-x[None,:])/best[1])**2)@best[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_primal_gate/development_v2';out=root/'results/online_primal_gate/analysis'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text());failures=json.loads((inp/'failures.json').read_text())
    assert len(rows)==run['stages']==sum(e['stages'] for e in episodes) and len(failures)==run['failures']
    lookup={(r['seed'],r['method'],r['repetition'],r['n']):r for r in rows};eps={(e['seed'],e['method'],e['repetition']):e for e in episodes}
    assert len(lookup)==len(rows) and len(eps)==len(episodes)
    checks=dict(states=0,posterior_replays=0,regression_replays=0,support_particles=0,own_state_links=0,
        risk_replays=0,recovery_events=0,charged_attempts=0,within_learner_state_bytes=0,proof_details_equal=0,rls_batch_stages=0)
    old=root/'results/recovered_online_comparison/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text()) if r['repetition']==0 and r['n']==4}
    tasks=[];max_error=0.
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
        vv=forward(xx,teacher[None])[0]+np.random.default_rng(seed+19000000).uniform(-.001,.001,24);q=np.linspace(0,1,p['query_points']);truth=forward(q,teacher[None])[0]
        for cfg in p['configs']:
            name=cfg['name'];trial_metrics=[]
            for rep in range(p['repetitions']):
                ep=eps[seed,name,rep];previous=None;ever=False;rr=[]
                for n in p['stages'][:ep['stages']]:
                    row=lookup[seed,name,rep,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256'];checks['states']+=1
                    with np.load(path) as z:
                        assert np.array_equal(xx[:n],z['x']) and np.array_equal(vv[:n],z['v']) and np.array_equal(q,z['q'])
                        if cfg['family']!='regression':
                            points=z['points'];anchor=z['anchor'];assert len(points)==2048 and np.array_equal(anchor,points[0])
                            assert np.max(abs(points))<=.12+1e-12 and np.max(abs(forward(xx[:n],points)-vv[:n]))<=.001+1e-8
                            digest=hashlib.sha256(anchor.tobytes()+points.tobytes()).hexdigest()
                            assert row['state_digest']==digest and row['previous_state_digest']==previous;previous=digest
                            checks['own_state_links']+=int(n>4);checks['support_particles']+=len(points)
                            replay=np.zeros(len(q))
                            for start in range(0,len(points),193):replay+=forward(q,points[::-1][start:start+193]).sum(0)/len(points)
                            rec=row['recovery'];events=rec['attempts'];assert events[-1]['success'] and all(not e['success'] for e in events[:-1])
                            assert all(e['seconds']>=0 for e in events) and sum(e['seconds'] for e in events)<=row['write_seconds']+1e-6
                            if rec['triggered']:
                                assert rec['all_observations_reused']==n and events[0]['error']=='Geometry found no nonzero cell'
                                assert [e['features'] for e in events[1:]]==p['recovery_feature_budgets'][:len(events)-1]
                                checks['recovery_events']+=1
                            else:assert len(events)==1
                            ever|=rec['triggered'];assert row['ever_recovered']==ever;checks['charged_attempts']+=len(events)
                            baseline=lookup[seed,cfg['learner']+'_c5',rep,n]
                            with np.load(inp/baseline['state_file']) as b:assert all(z[k].tobytes()==b[k].tobytes() for k in z.files)
                            checks['within_learner_state_bytes']+=1;checks['posterior_replays']+=1
                        else:
                            replay=regression_replay(xx[:n],vv[:n],q,name,z,row);checks['regression_replays']+=1
                            if name=='residual_linear_rls':checks['rls_batch_stages']+=1
                        error=float(np.max(abs(replay-z['prediction'])));assert error<1e-8,(name,n,error);max_error=max(max_error,error)
                        risk=float(np.trapezoid((replay-truth)**2,x=q));assert abs(risk-row['query_mse'])<1e-9;checks['risk_replays']+=1
                    if 'detail_file' in row:
                        assert sha(inp/row['detail_file'])==row['detail_sha256'];detail=json.loads((inp/row['detail_file']).read_text());ref=oldrows[seed,cfg.get('ungated_comparator',name)]
                        assert sha(old/ref['detail_file'])==ref['detail_sha256'];reference=json.loads((old/ref['detail_file']).read_text())
                        for k in ['credit_proofs','credit_bank','positive_mode_keys','discovery_bank_sha256']:
                            assert detail.get(k)==reference.get(k)
                        checks['proof_details_equal']+=1
                    rr.append(row)
                assert len(rr)>0
                time_sum=sum(r['total_seconds'] for r in rr)
                if ep['complete']:assert abs(time_sum-ep['charged_seconds'])<1e-9
                else:assert ep['charged_seconds']>=time_sum and ep['failed_n']==p['stages'][len(rr)]
                trial_metrics.append(dict(complete=ep['complete'],time=time_sum if ep['complete'] else None,
                    mse=float(np.mean([r['query_mse'] for r in rr])) if ep['complete'] else None,
                    first_time=rr[0]['total_seconds'],first_mse=rr[0]['query_mse'],last_mse=rr[-1]['query_mse'] if ep['complete'] else None,
                    recoveries=ep['recovery_stages'],charged_seconds=ep['charged_seconds'],final_state_bytes=rr[-1]['state_bytes']))
            numeric=[k for k in trial_metrics[0] if k!='complete'];record=dict(seed=seed,method=name,complete=all(r['complete'] for r in trial_metrics))
            for k in numeric:record[k]=None if any(r[k] is None for r in trial_metrics) else float(np.mean([r[k] for r in trial_metrics]))
            tasks.append(record)
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    lookup={(r['seed'],r['method']):r for r in tasks};summaries=[];paired=[];stages=[]
    for cfg in p['configs']:
        name=cfg['name'];rr=[r for r in tasks if r['method']==name]
        summary=dict(method=name,complete_tasks=sum(r['complete'] for r in rr),family=cfg['family'])
        for k in numeric:summary[k]=None if any(r[k] is None for r in rr) else float(np.mean([r[k] for r in rr]))
        summaries.append(summary);comparison=dict(comparator=name)
        for k in ['time','mse','first_time','first_mse','last_mse']:
            values=[None if lookup[s,p['primary']][k] is None or lookup[s,name][k] is None else lookup[s,p['primary']][k]-lookup[s,name][k] for s in p['seeds']]
            comparison[k]=None if any(v is None for v in values) else dict(mean=float(np.mean(values)),ci95=ci(values),negative_tasks=int(np.sum(np.array(values)<0)))
        paired.append(comparison)
        for n in p['stages']:
            ss=[r for r in rows if r['method']==name and r['n']==n];complete=len(ss)==len(p['seeds'])*p['repetitions']
            stages.append(dict(method=name,n=n,complete=complete,mse=float(np.mean([r['query_mse'] for r in ss])) if complete else None,
                time=float(np.mean([r['total_seconds'] for r in ss])) if complete else None))

    gate_contrasts=[];components=[]
    for cfg in p['configs']:
        name=cfg['name'];rr=[r for r in rows if r['method']==name and r['n']==4]
        parts=[]
        for row in rr:
            calls=row['screen_calls'];gates=[c['credit']['primal_gate'] for c in calls if 'primal_gate' in c['credit']]
            parts.append(dict(credit_seconds=sum(c['credit']['seconds'] for c in calls),
                gate_seconds=sum(g['seconds'] for g in gates),
                gate_construction_seconds=gates[0]['construction_seconds'] if gates else 0.,
                gate_skipped=sum(g['skipped'] for g in gates),credit_proofs=sum(c['credit_rejected'] for c in calls),
                gate_peak_cached_numeric_bytes=max([g['cached_upper_numeric_bytes']+g['upper_output_numeric_bytes'] for g in gates],default=0)))
        components.append(dict(method=name,**{k:float(np.mean([r[k] for r in parts])) for k in parts[0]}))
        if not cfg.get('primal_upper_gate'):continue
        comparison=dict(method=name,comparator=cfg['ungated_comparator'])
        for k in ['time','first_time','mse','first_mse']:
            values=[lookup[s,name][k]-lookup[s,cfg['ungated_comparator']][k] for s in p['seeds']]
            comparison[k]=dict(mean=float(np.mean(values)),ci95=ci(values),negative_tasks=int(np.sum(np.array(values)<0)))
        gate_contrasts.append(comparison)

    result=dict(gate_contrasts=gate_contrasts,components=components,complete=True,all_methods_completed=not failures,checks=checks,max_prediction_error=max_error,summaries=summaries,
        primary_comparisons=paired,failed_episodes=failures,source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','rows.json','episodes.json','run_audit.json']},
        scope='16 reused development tasks; complete charged common recovery; unadjusted descriptive task-paired intervals')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',result),('tasks.json',tasks),('stages.json',stages)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    names=[p['primary'],'alm_c5','alm_residual_full_c5_interval_endpoints','adam60_native_full_c5_interval_endpoints',
        'pc_native_full_c5_interval_endpoints','nodual_native_full_c5_interval_endpoints','direct4096_c5','prior4096_ridge','rbf_loocv','residual_linear_ridge']
    labels=['ALM gated','ALM no credit','ALM residual','Adam joint','PC','No dual','Direct prior4096','Fixed-feature ridge','RBF','Linear residual ridge']
    ss={r['method']:r for r in summaries};fig,axes=plt.subplots(1,2,figsize=(12.8,5.5),layout='constrained');colors=plt.get_cmap('tab10').colors
    for i,(name,label) in enumerate(zip(names,labels)):
        row=ss[name]
        if row['time'] is None:continue
        axes[0].scatter(row['time'],row['mse'],color=colors[i],s=58,label=label)
        rr=[r for r in stages if r['method']==name and r['complete']]
        axes[1].plot([r['n'] for r in rr],[r['mse'] for r in rr],color=colors[i],marker='o',label=label)
    axes[0].set(xlabel='Charged full-stream write + read (s, log scale)',ylabel='Mean unseen-query MSE',xscale='log',title='All tasks and shared recovery costs')
    axes[1].set(xlabel='Observed support count',ylabel='Unseen-query MSE (log scale)',yscale='log',title='Complete-task sample efficiency',xticks=p['stages'])
    for ax in axes:ax.grid(alpha=.22)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Primal gate: 24 methods, 16 reused development tasks; not confirmation',fontsize=12)
    fig.savefig(out/'online_primal_gate.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(complete=True,checks=checks,output=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

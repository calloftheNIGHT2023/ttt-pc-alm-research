"""Independent replay and complete-cell reporting for cold support blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_recovered_online_comparison as replay


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    a=np.asarray(values);rng=np.random.default_rng(483027)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/context_block_scaling/development';p=json.loads((inp/'protocol.json').read_text())
    run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    oldaudit=json.loads((root/'results/recovered_online_comparison/analysis/summary.json').read_text())
    assert oldaudit['source_sha256']==sha(Path(replay.__file__))
    rows=json.loads((inp/'rows.json').read_text());assert len(rows)==run['adaptations']==2304
    lookup={(r['seed'],r['method'],r['repetition'],r['n']):r for r in rows};assert len(lookup)==len(rows)
    checks=dict(states=0,posterior_states=0,regression_states=0,support_particles=0,query_risks=0,details=0,
        same_learner_states=0,matched_terminal_states=0,failures=0,recovery_events=0,charged_attempts=0)
    maximum=0.;tasks=[];components=[]
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
        vv=replay.forward(xx,teacher[None])[0]+np.random.default_rng(seed+19000000).uniform(-.001,.001,24)
        q=np.linspace(0,1,p['query_points']);truth=replay.forward(q,teacher[None])[0]
        for cfg in p['configs']:
            for n in p['block_sizes']:
                rr=[]
                for rep in range(p['repetitions']):
                    row=lookup[seed,cfg['name'],rep,n];rr.append(row);assert row['charged_seconds']>=0
                    if cfg['family']!='regression':
                        base=lookup[seed,cfg['learner']+'_c5',rep,n]
                        assert base['complete']==row['complete'];checks['matched_terminal_states']+=1
                    if not row['complete']:
                        assert 'state_file' not in row and 'query_mse' not in row
                        # Only the explicitly bounded search exhaustion is an expected algorithm failure.
                        assert row['recovery_attempts'] is not None,row['traceback']
                        events=row['recovery_attempts'];assert len(events)==3 and all(not e['success'] for e in events)
                        assert [e['features'] for e in events[1:]]==p['recovery_feature_budgets']
                        assert sum(e['seconds'] for e in events)<=row['charged_seconds']+1e-6
                        checks['charged_attempts']+=len(events);checks['failures']+=1;continue
                    path=inp/row['state_file'];assert sha(path)==row['state_sha256'];checks['states']+=1
                    assert abs(row['write_seconds']+row['read_seconds']-row['charged_seconds'])<1e-12
                    with np.load(path) as z:
                        assert np.array_equal(z['x'],xx[:n]) and np.array_equal(z['v'],vv[:n]) and np.array_equal(z['q'],q)
                        if cfg['family']!='regression':
                            points=z['points'];assert len(points)==2048 and np.array_equal(z['anchor'],points[0])
                            assert np.max(abs(points))<=.12+1e-12
                            assert np.max(abs(replay.forward(xx[:n],points)-vv[:n]))<=.001+1e-8
                            digest=hashlib.sha256(z['anchor'].tobytes()+points.tobytes()).hexdigest()
                            assert row['state_digest']==digest and row['previous_state_digest'] is None
                            curve=np.zeros(len(q))
                            for k in range(0,len(points),193):curve+=replay.forward(q,points[::-1][k:k+193]).sum(0)/len(points)
                            rec=row['recovery'];events=rec['attempts'];assert events[-1]['success'] and all(not e['success'] for e in events[:-1])
                            assert sum(e['seconds'] for e in events)<=row['write_seconds']+1e-6
                            if rec['triggered']:
                                assert rec['all_observations_reused']==n and [e['features'] for e in events[1:]]==p['recovery_feature_budgets'][:len(events)-1]
                                checks['recovery_events']+=1
                            else:assert len(events)==1
                            checks['charged_attempts']+=len(events);checks['posterior_states']+=1;checks['support_particles']+=len(points)
                            with np.load(inp/base['state_file']) as b:assert all(z[k].tobytes()==b[k].tobytes() for k in z.files)
                            assert row['positive_mode_keys']==base['positive_mode_keys'];checks['same_learner_states']+=1
                        else:
                            curve=replay.regression_replay(xx[:n],vv[:n],q,cfg['name'],z,row);checks['regression_states']+=1
                        error=float(np.max(abs(curve-z['prediction'])));assert error<1e-8;maximum=max(maximum,error)
                        risk=float(np.trapezoid((curve-truth)**2,x=q));assert abs(risk-row['query_mse'])<1e-9;checks['query_risks']+=1
                    if 'detail_file' in row:
                        path=inp/row['detail_file'];assert sha(path)==row['detail_sha256'];detail=json.loads(path.read_text());checks['details']+=1
                        assert detail['positive_mode_keys']==row['positive_mode_keys']
                        calls=detail.get('screen_calls',[])
                        components.append(dict(seed=seed,method=cfg['name'],n=n,recovered=row['recovery']['triggered'],
                            geometry_calls=row['geometry_calls'],credit_proofs=len(detail.get('credit_proofs',[])),
                            credit_seconds=sum(c.get('credit',{}).get('seconds',0.) for c in calls),
                            contract_rejections=sum(c.get('contract_rejected',0) for c in calls),
                            input_patterns=sum(c.get('input_cells',0) for c in calls),
                            bank_bytes=detail.get('factorized',{}).get('bank_bytes',0)))
                complete=all(r['complete'] for r in rr)
                tasks.append(dict(seed=seed,method=cfg['name'],n=n,complete=complete,
                    charged_seconds=float(np.mean([r['charged_seconds'] for r in rr])),
                    query_mse=float(np.mean([r['query_mse'] for r in rr])) if complete else None,
                    state_bytes=max(r.get('state_bytes',0) for r in rr),
                    recoveries=sum(bool(r.get('recovery') and r['recovery']['triggered']) for r in rr)))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    assert checks['failures']==run['failures'] and checks['states']==run['successes']
    table=[];paired=[];tasklookup={(r['seed'],r['method'],r['n']):r for r in tasks}
    for n in p['block_sizes']:
        for cfg in p['configs']:
            name=cfg['name'];rr=[tasklookup[s,name,n] for s in p['seeds']];complete=all(r['complete'] for r in rr)
            cc=[r for r in components if r['method']==name and r['n']==n]
            table.append(dict(method=name,n=n,complete_tasks=sum(r['complete'] for r in rr),
                charged_seconds=float(np.mean([r['charged_seconds'] for r in rr])),query_mse=float(np.mean([r['query_mse'] for r in rr])) if complete else None,
                recoveries=sum(r['recoveries'] for r in rr),max_persistent_state_bytes=max(r['state_bytes'] for r in rr),
                component_available_tasks=len(cc),component_geometry_mean=float(np.mean([r['geometry_calls'] for r in cc])) if cc else None,
                component_credit_proofs_mean=float(np.mean([r['credit_proofs'] for r in cc])) if cc else None,
                component_credit_seconds_mean=float(np.mean([r['credit_seconds'] for r in cc])) if cc else None))
            aa=[tasklookup[s,p['primary'],n] for s in p['seeds']]
            delta=np.array([a['charged_seconds']-b['charged_seconds'] for a,b in zip(aa,rr)])
            risk=None
            if all(a['complete'] and b['complete'] for a,b in zip(aa,rr)):
                d=np.array([a['query_mse']-b['query_mse'] for a,b in zip(aa,rr)])
                risk=dict(mean=float(d.mean()),ci95=ci(d))
            paired.append(dict(n=n,comparator=name,same_success_pattern=all(a['complete']==b['complete'] for a,b in zip(aa,rr)),
                time=dict(mean=float(delta.mean()),ci95=ci(delta)),query_mse=risk,
                warning='Timing includes failed work; speed dominance requires comparable output quality and completion'))
    aggregate=[]
    for cfg in p['configs']:
        rr=[r for r in tasks if r['method']==cfg['name']]
        aggregate.append(dict(method=cfg['name'],complete_task_blocks=sum(r['complete'] for r in rr),task_blocks=len(rr),
            equal_cell_charged_seconds=float(np.mean([r['charged_seconds'] for r in rr])),
            equal_cell_query_mse=float(np.mean([r['query_mse'] for r in rr])) if all(r['complete'] for r in rr) else None))
    result=dict(passed=True,checks=checks,max_prediction_replay_error=maximum,by_block=table,paired=paired,aggregate=aggregate,
        source_sha256=sha(Path(__file__)),regression_replay_source_sha256=sha(Path(replay.__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','rows.json','run_audit.json']},scope=p['scope'])
    out=inp.parent/'analysis';out.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',result),('tasks.json',tasks),('components.json',components)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    names=[p['primary'],'alm_c5','alm_residual_full_c5_interval_endpoints','adam60_native_full_c5_interval_endpoints',
        'pc_native_full_c5_interval_endpoints','nodual_native_full_c5_interval_endpoints','direct4096_c5','prior4096_ridge']
    labels=['ALM joint','ALM no credit','ALM residual','Adam joint','PC joint','No dual','Direct4096','Fixed-feature ridge']
    fig,axes=plt.subplots(1,3,figsize=(15,5.3),layout='constrained')
    for name,label in zip(names,labels):
        rr=[r for r in table if r['method']==name];xx=[r['n'] for r in rr]
        axes[0].plot(xx,[r['charged_seconds'] for r in rr],marker='o',label=label)
        axes[1].plot(xx,[r['complete_tasks']/16 for r in rr],marker='o',label=label)
        axes[2].plot(xx,[r['query_mse'] if r['query_mse'] is not None else np.nan for r in rr],marker='o',label=label)
    axes[0].set(ylabel='Charged write + read seconds',title='Includes all failed search costs')
    axes[1].set(ylabel='Completed tasks / 16',title='No successful-subset quality averages',ylim=(-.03,1.03))
    axes[2].set(ylabel='Unseen-query MSE',yscale='log',title='Gaps mean at least one failed task')
    for ax in axes:ax.set(xlabel='Cold initial support block',xticks=p['block_sizes']);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Same 16 development tasks, state reset for every block; not an online stream')
    fig.savefig(out/'context_block_scaling.png',dpi=165);plt.close(fig)
    print(json.dumps(dict(passed=True,checks=checks,output=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

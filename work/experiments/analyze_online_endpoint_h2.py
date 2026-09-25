"""Independent readout/state audit and task-level matched time contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(x,points):
    h=np.broadcast_to(x,(len(points),len(x)))
    for j in range(4):h=np.maximum(0.,1.-np.abs(2.*(h+points[:,j,None])-1.))
    return h


def ci(values):
    a=np.array(values);rng=np.random.default_rng(482039)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/online_endpoint_h2/development';out=root/'results/online_endpoint_h2/analysis'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text())
    failures=json.loads((inp/'failures.json').read_text());assert len(failures)==run['failures']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    lookup={(r['seed'],r['method'],r['repetition'],r['n']):r for r in rows}
    eps={(e['seed'],e['method'],e['repetition']):e for e in episodes};assert len(lookup)==len(rows) and len(eps)==len(episodes)
    checks=dict(state_hashes=0,prediction_replays=0,risk_replays=0,support_particles=0,own_state_links=0,
        first_write_same_discovery=0,same_frontier_input_hashes=0);tasks=[];max_error=0.
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
        vv=forward(xx,teacher[None])[0]+np.random.default_rng(seed+19000000).uniform(-.001,.001,24)
        q=np.linspace(0,1,257);truth=forward(q,teacher[None])[0]
        for cfg in p['configs']:
            name=cfg['name'];trial_metrics=[]
            for rep in range(p['repetitions']):
                e=eps[seed,name,rep];previous=None;rr=[]
                for n in p['stages'][:e['stages']]:
                    row=lookup[seed,name,rep,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256'];checks['state_hashes']+=1
                    with np.load(path) as z:
                        assert np.array_equal(q,z['q']) and np.array_equal(xx[:n],z['x']) and np.array_equal(vv[:n],z['v'])
                        prediction=z['prediction']
                        if cfg['family']!='regression':
                            points=z['points'];anchor=z['anchor'];assert np.array_equal(anchor,points[0]) and len(points)==2048
                            digest=hashlib.sha256(anchor.tobytes()+points.tobytes()).hexdigest()
                            assert digest==row['state_digest'] and previous==row['previous_state_digest'];previous=digest
                            if n>4:checks['own_state_links']+=1
                            assert np.max(abs(points))<=.12+1e-12 and np.max(abs(forward(xx[:n],points)-vv[:n]))<=.001+1e-8
                            checks['support_particles']+=len(points);replay=np.zeros(len(q))
                            for start in range(0,len(points),193):
                                pp=points[::-1][start:start+193]
                                for first in range(0,len(q),67):replay[first:first+67]+=forward(q[first:first+67],pp).sum(0)/len(points)
                            base_row=lookup[seed,cfg['learner']+'_c5',rep,n]
                            with np.load(inp/base_row['state_file']) as ref:
                                assert all(np.array_equal(z[k],ref[k]) for k in z.files)
                            if n==4:
                                assert row['discovery_bank_sha256']==base_row['discovery_bank_sha256']
                                assert row['positive_mode_keys']==base_row['positive_mode_keys'] and row['completion_proposals']==base_row['completion_proposals']
                                assert [c['input_pattern_hash'] for c in row['screen_calls']]==[c['input_pattern_hash'] for c in base_row['screen_calls']]
                                checks['first_write_same_discovery']+=1;checks['same_frontier_input_hashes']+=len(row['screen_calls'])
                        else:
                            bank=np.random.default_rng(731).uniform(-.12,.12,(4096,4));features=forward(xx[:n],bank);mu=features.mean(0);centered=features-mu
                            alpha=np.linalg.solve(centered.T@centered/4096+np.eye(n)*.001**2/3,vv[:n]-mu)
                            replay=((1+centered@alpha)/4096)@forward(q,bank)
                        error=float(np.max(abs(replay-prediction)));assert error<1e-10;max_error=max(max_error,error)
                        risk=float(np.trapezoid((replay-truth)**2,x=q));assert abs(risk-row['query_mse'])<1e-12
                        checks['prediction_replays']+=1;checks['risk_replays']+=1
                    if 'detail_file' in row:assert sha(inp/row['detail_file'])==row['detail_sha256']
                    rr.append(row)
                assert len(rr)>0
                if not e['complete']:assert e['failed_n']==p['stages'][len(rr)]
                first=rr[0];trial_metrics.append(dict(complete=e['complete'],first_time=first['total_seconds'],first_mse=first['query_mse'],
                    time=sum(r['total_seconds'] for r in rr) if e['complete'] else None,
                    mse=float(np.mean([r['query_mse'] for r in rr])) if e['complete'] else None,
                    first_geometry_calls=first['geometry_calls'],credit_rejected=sum(c['credit_rejected'] for c in first['screen_calls']),
                    contract_rejected=sum(c['contract_rejected'] for c in first['screen_calls']),
                    screen_seconds=sum(c['contraction_seconds']+c['credit']['seconds'] for c in first['screen_calls']),
                    credit_seconds=sum(c['credit']['seconds'] for c in first['screen_calls']),
                    collection_seconds=first['credit_snapshot'].get('collection_seconds',0.),
                    selection_seconds=first['credit_snapshot'].get('selection_seconds',0.),
                    directions=first['credit_snapshot'].get('selected_rows',0)))
            fields=list(trial_metrics[0]);complete=all(r['complete'] for r in trial_metrics)
            record=dict(seed=seed,method=name,complete=complete)
            for key in fields:
                if key=='complete':continue
                record[key]=None if any(r[key] is None for r in trial_metrics) else float(np.mean([r[key] for r in trial_metrics]))
            tasks.append(record)
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    lookup={(r['seed'],r['method']):r for r in tasks};summaries=[];paired=[]
    numeric=[k for k in fields if k!='complete']
    for cfg in p['configs']:
        name=cfg['name'];rr=[r for r in tasks if r['method']==name];record=dict(method=name,complete_tasks=sum(r['complete'] for r in rr))
        for key in numeric:record[key]=None if any(r[key] is None for r in rr) else float(np.mean([r[key] for r in rr]))
        summaries.append(record)
        if cfg['family']=='regression':continue
        c20=cfg['learner']+'_c20';delta={}
        for key in ['first_time','time']:
            values=[None if lookup[s,name][key] is None or lookup[s,c20][key] is None else lookup[s,name][key]-lookup[s,c20][key] for s in p['seeds']]
            delta[key]=None if any(v is None for v in values) else dict(mean=float(np.mean(values)),ci95=ci(values),faster_tasks=int(np.sum(np.array(values)<0)))
        paired.append(dict(method=name,comparator=c20,delta=delta))
    paired_c5=[]
    for cfg in p['configs']:
        if cfg['family']=='regression':continue
        name=cfg['name'];target=cfg['learner']+'_c5';delta={}
        for key in ['first_time','time']:
            values=[None if lookup[s,name][key] is None or lookup[s,target][key] is None else lookup[s,name][key]-lookup[s,target][key] for s in p['seeds']]
            delta[key]=None if any(v is None for v in values) else dict(mean=float(np.mean(values)),ci95=ci(values),faster_tasks=int(np.sum(np.array(values)<0)))
        paired_c5.append(dict(method=name,comparator=target,delta=delta))
    comparisons=[]
    for target in ['alm_c5','alm_c20','alm_native_full_c5','alm_native_full_c5_interval','alm_native_full_c20_interval_endpoints','alm_residual_full_c5_interval_endpoints','alm_residual_full_c5_interval','direct4096_c5','direct4096_c20','adam60_native_full_c5_interval_endpoints','pc_native_full_c5_interval_endpoints','nodual_native_full_c5_interval_endpoints','prior4096_ridge']:
        result=dict(comparator=target)
        for key in ['first_time','time','mse']:
            v=[None if lookup[s,target][key] is None else lookup[s,'alm_native_full_c5_interval_endpoints'][key]-lookup[s,target][key] for s in p['seeds']]
            result[key]=None if any(value is None for value in v) else dict(mean=float(np.mean(v)),ci95=ci(v))
        comparisons.append(result)
    verifier_pairs=[];components=[]
    for cfg in p['configs']:
        if cfg.get('interval_verifier') and cfg.get('interval_comparator',cfg['rational_comparator']) in {c['name'] for c in p['configs']}:
            name=cfg['name'];target=cfg.get('interval_comparator',cfg['rational_comparator']);delta={}
            for key in ['first_time','time']:
                values=[None if lookup[s,name][key] is None or lookup[s,target][key] is None else lookup[s,name][key]-lookup[s,target][key] for s in p['seeds']]
                delta[key]=None if any(v is None for v in values) else dict(mean=float(np.mean(values)),ci95=ci(values))
            verifier_pairs.append(dict(method=name,comparator=target,delta=delta))
        if not cfg.get('factorized_full'):continue
        rr=[r for r in rows if r['method']==cfg['name'] and r['n']==4]
        record=dict(method=cfg['name'])
        for key in ['float_seconds','interval_seconds','exact_seconds','seconds']:
            record[key]=float(np.mean([sum(c['credit'].get(key,0.) for c in row['screen_calls']) for row in rr]))
        record['other_seconds']=record['seconds']-record['float_seconds']-record['interval_seconds']-record['exact_seconds']
        assert record['other_seconds']>=-1e-8
        components.append(record)
    result=dict(complete=True,checks=checks,max_prediction_error=max_error,summaries=summaries,paired_vs_own_c20=paired,
        implementation_pairs=verifier_pairs,credit_components=components,
        paired_vs_own_c5=paired_c5,alm_endpoint_comparisons=comparisons,failed_episodes=failures,source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(inp/'protocol.json'),rows_sha256=sha(inp/'rows.json'),
        scope='16 previously used development tasks; exact-output contrasts within learner, descriptive paired intervals without multiplicity adjustment')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('tasks.json',tasks)]: (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2));colors={'c5':'#888888','native_full_c5':'#d47728','native_full_c5_interval':'#7293bb','native_full_c5_interval_endpoints':'#348c70'}
    ss={r['method']:r for r in summaries};learners=['alm','adam60','pc','nodual','direct4096']
    for j,mode in enumerate(['c5','native_full_c5','native_full_c5_interval','native_full_c5_interval_endpoints']):
        xs=[];ys=[]
        for i,learner in enumerate(learners):
            name=learner+'_'+mode
            if name not in ss:continue
            xs.append(i+(j-2)*.14);ys.append(ss[name]['first_time'])
        axes[0].bar(xs,ys,.13,label=mode,color=colors[mode])
    axes[0].set(xticks=range(5),xticklabels=learners,ylabel='First write + read (s)',title='Same predictions, charged implementations');axes[0].legend(fontsize=8)
    names=['alm_c5','alm_native_full_c5_interval','alm_native_full_c5_interval_endpoints','alm_residual_full_c5_interval_endpoints','direct4096_c5','prior4096_ridge']
    for i,name in enumerate(names):axes[1].scatter(ss[name]['time'],ss[name]['mse'],s=70,label=name)
    axes[1].set(xlabel='Full four-stage write + read (s)',ylabel='Unseen-query MSE',title='Complete-task quality and cost');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Endpoint + interval H2: 16 development tasks, two repeats; no confirmation claim');fig.tight_layout()
    fig.savefig(out/'online_endpoint_h2.png',dpi=170);plt.close(fig)
    cc={r['method']:r for r in components};fig,ax=plt.subplots(figsize=(12.4,5.0),layout='constrained')
    names=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native']
    labels=['ALM joint','ALM residual','Adam joint','Adam residual','PC','No dual']
    fields=['float_seconds','interval_seconds','exact_seconds','other_seconds'];colors=['#287c8e','#52a688','#d47728','#aaaaaa']
    for side,suffix in [(-1,''),(0,'_interval'),(1,'_interval_endpoints')]:
        xs=np.arange(len(names))+side*.25;bottom=np.zeros(len(names))
        for field,color in zip(fields,colors):
            values=np.array([cc[name+'_full_c5'+suffix][field]*1000 for name in names])
            ax.bar(xs,values,.24,bottom=bottom,color=color,label=field.replace('_seconds','') if side==-1 else None)
            bottom+=values
    ax.set_xticks(np.arange(len(names)),labels);ax.set_ylabel('Actual causal credit screening (ms)')
    ax.set_title('Per group: rational / interval / endpoint + interval; C5 own trajectories')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'online_endpoint_components.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(complete=True,checks=checks,output=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

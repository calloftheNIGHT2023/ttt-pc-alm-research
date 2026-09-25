"""Independently replay all full-stream states and paired implementation costs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deferred_typed_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(delta):
    idx=np.random.default_rng(698021).integers(0,len(delta),(20000,len(delta)))
    return np.quantile(delta[idx].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    inp=root/'deferred_typed_memory/development';out=root/'deferred_typed_memory/analysis';oldroot=root/'typed_routing_memory'
    protocol=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['complete'] and run['rows']==4352
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(Path(__file__).with_name('deferred_typed_development.json'))==protocol['config_sha256']
    assert sha(Path(__file__).with_name(protocol['base_config']))==protocol['base_config_sha256']
    rows=json.loads((inp/'episodes.json').read_text());checkpoint=[json.loads(line) for line in (inp/'episodes.jsonl').read_text().splitlines()]
    assert rows==checkpoint
    lookup={(r['repetition'],r['seed'],r['method'],r['implementation'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)==4352
    old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'development/episodes.json').read_text())}
    ce_reference=json.loads((oldroot/'analysis/summary.json').read_text());assert ce_reference['audit']['states_and_queries']==1280
    assert sha(Path(__file__).with_name('analyze_typed_routing_memory.py'))==ce_reference['analysis_source_sha256']
    ce={r['method']:r for r in ce_reference['summary']};configs=protocol['configs'];seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']))
    states=0;queries=0;query_rows=0;proofs=0;queues=0;later=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        replay_cache={}
        for cfg in configs:
            method=cfg['name'];implementation=cfg['routing_implementation']
            for rep in range(protocol['repetitions']):
                for n in protocol['stages']:
                    row=lookup[rep,seed,method,implementation,n];reference=old[seed,method,n]
                    path=inp/row['state_file'];rp=oldroot/'development'/reference['state_file'];assert sha(path)==row['state_sha256'] and sha(rp)==row['original_state_sha256']==reference['state_sha256']
                    with np.load(path) as z:anchor=z['anchor'];samples=z['samples']
                    with np.load(rp) as z:assert np.array_equal(anchor,z['anchor']) and np.array_equal(samples,z['samples']);states+=1
                    if (method,n) not in replay_cache:
                        pred=model.old.old.old.old.posterior.make_predict(samples);answer=pred(q)
                        replay_cache[method,n]=(float(np.mean((answer-target)**2)),float(np.mean((np.clip(answer,0,1)-target)**2)),float(np.max(np.abs(pred(x[:n])-v[:n]))))
                        queries+=1
                    assert replay_cache[method,n]==(row['raw_query_mse'],row['query_mse'],row['support_max_error']);query_rows+=1
                    if n==4 and cfg.get('route_mode','none')!='none':
                        for key in ['route_original_pattern_keys','route_extra_all_keys','route_extra_pattern_keys','route_extra_origins']:assert row[key]==reference[key]
                        queues+=1
                        for now,before in zip(row['feedback_details'],reference['feedback_details']):
                            assert now['proofs']==before['proofs'];proofs+=1
                            if implementation=='deferred':
                                assert now['deferred_route_flushed'] and now['deferred_max_c20_batch']<=256
                                assert now['deferred_c20_rows']==now['route_stats']['distinct_modes']-now['route_stats']['causal_clause_removed']
                    if n!=4:assert row['effective_generator']=='direct' and not row['routing_active'];later+=1
        print(json.dumps(dict(seed=seed,states=states,queries=queries)),flush=True)
    time_by_task={};mse_by_task={};summary=[]
    for cfg in configs:
        key=cfg['name'],cfg['routing_implementation'];method,implementation=key
        times=np.array([[[lookup[r,s,*key,n]['adaptation_seconds']+lookup[r,s,*key,n]['read_queries_seconds'] for n in protocol['stages']] for r in range(protocol['repetitions'])] for s in seeds])
        quality=np.array([[lookup[0,s,*key,n]['query_mse'] for n in protocol['stages']] for s in seeds])
        time_by_task[key]=times.sum(2).mean(1);mse_by_task[key]=quality.mean(1)
        first=[lookup[r,s,*key,4] for s in seeds for r in range(protocol['repetitions'])]
        details=[d for row in first for d in row.get('feedback_details',[]) if 'route_counts' in d]
        summary.append(dict(method=method,implementation=implementation,mean_query_mse=float(quality.mean()),mean_conditional_excess=ce[method]['mean_conditional_excess'],
            mean_full_seconds=float(times.sum(2).mean()),mean_first_seconds=float(times[:,:,0].mean()),task_time_means=time_by_task[key].tolist(),
            mean_first_geometry_calls=float(np.mean([z.get('geometry_calls',z.get('initial_geometry_calls',0)+z.get('completion_geometry_calls',0)) for z in first])),
            mean_route_seconds=sum(d['route_stats']['routing_seconds'] for d in details)/len(first),
            mean_construction_seconds=sum(d['route_counts']['construction_seconds'] for d in details)/len(first),
            max_old_plus_new_state_bytes=max(z['previous_state_bytes']+z['persistent_state_bytes'] for z in rows if (z['method'],z['implementation'])==key),
            mean_deferred_c20_calls=sum(d.get('deferred_c20_calls',0) for d in details)/len(first),
            max_deferred_pending_keys=max([d.get('deferred_peak_pending_keys',0) for d in details],default=0)))
    implementation_pairs=[]
    for cfg in configs:
        if cfg['routing_implementation']!='deferred':continue
        method=cfg['name'];a=time_by_task[method,'deferred'];b=time_by_task[method,'eager'];delta=a-b
        implementation_pairs.append(dict(method=method,mean_full_seconds_delta=float(delta.mean()),descriptive_ci95=ci(delta),
            ratio_of_mean_full_times=float(a.mean()/b.mean()),faster_tasks=int((delta<0).sum()),
            quality_bitwise_unchanged=True,task_deltas=delta.tolist()))
    active=[]
    for cfg in configs:
        if cfg['routing_implementation']=='deferred' or cfg.get('route_mode','none')=='none':
            active.append(next(z for z in summary if (z['method'],z['implementation'])==(cfg['name'],cfg['routing_implementation'])))
    assert len(active)==20
    primary=protocol['primary_candidate'],protocol['primary_implementation'];cross=[]
    for comparator in active:
        key=comparator['method'],comparator['implementation']
        if key==primary:continue
        dq=mse_by_task[primary]-mse_by_task[key];dt=time_by_task[primary]-time_by_task[key]
        cross.append(dict(comparator=key[0],implementation=key[1],mse_delta=float(dq.mean()),descriptive_mse_ci95=ci(dq),
            full_seconds_delta=float(dt.mean()),descriptive_time_ci95=ci(dt),
            conditional_excess_delta=ce[primary[0]]['mean_conditional_excess']-ce[key[0]]['mean_conditional_excess']))
    result=dict(analysis_source_sha256=sha(Path(__file__)),input_sha256={k:sha(inp/k) for k in ['protocol.json','run_audit.json','episodes.json','episodes.jsonl']},
        conditional_reference_sha256=sha(oldroot/'analysis/summary.json'),
        audit=dict(source_hashes=len(protocol['source_sha256']),states_bitwise=states,distinct_method_state_query_replays=queries,query_rows_checked=query_rows,queue_and_origin_replays=queues,proof_sequence_replays=proofs,later_direct_checks=later),
        summary=summary,active_methods=active,implementation_pairs=implementation_pairs,primary_cross_method_pairs=cross,
        scope='2 timing repeats averaged within each of 16 old tasks; descriptive task bootstrap, no multiplicity correction; CE inherited only after bitwise state proof')
    assert states==query_rows==4352 and queries==1280 and queues==proofs==896 and later==3264
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    names=[r['method'] for r in implementation_pairs];y=np.arange(len(names));fig,ax=plt.subplots(figsize=(11,8))
    for offset,kind,color in [(-.18,'eager','#8290a5'),(.18,'deferred','#177e89')]:
        values=[next(r['mean_full_seconds'] for r in summary if r['method']==name and r['implementation']==kind) for name in names]
        ax.barh(y+offset,values,.34,label=kind,color=color)
        for i,value in enumerate(values):ax.text(value+.01,i+offset,f'{value:.3f}',va='center',fontsize=8)
    ax.set_yticks(y,names,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Complete stream adaptation + query reading, seconds');ax.set_title('Same numerical output; full cost of eager versus deferred routing');ax.legend();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'deferred_full_timing.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,10),sharey=True);y=np.arange(len(active))
    for ax,metric,title in zip(axes,['mean_query_mse','mean_full_seconds','mean_conditional_excess'],['Actual full-stream query MSE','Complete stream seconds','First-write conditional excess']):
        values=[r[metric] for r in active];ax.barh(y,values,color=['#177e89' if r['method']==primary[0] else '#8290a5' for r in active]);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,[r['method'] for r in active],fontsize=8);axes[0].invert_yaxis();axes[2].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    fig.suptitle('All enhanced controls share deferred routing; quality unchanged');fig.tight_layout();fig.savefig(out/'deferred_strong_controls.png',dpi=170);plt.close(fig)
    fa=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),figures={name:sha(out/name) for name in ['deferred_full_timing.png','deferred_strong_controls.png']})
    (out/'figure_audit.json').write_text(json.dumps(fa,indent=2),encoding='utf-8');print(json.dumps(dict(audit=result['audit'],implementation_pairs=implementation_pairs,active_methods=active,cross=cross),indent=2),flush=True)


if __name__=='__main__':main()

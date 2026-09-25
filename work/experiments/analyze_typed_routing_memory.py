"""Full state/query/proof and conditional-risk audit of typed online routing."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import typed_routing_memory as model
import neighbor_mode_memory as neighbor


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(delta):
    indices=np.random.default_rng(621591).integers(0,len(delta),(20000,len(delta)))
    return np.quantile(delta[indices].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    inp=root/'typed_routing_memory/development';out=root/'typed_routing_memory/analysis';oldroot=root/'conflict_feedback_memory/development'
    protocol=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['complete'] and run['rows']==1280
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(Path(__file__).with_name('typed_routing_development.json'))==protocol['config_sha256']
    rows=json.loads((inp/'episodes.json').read_text());cfgs={c['name']:c for c in protocol['configs']};lookup={(r['seed'],r['method'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)==1280
    seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));stages=protocol['stages']
    old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'episodes.json').read_text()) if r['repetition']==0}
    old_ref=0;state_checks=0;proof_checks=0;lp_checks=0;conditional=[];samples={};primary_matches=[];safe_checks=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        cr=root/'posterior_state_reuse/conditional_risk';audit=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/audit['curve_file'];assert sha(cp)==audit['curve_sha256']
        with np.load(cp) as z:grid=z['q'];weights=z['weights'];means=z['region_means'];patterns=z['patterns'];full=np.einsum('k,rkq->rq',weights,means)
        positive=set(patterns);positive_array=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,4) for k in patterns]);seen=set()
        for name,cfg in cfgs.items():
            for n in stages:
                row=lookup[seed,name,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256']
                with np.load(path) as z:state=z['anchor'].copy(),z['samples'].copy()
                samples[seed,name,n]=state;pred=model.old.old.old.old.posterior.make_predict(state[1])(q)
                assert float(np.mean((pred-target)**2))==row['raw_query_mse'] and float(np.mean((np.clip(pred,0,1)-target)**2))==row['query_mse'];state_checks+=1
                if 'reference_method' in cfg:
                    reference=old[seed,cfg['reference_method'],n];rp=oldroot/reference['state_file'];assert sha(rp)==reference['state_sha256']
                    with np.load(rp) as z:assert np.array_equal(state[0],z['anchor']) and np.array_equal(state[1],z['samples']);old_ref+=1
                if n!=4:assert not row['routing_active'] and row['effective_generator']=='direct'
            first=lookup[seed,name,4];assert set(first['positive_mode_keys'])<=positive
            for obs in first.get('feedback_details',[]):
                if cfg.get('route_mode','none')!='none':
                    stats=obs['route_stats'];assert stats['distinct_modes']==stats['causal_clause_removed']+stats['c20_removed']+obs['route_mode_survivors']
                    assert obs['route_counts']['inputs']==obs['repairs_triggered']
                if name==protocol['primary_candidate']:assert obs['kind']=='local' and not first['global_bp_credit_used'] and not first['global_bp_parameter_update']
                for proof in obs['proofs']:
                    reg=np.frombuffer(bytes.fromhex(proof['parent']),np.uint8).reshape(4,4);a=np.array(proof['a']);p=np.array(proof['p'])
                    exact=model.conflict.extract(x,v,reg,p,a);assert exact is not None and all(proof[k]==value for k,value in exact.items())
                    assert model.feedback.causal.normal.exact_optimum(x,v,reg,a)==proof['optimized_exact'];assert not proof['c20_detects_parent'];proof_checks+=1
                    assert 0<proof['accepted_event']<=obs['callbacks'] and not model.conflict.clause_mask(positive_array,[proof]).any();safe_checks+=len(positive)
                    if proof['parent'] not in seen:
                        _,_,matrix,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4)
                        assert lp.status==2;seen.add(proof['parent']);lp_checks+=1
            actual=model.old.old.old.old.posterior.make_predict(samples[seed,name,4][1])(grid)
            excess=float(np.mean([np.trapezoid((actual-full[a])*(actual-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            mask=np.array([p in set(first['positive_mode_keys']) for p in patterns]);mass=float(weights[mask].sum());trunc=None
            if mass:
                ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);trunc=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            source=cfg.get('admission_method',cfg.get('feedback_reference',cfg.get('reference_method','alm_feedback_full')));baseline=set(old[seed,source,4]['positive_mode_keys'])
            conditional.append(dict(seed=seed,method=name,conditional_excess=excess,ideal_truncation=trunc,mass=mass,baseline=source,
                gained_modes=sorted(set(first['positive_mode_keys'])-baseline),lost_modes=sorted(baseline-set(first['positive_mode_keys']))))
        for n in stages:
            a=samples[seed,protocol['primary_candidate'],n];b=samples[seed,'alm_feedback_full',n]
            primary_matches.append(dict(seed=seed,n_context=n,anchor_equal=bool(np.array_equal(a[0],b[0])),samples_equal=bool(np.array_equal(a[1],b[1]))))
        print(json.dumps(dict(seed=seed,states=state_checks,proofs=proof_checks,lp_modes=lp_checks)),flush=True)
    quality={};costs={};firsttimes={};ce={};summary=[]
    for name in cfgs:
        quality[name]=np.array([np.mean([lookup[s,name,n]['query_mse'] for n in stages]) for s in seeds])
        times=np.array([[lookup[s,name,n]['adaptation_seconds']+lookup[s,name,n]['read_queries_seconds'] for n in stages] for s in seeds]);costs[name]=times.sum(1);firsttimes[name]=times[:,0]
        first=[lookup[s,name,4] for s in seeds];cc=[r for r in conditional if r['method']==name];ce[name]=np.array([next(r['conditional_excess'] for r in cc if r['seed']==s) for s in seeds])
        details=[d for r in first for d in r.get('feedback_details',[]) if 'route_counts' in d]
        summary.append(dict(method=name,mean_query_mse=float(quality[name].mean()),mean_full_seconds=float(costs[name].mean()),mean_first_seconds=float(firsttimes[name].mean()),
            mean_first_geometry_calls=float(np.mean([r.get('geometry_calls',r.get('initial_geometry_calls',0)+r.get('completion_geometry_calls',0)) for r in first])),
            mean_conditional_excess=float(ce[name].mean()),mean_mass=float(np.mean([r['mass'] for r in cc])),mean_ideal_truncation=None if any(r['ideal_truncation'] is None for r in cc) else float(np.mean([r['ideal_truncation'] for r in cc])),
            gained_modes=sum(len(r['gained_modes']) for r in cc),lost_modes=sum(len(r['lost_modes']) for r in cc),feasible_streams=sum(all(lookup[s,name,n]['support_feasible'] for n in stages) for s in seeds),
            maximum_old_plus_new_state_bytes=max(r['previous_state_bytes']+r['persistent_state_bytes'] for r in rows if r['method']==name),
            route_totals={k:sum(d['route_counts'][k] for d in details) for k in ['inputs','candidates','segments']},
            route_mean_seconds={k:sum(d['route_counts'][k] for d in details)/len(seeds) for k in ['construction_seconds','pattern_seconds','trigger_seconds']},
            route_mean_screen_seconds=sum(d['route_stats']['routing_seconds'] for d in details)/len(seeds),
            extra_geometries=sum(r.get('route_extra_geometry_calls',0) for r in first),max_route_numeric_key_bytes=max([d['route_key_numeric_bytes'] for d in details],default=0)))
    primary=protocol['primary_candidate'];paired=[]
    pairs=[(primary,name) for name in cfgs if name!=primary]+[('alm_tied_both_full','alm_tied_forward_full'),('alm_tied_forward_k24',primary),('adam16_coordinate_full','adam16_tied_full')]
    for a,b in pairs:
        dq=quality[a]-quality[b];dt=costs[a]-costs[b];dc=ce[a]-ce[b]
        paired.append(dict(candidate=a,comparator=b,primary=a==primary and b in protocol['primary_comparators'],mse_delta=float(dq.mean()),descriptive_mse_ci95=ci(dq),
            better_tasks=int((dq<-1e-15).sum()),equal_tasks=int((np.abs(dq)<=1e-15).sum()),worse_tasks=int((dq>1e-15).sum()),mean_full_time_delta=float(dt.mean()),descriptive_time_ci95=ci(dt),
            conditional_excess_delta=float(dc.mean()),descriptive_conditional_excess_ci95=ci(dc),mean_first_time_delta=float((firsttimes[a]-firsttimes[b]).mean())))
    assert state_checks==1280 and old_ref==384
    result=dict(analysis_source_sha256=sha(Path(__file__)),audit=dict(source_hashes=len(protocol['source_sha256']),states_and_queries=state_checks,old_reference_states=old_ref,
        exact_proofs=proof_checks,independent_lp_modes=lp_checks,complete_positive_clause_checks=safe_checks,own_state_later_direct=True),
        summary=summary,paired=paired,primary_state_comparisons=primary_matches,scope='16 old development streams; single interleaved timing repeat; confidence intervals descriptive, not fresh confirmation; no multiple-comparison correction')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'conditional_risks.json').write_text(json.dumps(conditional,indent=2),encoding='utf-8')
    y=np.arange(len(summary));fig,axes=plt.subplots(1,3,figsize=(17,10),sharey=True)
    for ax,key,label in zip(axes,['mean_query_mse','mean_full_seconds','mean_conditional_excess'],['Actual full-stream query MSE','Complete fit + read seconds (1 repeat)','First-write conditional excess']):
        values=np.array([r[key] for r in summary]);ax.barh(y,values,color=['#177e89' if r['method']==primary else '#c18c40' if 'coordinate' in r['method'] else '#8290a5' for r in summary]);ax.set_xlim(0,max(values)*1.20);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.text(value+max(values)*.01,i,f'{value:.5f}' if key!='mean_full_seconds' else f'{value:.2f}',va='center',fontsize=7)
        if key=='mean_conditional_excess':ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0].set_yticks(y,[r['method'] for r in summary],fontsize=8);axes[0].invert_yaxis();fig.suptitle('Typed routing: actual online predictive memory with matched strong controls');fig.tight_layout();fig.savefig(out/'typed_routing_online.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

"""Complete physical-cut stream evaluation and exact proof audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import physical_cut_memory as model
import neighbor_mode_memory as neighbor


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(delta):
    ix=np.random.default_rng(914215).integers(0,len(delta),(20000,len(delta)));return np.quantile(delta[ix].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results';inp=root/'physical_cut_memory/development';out=root/'physical_cut_memory/analysis';oldroot=root/'conflict_feedback_memory/development'
    protocol=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'episodes.json').read_text());cfgs={c['name']:c for c in protocol['configs']}
    seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));stages=protocol['stages'];assert len(rows)==len(cfgs)*len(seeds)*len(stages)
    for name,h in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==h,name
    assert sha(Path(__file__).with_name('physical_cut_development.json'))==protocol['config_sha256']
    lookup={(r['seed'],r['method'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'episodes.json').read_text()) if r['repetition']==0};cache={};oldcache={}
    reference_checks=0;proof_checks=0;lp_checks=0;conditional=[];primary_state_comparisons=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        cr=root/'posterior_state_reuse/conditional_risk';audit=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/audit['curve_file'];assert sha(cp)==audit['curve_sha256']
        with np.load(cp) as z:grid=z['q'];weights=z['weights'];means=z['region_means'];patterns=z['patterns'];positive=set(patterns);full=np.einsum('k,rkq->rq',weights,means)
        checked=set();positive_array=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,4) for k in patterns])
        for name,cfg in cfgs.items():
            for n in stages:
                row=lookup[seed,name,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256']
                with np.load(path) as z:state=z['anchor'].copy(),z['samples'].copy()
                cache[seed,name,n]=state;pred=model.old.old.old.old.posterior.make_predict(state[1])(q)
                assert float(np.mean((pred-target)**2))==row['raw_query_mse'] and float(np.mean((np.clip(pred,0,1)-target)**2))==row['query_mse']
                if n!=4:assert row['effective_generator']=='direct' and not row['physical_active']
                if 'reference_method' in cfg:
                    key=seed,cfg['reference_method'],n;ref=old[key];rp=oldroot/ref['state_file'];assert sha(rp)==ref['state_sha256']
                    if key not in oldcache:
                        with np.load(rp) as z:oldcache[key]=z['anchor'].copy(),z['samples'].copy()
                    assert np.array_equal(state[0],oldcache[key][0]) and np.array_equal(state[1],oldcache[key][1]);reference_checks+=1
                if n==4:assert set(row['positive_mode_keys'])<=positive
            first=lookup[seed,name,4]
            for obs in first.get('feedback_details',[]):
                if name=='alm_physical_full':assert obs['kind']=='local' and not first['global_bp_credit_used'] and not first['global_bp_parameter_update']
                assert obs['physical_activity']+obs['physical_bias']==obs['physical_accepted']<=obs['physical_triggered']
                for proof in obs['proofs']:
                    reg=np.frombuffer(bytes.fromhex(proof['parent']),np.uint8).reshape(4,4);pp=np.array(proof['p']);aa=np.array(proof['a'])
                    exact=model.conflict.extract(x,v,reg,pp,aa);assert exact is not None and all(proof[k]==val for k,val in exact.items())
                    assert model.feedback.causal.normal.exact_optimum(x,v,reg,aa)==proof['optimized_exact'];assert not proof['c20_detects_parent'];assert 0<proof['accepted_event']<=obs['callbacks'];proof_checks+=1
                    if proof['parent'] not in checked:
                        _,_,matrix,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2;checked.add(proof['parent']);lp_checks+=1
                    assert not model.conflict.clause_mask(positive_array,[proof]).any()
            actual=model.old.old.old.old.posterior.make_predict(cache[seed,name,4][1])(grid)
            excess=float(np.mean([np.trapezoid((actual-full[a])*(actual-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            mask=np.array([p in set(first['positive_mode_keys']) for p in patterns]);mass=float(weights[mask].sum());trunc=None
            if mass:
                ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);trunc=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            baseline=set(lookup[seed,'alm_retained_full' if name.startswith('alm') else 'adam16_retained_full',4]['positive_mode_keys'])
            conditional.append(dict(seed=seed,method=name,conditional_excess=excess,ideal_truncation=trunc,mass=mass,
                                    gained_modes=sorted(set(first['positive_mode_keys'])-baseline),lost_modes=sorted(baseline-set(first['positive_mode_keys']))))
        for n in stages:
            a=cache[seed,'alm_physical_full',n];b=cache[seed,'alm_retained_full',n]
            primary_state_comparisons.append(dict(seed=seed,n_context=n,anchor_equal=bool(np.array_equal(a[0],b[0])),samples_equal=bool(np.array_equal(a[1],b[1]))))
        print(json.dumps(dict(seed=seed,states=len(cache),proofs=proof_checks,lp_modes=lp_checks)),flush=True)
    summary=[];quality={};costs={};firsttimes={}
    for name in cfgs:
        quality[name]=np.array([np.mean([lookup[s,name,n]['query_mse'] for n in stages]) for s in seeds])
        times=np.array([[lookup[s,name,n]['adaptation_seconds']+lookup[s,name,n]['read_queries_seconds'] for n in stages] for s in seeds]);costs[name]=times.sum(1);firsttimes[name]=times[:,0]
        first=[lookup[s,name,4] for s in seeds];cc=[r for r in conditional if r['method']==name];details=[d for r in first for d in r.get('feedback_details',[])]
        summary.append(dict(method=name,mean_query_mse=float(quality[name].mean()),mean_full_seconds=float(costs[name].mean()),mean_first_seconds=float(times[:,0].mean()),
                            mean_geometry_calls=float(np.mean([r.get('geometry_calls',r.get('initial_geometry_calls',0)+r.get('completion_geometry_calls',0)) for r in first])),
                            mean_conditional_excess=float(np.mean([r['conditional_excess'] for r in cc])),mean_mass=float(np.mean([r['mass'] for r in cc])),
                            mean_ideal_truncation=None if any(r['ideal_truncation'] is None for r in cc) else float(np.mean([r['ideal_truncation'] for r in cc])),
                            gained_modes=sum(len(r['gained_modes']) for r in cc),lost_modes=sum(len(r['lost_modes']) for r in cc),
                            feasible_streams=sum(all(lookup[s,name,n]['support_feasible'] for n in stages) for s in seeds),
                            maximum_old_plus_new_state_bytes=max(r['previous_state_bytes']+r['persistent_state_bytes'] for r in rows if r['method']==name),
                            feedback_totals={k:sum(d[k] for d in details) for k in ['clauses','physical_triggered','physical_accepted','physical_candidates','physical_segments','physical_empty_segments','physical_unrepresentable_segments','physical_no_point','physical_exact_cut_checks','physical_activity','physical_bias']},
                            max_library_numeric_bytes=max([d['library_numeric_bytes'] for d in details],default=0),
                            mean_feedback_seconds={k:float(sum(d[k] for d in details)/len(seeds)) for k in ['physical_seconds','feedback_seconds','screen_seconds','exact_learning_seconds']}))
    candidate=protocol['primary_candidate'];paired=[]
    for a,b in [(candidate,n) for n in cfgs if n!=candidate]+[('alm_watch_full','alm_retained_full')]:
        dq=quality[a]-quality[b];dt=costs[a]-costs[b]
        paired.append(dict(candidate=a,comparator=b,primary=a==candidate and b in protocol['primary_comparators'],mse_delta=float(dq.mean()),descriptive_mse_ci95=ci(dq),
                           better_tasks=int((dq<-1e-15).sum()),equal_tasks=int((np.abs(dq)<=1e-15).sum()),worse_tasks=int((dq>1e-15).sum()),
                           mean_full_time_delta=float(dt.mean()),descriptive_time_ci95=ci(dt),mean_first_time_delta=float((firsttimes[a]-firsttimes[b]).mean())))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes=len(rows),unique_queries_replayed=len(cache),old_reference_states_bitwise=reference_checks,
                           exact_dynamic_proofs=proof_checks,independent_lp_modes=lp_checks,later_own_state_direct=True,
                           primary_vs_original_anchor_equal=sum(r['anchor_equal'] for r in primary_state_comparisons),primary_vs_original_samples_equal=sum(r['samples_equal'] for r in primary_state_comparisons)),
                summary=summary,paired=paired,primary_state_comparisons=primary_state_comparisons,analysis_source_sha256=sha(Path(__file__)),
                scope='16 old streams; single timing repetition descriptive only; query answers evaluator-only; full cost includes Fraction; conditional risk numerical')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'conditional_risks.json').write_text(json.dumps(conditional,indent=2),encoding='utf-8')
    y=np.arange(len(summary));fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True)
    for ax,key,label in zip(axes,['mean_query_mse','mean_full_seconds','mean_conditional_excess'],['Actual full-stream query MSE','Full fit + query seconds (1 repeat)','First-write conditional excess']):
        vals=np.array([r[key] for r in summary]);ax.barh(y,vals,color=['#177E89' if r['method'].startswith('alm') else '#D79434' if r['method'].startswith('adam') else '#748299' for r in summary]);ax.set_xlim(0,max(vals)*1.22);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(4))
        if key=='mean_conditional_excess':ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        for i,val in enumerate(vals):ax.text(val+max(vals)*.01,i,f'{val:.5f}' if key!='mean_full_seconds' else f'{val:.2f}',va='center',fontsize=7)
    axes[0].set_yticks(y,[r['method'] for r in summary],fontsize=8);axes[0].invert_yaxis();fig.suptitle('Physical residual cuts: actual causal trajectories and complete costs');fig.tight_layout();fig.savefig(out/'physical_cut_online.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

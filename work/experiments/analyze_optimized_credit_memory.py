"""Complete online audit and paired costs for analytically optimized credits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import optimized_credit_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(values):
    ids=np.random.default_rng(192823).integers(0,len(values),(20000,len(values)))
    return np.quantile(values[ids].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results'
    inp=root/'optimized_credit_memory/development';out=root/'optimized_credit_memory/analysis';oldroot=root/'budgeted_credit_memory/development'
    protocol=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'episodes.json').read_text());cfgs={c['name']:c for c in protocol['configs']};seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));stages=protocol['stages']
    assert len(rows)==len(cfgs)*len(seeds)*len(stages)*protocol['repetitions']
    for name,h in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==h,name
    assert sha(Path(__file__).with_name('optimized_credit_development.json'))==protocol['config_sha256']
    lookup={(r['repetition'],r['seed'],r['method'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'episodes.json').read_text()) if r['repetition']==0}
    diag={(r['seed'],r['method']):r for r in json.loads((root/'optimized_branch_dual/diagnostic/rows.json').read_text()) if r['bank']=='native'}
    cache={};oldcache={};reference_checks=0;prefix_checks=0;conditional=[];uncapped_matches=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
        cr=root/'posterior_state_reuse/conditional_risk';ca=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as curve:
            grid=curve['q'];weights=curve['weights'];means=curve['region_means'];patterns=curve['patterns'];positive=set(patterns);full=np.einsum('k,rkq->rq',weights,means)
        for name,cfg in cfgs.items():
            for rep in range(protocol['repetitions']):
                for n in stages:
                    row=lookup[rep,seed,name,n];path=inp/row['state_file'];assert sha(path)==row['state_sha256']
                    with np.load(path) as z:state=z['anchor'].copy(),z['samples'].copy()
                    key=seed,name,n
                    if key not in cache:
                        cache[key]=state;pred=model.old.old.posterior.make_predict(state[1])(q)
                        assert float(np.mean((pred-target)**2))==row['raw_query_mse'] and float(np.mean((np.clip(pred,0,1)-target)**2))==row['query_mse']
                    else:assert np.array_equal(state[0],cache[key][0]) and np.array_equal(state[1],cache[key][1])
                    if n!=4:assert row['effective_generator']=='direct' and not row['optimized_normal']
                    if 'reference_method' in cfg:
                        oldkey=seed,cfg['reference_method'],n;prior=old[oldkey]
                        if oldkey not in oldcache:
                            op=oldroot/prior['state_file'];assert sha(op)==prior['state_sha256']
                            with np.load(op) as z:oldcache[oldkey]=z['anchor'].copy(),z['samples'].copy()
                        assert np.array_equal(state[0],oldcache[oldkey][0]) and np.array_equal(state[1],oldcache[oldkey][1]);reference_checks+=1
                    if n==4:
                        assert set(row['positive_mode_keys'])<=positive
                        if 'diagnostic_reference' in cfg:
                            expected=diag[seed,cfg['diagnostic_reference']]['after_keys'];cap=cfg.get('geometry_budget')
                            assert row['evaluated_pattern_keys']==(expected if cap is None else expected[:cap]);prefix_checks+=1
                            assert set(row['positive_mode_keys'])==set(row['evaluated_pattern_keys'])&positive
            row=lookup[0,seed,name,4];actual=model.old.old.posterior.make_predict(cache[seed,name,4][1])(grid)
            ce=float(np.mean([np.trapezoid((actual-full[a])*(actual-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            mask=np.array([p in set(row['positive_mode_keys']) for p in patterns]);mass=float(weights[mask].sum());trunc=None
            if mass:
                ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);trunc=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
            conditional.append(dict(seed=seed,method=name,conditional_excess=ce,mass=mass,ideal_truncation=trunc))
        for n in stages:
            baseline=cache[seed,'alm_local_full',n]
            for name in ['alm_none_full','alm_opt_full','alm_residual_opt_full']:
                state=cache[seed,name,n];assert np.array_equal(state[0],baseline[0]) and np.array_equal(state[1],baseline[1]);uncapped_matches+=1
        print(json.dumps(dict(seed=seed,states=len(cache))),flush=True)
    summary=[];quality={};times={};fulltime={}
    for name in cfgs:
        quality[name]=np.array([np.mean([lookup[0,s,name,n]['query_mse'] for n in stages]) for s in seeds])
        tt=np.array([[[lookup[rep,s,name,n]['adaptation_seconds']+lookup[rep,s,name,n]['read_queries_seconds'] for n in stages] for s in seeds] for rep in range(protocol['repetitions'])]);times[name]=tt;fulltime[name]=tt.sum(2).mean(0)
        first=[lookup[rep,s,name,4] for rep in range(protocol['repetitions']) for s in seeds];cc=[r for r in conditional if r['method']==name]
        summary.append(dict(method=name,mean_query_mse=float(quality[name].mean()),mean_full_seconds=float(fulltime[name].mean()),mean_first_seconds=float(tt[:,:,0].mean()),
            mean_geometry_calls=float(np.mean([r.get('geometry_calls',r.get('initial_geometry_calls',0)+r.get('completion_geometry_calls',0)) for r in first])),
            mean_normal_seconds=float(np.mean([r.get('normal_cost',{}).get('seconds',0) for r in first])),mean_conditional_excess=float(np.mean([r['conditional_excess'] for r in cc])),
            mean_mass=float(np.mean([r['mass'] for r in cc])),mean_ideal_truncation=None if any(r['ideal_truncation'] is None for r in cc) else float(np.mean([r['ideal_truncation'] for r in cc])),
            feasible_streams=sum(all(lookup[0,s,name,n]['support_feasible'] for n in stages) for s in seeds),maximum_old_plus_new_state_bytes=max(r['previous_state_bytes']+r['persistent_state_bytes'] for r in rows if r['method']==name)))
    candidate=protocol['primary_candidate'];pairs=[(candidate,name) for name in cfgs if name!=candidate]+[tuple(p) for p in protocol['primary_increment_pairs']]+[('adam16_opt_full','adam16_bp_full'),('alm_opt_k24','adam16_opt_k24'),('alm_opt_k24','adam60_opt_k24')];paired=[]
    for a,b in pairs:
        dq=quality[a]-quality[b];dt=fulltime[a]-fulltime[b];df=(times[a][:,:,0]-times[b][:,:,0]).mean(0)
        paired.append(dict(candidate=a,comparator=b,primary=(a==candidate and b in protocol['primary_comparators']) or [a,b] in protocol['primary_increment_pairs'],
            mse_delta=float(dq.mean()),descriptive_mse_ci95=ci(dq),better_tasks=int((dq<-1e-15).sum()),equal_tasks=int((np.abs(dq)<=1e-15).sum()),
            mean_full_time_delta=float(dt.mean()),descriptive_time_ci95=ci(dt),mean_first_time_delta=float(df.mean()),descriptive_first_time_ci95=ci(df)))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes=len(rows),unique_states_replayed=len(cache),reference_states_bitwise=reference_checks,diagnostic_prefix_matches=prefix_checks,uncapped_alm_state_equivalences=uncapped_matches,later_stages_own_state_direct=True),
        summary=summary,paired=paired,analysis_source_sha256=sha(Path(__file__)),scope='16 old tasks; repeated timings not additional quality samples; numerical conditional-risk reference')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'conditional_risks.json').write_text(json.dumps(conditional,indent=2),encoding='utf-8')
    y=np.arange(len(summary));fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True)
    for ax,key,label in zip(axes,['mean_query_mse','mean_full_seconds','mean_conditional_excess'],['Actual full-stream query MSE','Mean full fit + query time (s)','First-write conditional excess']):
        vv=np.array([r[key] for r in summary]);ax.barh(y,vv,color=['#167e79' if r['method'].startswith('alm') else '#d79434' if r['method'].startswith('adam') else '#7c8d95' for r in summary]);ax.set_xlim(0,max(vv)*1.25);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(4))
        if key=='mean_conditional_excess':ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        for i,v in enumerate(vv):ax.text(v+max(vv)*.015,i,f'{v:.5f}' if key!='mean_full_seconds' else f'{v:.3f}',va='center',fontsize=7)
    axes[0].set_yticks(y,[r['method'] for r in summary],fontsize=8);axes[0].invert_yaxis();fig.suptitle('Analytic branch-normal credit: actual online state and complete costs');fig.tight_layout();fig.savefig(out/'optimized_credit_results.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

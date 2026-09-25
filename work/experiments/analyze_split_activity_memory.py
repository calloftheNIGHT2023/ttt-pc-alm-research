"""Saved-state replay, prior diagnostic prediction check, and all comparisons."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import split_activity_mode_memory as model


def ci(d):
    indices=np.random.default_rng(172751).integers(0,len(d),(20000,len(d)))
    return np.quantile(d[indices].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);a=parser.parse_args()
    root=a.project/'results/split_activity_modes';inp=root/'development';out=root/'analysis';refroot=a.project/'results/posterior_state_reuse'
    protocol=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'episodes.json').read_text())
    assert len(rows)==832;lookup={(r['seed'],r['method'],r['n_context']):r for r in rows};assert len(lookup)==832
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256(Path(__file__).with_name('split_activity_development.json').read_bytes()).hexdigest()==protocol['config_sha256']
    diagnostic={(r['seed'],r['method']):r for r in json.loads((root/'diagnostic/audits.json').read_text())}
    oldroot=a.project/'results/neighbor_mode_completion/development';oldrows=json.loads((oldroot/'episodes.json').read_text())
    oldlookup={(r['seed'],r['method'],r['n_context']):r for r in oldrows}
    oldnames=dict(alm16_forward='alm16_h0',adam16_forward='adam16_h0',alm16_h1='alm16_h1',
                  adam16_h1='adam16_h1',adam8_h2='adam8_h2',adam60_h2='adam60_h2',direct4096_h2='direct4096_h2')
    seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));conditional=[];replayed=0;exact_diagnostic=0;exact_baseline=0;maxdiff=0.
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        audit=json.loads((refroot/f'conditional_risk/audit_{seed}.json').read_text());path=refroot/'conditional_risk'/audit['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['curve_sha256']
        with np.load(path) as curve:
            keys=list(curve['patterns']);weights=curve['weights'];means=curve['region_means'];grid=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
            for cfg in protocol['configs']:
                name=cfg['name']
                for n in protocol['stages']:
                    r=lookup[seed,name,n];path=inp/r['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['state_sha256']
                    with np.load(path) as state:
                        assert np.array_equal(state['anchor'],r['anchor_output'])
                        predict=model.posterior.make_predict(state['samples']);mse=float(np.mean((predict(q)-target)**2));gap=abs(mse-r['raw_query_mse'])
                        maxdiff=max(maxdiff,gap);assert gap<1e-15
                        if name in oldnames:
                            oldr=oldlookup[seed,oldnames[name],n];oldpath=oldroot/oldr['state_file']
                            assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==oldr['state_sha256']
                            with np.load(oldpath) as saved:assert np.array_equal(state['samples'],saved['samples']) and np.array_equal(state['anchor'],saved['anchor'])
                            exact_baseline+=1
                        if n==4:actual=predict(grid)
                    replayed+=1
                    if n!=4:assert not r['split_proposals_active'] and r['effective_generator']=='direct'
                r=lookup[seed,name,4];found=set(r['positive_mode_keys']);assert found<=set(keys)
                if cfg.get('split_proposals') and cfg['generator']!='adam':
                    audit=diagnostic[seed,name.replace('_split','')];path=root/'diagnostic'/audit['trace_file']
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['trace_sha256']
                    trace=json.loads(path.read_text());expected=(set(trace['forward'])|set(trace['split']))&set(keys)
                    assert found==expected,(seed,name);exact_diagnostic+=1
                mask=np.array([k in found for k in keys]);w=weights*mask;mass=w.sum();w/=mass
                pred=np.einsum('k,rkq->rq',w,means)
                ce=np.mean([np.trapezoid((actual-full[i])*(actual-full[j]),x=grid) for i,j in [(0,1),(2,3)]])
                te=np.mean([np.trapezoid((pred[i]-full[i])*(pred[j]-full[j]),x=grid) for i,j in [(0,1),(2,3)]])
                conditional.append(dict(seed=seed,method=name,conditional_excess=float(ce),ideal_truncation=float(te),mass=float(mass)))
        print(json.dumps(dict(replayed_seed=seed,states=replayed)),flush=True)
    summary=[];risk={};cost={}
    for cfg in protocol['configs']:
        name=cfg['name'];rr=[[lookup[s,name,n] for n in protocol['stages']] for s in seeds]
        mse=np.array([[r['query_mse'] for r in row] for row in rr]);times=np.array([[r['adaptation_seconds']+r['read_queries_seconds'] for r in row] for row in rr])
        risk[name]=mse.mean(1);cost[name]=times.sum(1);cr=[r for r in conditional if r['method']==name]
        summary.append(dict(method=name,mean_query_mse=float(mse.mean()),stage_query_mse=mse.mean(0).tolist(),
            median_full_seconds=float(np.median(times.sum(1))),median_first_seconds=float(np.median(times[:,0])),
            mean_conditional_excess=float(np.mean([r['conditional_excess'] for r in cr])),mean_ideal_truncation=float(np.mean([r['ideal_truncation'] for r in cr])),
            mean_mass=float(np.mean([r['mass'] for r in cr])),
            mean_extra_split_proposals=float(np.mean([lookup[s,name,4].get('additional_split_patterns',0) for s in seeds])),
            max_old_plus_new_state_bytes=max(r['persistent_state_bytes']+r['previous_state_bytes'] for row in rr for r in row),
            feasible_streams=int(np.all([[r['support_feasible'] for r in row] for row in rr],axis=1).sum())))
    candidate=protocol['primary_candidate'];paired=[]
    for name in risk:
        if name==candidate:continue
        delta=risk[candidate]-risk[name];td=cost[candidate]-cost[name]
        paired.append(dict(comparator=name,primary=name in protocol['primary_comparators'],mse_delta=float(delta.mean()),ci95=ci(delta),
            better_streams=int((delta<-1e-15).sum()),equal_streams=int((abs(delta)<=1e-15).sum()),
            mean_time_delta=float(td.mean()),time_delta_ci95=ci(td)))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),states_replayed=replayed,max_query_replay_difference=maxdiff,
        baseline_states_exact=exact_baseline,diagnostic_positive_sets_exact=exact_diagnostic,later_stages_direct_only=True),summary=summary,paired=paired,
        scope='16 old development streams; descriptive intervals; conditional metrics use numerical full-mode reference')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',result),('conditional_risks.json',conditional)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5.4));names=[r['method'] for r in summary];pos=np.arange(len(names))
    colors=['#137f79' if n.startswith('alm') else '#d69532' if n.startswith('adam') else '#71808c' for n in names]
    axes[0].barh(pos,[r['mean_ideal_truncation'] for r in summary],color=colors)
    axes[0].set_xscale('symlog',linthresh=1e-7);axes[0].set_yticks(pos,names,fontsize=8);axes[0].invert_yaxis()
    axes[0].set_xlabel('First-write ideal missing-mode risk (symlog)')
    for r,color in zip(summary,colors):
        axes[1].scatter(r['median_full_seconds'],r['mean_query_mse'],color=color)
        axes[1].annotate(r['method'],(r['median_full_seconds'],r['mean_query_mse']),xytext=(3,4),textcoords='offset points',fontsize=6)
    axes[1].set_xlabel('Median full stream fit + query reads (s)');axes[1].set_ylabel('Mean unseen-query MSE')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Split-activity proposals: actual online validation, 16 old streams');fig.tight_layout()
    fig.savefig(out/'split_activity_results.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

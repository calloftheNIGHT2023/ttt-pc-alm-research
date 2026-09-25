"""Full saved-state replay, graph-prediction audit, and matched result analysis."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neighbor_mode_memory as model
from diagnose_missing_mode_graph import distances


def ci(delta):
    rng=np.random.default_rng(91173);indices=rng.integers(0,len(delta),(20000,len(delta)))
    return np.quantile(delta[indices].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'episodes.json').read_text());configs=protocol['configs'];names=[c['name'] for c in configs]
    assert len(rows)==len(configs)*protocol['count']*len(protocol['stages'])==1600
    lookup={(r['method'],r['seed'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    references={r['seed']:r for r in json.loads((a.reference_root/'first_write_reference/coverage.json').read_text())}
    seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));conditional=[];graphs=[];replay_max=0.;states_checked=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        ref=references[seed];path=a.reference_root/'first_write_reference'/ref['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['reference_sha256']
        data=json.loads(path.read_text());assert data['numerical_volume_reference_complete'];assert np.array_equal(x[:4],data['x']) and np.array_equal(v[:4],data['v'])
        cells=sorted(data['positive_regions'],key=lambda r:r['pattern']);keys=[c['pattern'] for c in cells];patterns=np.array([np.frombuffer(bytes.fromhex(k),np.uint8) for k in keys])
        adjacent=np.abs(patterns[:,None].astype(int)-patterns[None,:].astype(int)).sum(-1)==1
        initial_cache={}
        audit=json.loads((a.reference_root/f'conditional_risk/audit_{seed}.json').read_text());curvepath=a.reference_root/'conditional_risk'/audit['curve_file'];assert hashlib.sha256(curvepath.read_bytes()).hexdigest()==audit['curve_sha256']
        with np.load(curvepath) as curves:
            qgrid=curves['q'];weights=curves['weights'];means=curves['region_means'];second=curves['region_second_moments'];full=np.einsum('k,rkq->rq',weights,means)
            assert np.array_equal(curves['patterns'],np.array(keys))
            for c in configs:
                for n in protocol['stages']:
                    row=lookup[c['name'],seed,n];path=a.input/row['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['state_sha256']
                    with np.load(path) as state:
                        assert np.array_equal(state['anchor'],row['anchor_output']);assert state['anchor'].nbytes+state['samples'].nbytes==row['persistent_state_bytes']
                        predict=model.posterior.make_predict(state['samples']);mse=float(np.mean((predict(q)-target)**2));gap=abs(mse-row['raw_query_mse']);replay_max=max(replay_max,gap);assert gap<1e-15
                        assert abs(float(np.max(np.abs(predict(x[:n])-v[:n])))-row['support_max_error'])<1e-15
                        if n==4:actual=predict(qgrid)
                    states_checked+=1
                    if n!=4:
                        assert row['effective_generator']=='direct' and not row['archive_active'] and row['completion_proposals']==0
                row=lookup[c['name'],seed,4];cachekey=json.dumps({k:v for k,v in c.items() if k not in ['name','completion_rounds']},sort_keys=True)
                if cachekey not in initial_cache:
                    bank,_=model.discover(x[:4],v[:4],None,dict(**c,archive=True,pool='posterior_mix'))
                    found={r.tobytes().hex() for r in model.previous.interface.archived.signatures(x[:4],bank)}
                    initial_cache[cachekey]=np.array([k in found for k in keys])
                found=initial_cache[cachekey];assert int(found.sum())==row['initial_positive_regions']
                steps=distances(adjacent,found);expected={k for k,s in zip(keys,steps) if 0<=s<=c['completion_rounds']};actual_keys=set(row['positive_mode_keys'])
                assert actual_keys<=set(keys)
                if not row['proposal_budget_truncated']:assert expected==actual_keys,(seed,c['name'],expected-actual_keys,actual_keys-expected)
                mask=np.array([k in actual_keys for k in keys]);ww=weights*mask;mass=ww.sum();ww/=mass
                graphs.append(dict(seed=seed,method=c['name'],predicted_positive_modes=len(expected),actual_positive_modes=len(actual_keys),
                    exact_graph_prediction=expected==actual_keys,budget_truncated=row['proposal_budget_truncated'],mass_fraction=float(mass)))
                pred=np.einsum('k,rkq->rq',ww,means);s2=np.einsum('k,rkq->rq',ww,second);ce=[];te=[];se=[]
                for left,right in [(0,1),(2,3)]:
                    ce.append(float(np.trapezoid((actual-full[left])*(actual-full[right]),x=qgrid)))
                    te.append(float(np.trapezoid((pred[left]-full[left])*(pred[right]-full[right]),x=qgrid)))
                    se.append(float(np.trapezoid(((s2[left]+s2[right])/2-pred[left]*pred[right])/512,x=qgrid)))
                conditional.append(dict(seed=seed,method=c['name'],mass_fraction=float(mass),actual_excess=float(np.mean(ce)),truncation_excess=float(np.mean(te)),expected_sampling_excess=float(np.mean(se))))
        print(json.dumps(dict(replayed_seed=seed,states_checked=states_checked)),flush=True)
    vectors={};costs={};summary=[]
    for name in names:
        rr=[[lookup[name,s,n] for n in protocol['stages']] for s in seeds]
        errors=np.array([[r['query_mse'] for r in row] for row in rr]);times=np.array([[r['adaptation_seconds']+r['read_queries_seconds'] for r in row] for row in rr])
        vectors[name]=errors.mean(1);costs[name]=times.sum(1);cr=[r for r in conditional if r['method']==name]
        summary.append(dict(method=name,mean_query_mse=float(errors.mean()),mean_query_mse_by_stage=errors.mean(0).tolist(),median_full_stream_seconds=float(np.median(times.sum(1))),
            median_first_write_plus_read_seconds=float(np.median(times[:,0])),
            mean_first_conditional_excess=float(np.mean([r['actual_excess'] for r in cr])),mean_first_truncation_excess=float(np.mean([r['truncation_excess'] for r in cr])),
            mean_first_mass_fraction=float(np.mean([r['mass_fraction'] for r in cr])),
            first_budget_truncations=sum(lookup[name,s,4]['proposal_budget_truncated'] for s in seeds),
            mean_first_proposals=float(np.mean([lookup[name,s,4]['completion_proposals'] for s in seeds])),
            mean_first_additional_geometry_calls=float(np.mean([lookup[name,s,4]['completion_geometry_calls'] for s in seeds])),
            maximum_pool_numeric_bytes=max(r['pool_parameter_bytes']+r['pool_signature_bytes'] for row in rr for r in row),
            maximum_old_plus_new_state_bytes=max(r['persistent_state_bytes']+r['previous_state_bytes'] for row in rr for r in row),
            feasible_tasks_by_stage=[sum(lookup[name,s,n]['support_feasible'] for s in seeds) for n in protocol['stages']]))
    candidate=protocol['primary_candidate'];paired=[]
    for name in names:
        if name==candidate:continue
        delta=vectors[candidate]-vectors[name];time_delta=costs[candidate]-costs[name]
        paired.append(dict(candidate=candidate,comparator=name,primary=name in protocol['primary_comparators'],mean_query_mse_difference=float(delta.mean()),
            descriptive_ci95=ci(delta),lower_risk_streams=int((delta<-1e-15).sum()),equal_risk_streams=int((abs(delta)<=1e-15).sum()),
            median_full_time_difference=float(np.median(time_delta)),mean_full_time_difference_ci95=ci(time_delta)))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes=states_checked,maximum_query_replay_difference=replay_max,
        first_write_graph_checks=len(graphs),exact_graph_predictions=sum(r['exact_graph_prediction'] for r in graphs),budget_truncations=sum(r['budget_truncated'] for r in graphs),
        later_stages_direct_only=True),summary=summary,paired=paired,
        scope='old development tasks; descriptive intervals; all costs include actual proposal screening/geometry and independent query reads; no independent confirmation')
    a.out.mkdir(parents=True,exist_ok=True)
    for filename,value in [('summary.json',result),('graph_checks.json',graphs),('conditional_risks.json',conditional)]:
        (a.out/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));groups=[('alm16','#157e78'),('adam8','#e3a038'),('adam16','#c86938'),('adam60','#a44545'),('direct128','#87949f')]
    for family,color in groups:
        rr=[next(r for r in summary if r['method']==f'{family}_h{h}') for h in [0,1,2]]
        axes[0].plot([r['median_full_stream_seconds'] for r in rr],[r['mean_query_mse'] for r in rr],'o-',color=color,label=family)
        axes[1].plot([0,1,2],[r['mean_first_truncation_excess'] for r in rr],'o-',color=color,label=family)
    for name,marker in [('direct4096_h2','s'),('direct65536_h2','*'),('adam240_h2','D')]:
        r=next(r for r in summary if r['method']==name);axes[0].scatter(r['median_full_stream_seconds'],r['mean_query_mse'],marker=marker,s=75,label=name)
    axes[0].set_xlabel('Median full stream fit + 2048-query reads (s)');axes[0].set_ylabel('Mean unseen-query MSE');axes[0].set_yscale('log');axes[0].legend(fontsize=7)
    axes[1].set_xticks([0,1,2]);axes[1].set_xlabel('Actual adjacent-branch completion rounds');axes[1].set_ylabel('First-write ideal conditional truncation risk');axes[1].set_yscale('log');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Implemented mode completion — 25 frozen methods, 16 old streams');fig.tight_layout();fig.savefig(a.out/'neighbor_mode_results.png',dpi=165);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

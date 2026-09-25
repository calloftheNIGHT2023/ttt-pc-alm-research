"""Independent reference arithmetic/state replay and task-paired contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neighbor_mode_memory as geometry


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(x,points):
    h=np.broadcast_to(x,(len(points),len(x))).copy()
    for j in range(4):
        h=2*(h+points[:,j,None]);h=np.maximum(np.minimum(h,2-h),0.)
    return h


def moments(points,q):
    mean=np.zeros(len(q));second=np.zeros(len(q))
    # Different point traversal, reduction blocking and tent expression.
    for start in range(0,len(points),191):
        pp=points[::-1][start:start+191]
        for begin in range(0,len(q),127):
            h=forward(q[begin:begin+127],pp)
            mean[begin:begin+127]+=h.sum(0)/len(points);second[begin:begin+127]+=(h*h).sum(0)/len(points)
    return mean,second


def ci(values):
    a=np.array(values);rng=np.random.default_rng(482819)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_conditional_risk/development';online=root/'results/recovered_online_comparison/development';refroot=root/'results/recovered_support_reference'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(refroot/'exact_faces/coverage.json')==p['reference_coverage_sha256']
    rows=json.loads((inp/'rows.json').read_text());audits=json.loads((inp/'audits.json').read_text());names=p['methods']
    lookup={(r['seed'],r['method']):r for r in rows};assert len(lookup)==80==run['method_tasks']
    coverage={r['seed']:r for r in json.loads((refroot/'exact_faces/coverage.json').read_text())}
    checks=dict(curve_hashes=0,geometry_matches=0,reference_rng_replays=0,reference_samples=0,reference_moment_replays=0,
        frozen_states=0,online_prediction_replays=0,pair_metric_replays=0,aggregate_metric_replays=0)
    maximum=0.;metric_error=0.;reference_gaps=[]
    for audit in audits:
        seed=audit['seed'];path=inp/audit['curve_file'];assert sha(path)==audit['curve_sha256'];checks['curve_hashes']+=1
        reference=refroot/'development'/coverage[seed]['reference_file'];assert sha(reference)==audit['reference_sha256'];ref=json.loads(reference.read_text())
        polys=sorted(ref['positive_regions'],key=lambda r:r['pattern'])
        with np.load(path) as z:
            x=z['x'];v=z['v'];q=z['q'];points=z['reference_points'];mu=z['region_means'];second=z['region_seconds'];w=z['weights']
            assert x.tolist()==ref['x'] and v.tolist()==ref['v'] and np.array_equal(q,np.linspace(0,1,1025))
            assert z['methods'].tolist()==names and z['patterns'].tolist()==[r['pattern'] for r in polys]
            volumes=np.array([r['volume'] for r in polys]);assert np.array_equal(w,volumes/volumes.sum())
            for k,item in enumerate(polys):
                reg=np.frombuffer(bytes.fromhex(item['pattern']),np.uint8).reshape(4,4);_,_,a,rhs=geometry.pattern_matrix(x,v,reg)
                poly,_=geometry.posterior.polytope(a,rhs);assert poly is not None and np.isclose(poly['volume'],item['volume'],rtol=1e-9,atol=1e-30)
                checks['geometry_matches']+=1
                for r in range(4):
                    pp=points[r,k];assert np.max(abs(pp))<=.12+1e-12 and np.max(abs(forward(x,pp)-v))<=.001+1e-8
                    regenerated=geometry.posterior.sample(poly,2048,np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,k,r])))
                    assert pp.tobytes()==regenerated.tobytes();checks['reference_rng_replays']+=1;checks['reference_samples']+=len(pp)
                    mean,s2=moments(pp,q);error=max(float(np.max(abs(mean-mu[r,k]))),float(np.max(abs(s2-second[r,k]))));assert error<1e-12
                    maximum=max(maximum,error);checks['reference_moment_replays']+=1
            actual=z['actual_predictions'];masks=z['method_masks'];frozen={(s['method'],s['repetition']):s for s in audit['frozen_states']}
            for j,name in enumerate(names):
                found=next(c for c in coverage[seed]['coverage'] if c['method']==name)
                assert masks[j].tolist()==[item['pattern'] in set(found['found_patterns']) for item in polys]
                for rep in range(2):
                    item=frozen[name,rep];file=online/item['state_file'];assert sha(file)==item['state_sha256'];checks['frozen_states']+=1
                    with np.load(file) as state:pred=moments(state['points'],q)[0];assert np.max(abs(pred[::4]-state['prediction']))<1e-12
                    error=float(np.max(abs(pred-actual[j,rep])));assert error<1e-12;maximum=max(maximum,error);checks['online_prediction_replays']+=1
            # Reconstruct mixtures with matrix products, not the runner's einsum.
            full=np.stack([w@mu[r] for r in range(4)]);s2=np.stack([w@second[r] for r in range(4)])
            reference_gaps.extend(float(np.trapezoid((full[a]-full[b])**2,x=q)) for a,b in p['all_pairs'])
            for j,name in enumerate(names):
                row=lookup[seed,name];ww=w*masks[j];assert abs(ww.sum()-row['mass_fraction'])<1e-12;ww/=ww.sum()
                mm=np.stack([ww@mu[r] for r in range(4)]);ss=np.stack([ww@second[r] for r in range(4)])
                for item in row['pair_estimates']:
                    a,b=item['pair'];actual_integrand=np.mean([(curve-full[a])*(curve-full[b]) for curve in actual[j]],axis=0)
                    integrands=dict(actual_excess=actual_integrand,truncation_excess=(mm[a]-full[a])*(mm[b]-full[b]),
                        expected_sampling_excess=((ss[a]+ss[b])*.5-mm[a]*mm[b])/2048,
                        bayes_risk=(s2[a]+s2[b])*.5-full[a]*full[b])
                    for kind,step in [('fine',1),('coarse',2)]:
                        vals={key:float(np.trapezoid(value[::step],x=q[::step])) for key,value in integrands.items()}
                        vals['expected_total_excess']=vals['truncation_excess']+vals['expected_sampling_excess']
                        for key,value in vals.items():
                            error=abs(value-item[kind][key]);assert error<1e-12;metric_error=max(metric_error,error);checks['pair_metric_replays']+=1
                    perrep=[float(np.trapezoid((curve-full[a])*(curve-full[b]),x=q)) for curve in actual[j]]
                    assert np.max(abs(np.array(perrep)-item['online_repetition_actual_excess']))<1e-12
                primary=[e for e in row['pair_estimates'] if e['pair'] in p['primary_pairs']]
                for key in row['primary']:
                    assert abs(row['primary'][key]-np.mean([e['fine'][key] for e in primary]))<1e-15
                    assert abs(row['all_pair_mean'][key]-np.mean([e['fine'][key] for e in row['pair_estimates']]))<1e-15
                    assert abs(row['primary_grid_change'][key]-np.mean([e['fine'][key]-e['coarse'][key] for e in primary]))<1e-15
                    checks['aggregate_metric_replays']+=1
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    fields=list(rows[0]['primary']);summaries=[];paired=[]
    for name in names:
        rr=[lookup[s,name] for s in p['seeds']]
        item=dict(method=name,**{key:float(np.mean([r['primary'][key] for r in rr])) for key in fields},
            all_pair_mean={key:float(np.mean([r['all_pair_mean'][key] for r in rr])) for key in fields},
            max_abs_grid_change={key:max(abs(r['primary_grid_change'][key]) for r in rr) for key in fields},
            aggregate_pair_values={key:[float(np.mean([r['pair_estimates'][i]['fine'][key] for r in rr])) for i in range(6)] for key in fields})
        item['conditional_total_risk']=item['bayes_risk']+item['actual_excess'];summaries.append(item)
    for name in [p['primary_comparator'],'adam60_c5','pc_c5','nodual_c5']:
        result=dict(comparator=name)
        for key in ['actual_excess','truncation_excess','expected_sampling_excess','expected_total_excess']:
            delta=np.array([lookup[s,'alm_c5']['primary'][key]-lookup[s,name]['primary'][key] for s in p['seeds']])
            allpair=np.array([lookup[s,'alm_c5']['all_pair_mean'][key]-lookup[s,name]['all_pair_mean'][key] for s in p['seeds']])
            result[key]=dict(mean=float(delta.mean()),ci95=ci(delta),all_pair_mean=float(allpair.mean()),
                lower_tasks=int((delta<-1e-12).sum()),equal_tasks=int((abs(delta)<=1e-12).sum()),higher_tasks=int((delta>1e-12).sum()))
        paired.append(result)
    result=dict(passed=True,checks=checks,max_curve_replay_error=maximum,max_metric_replay_error=metric_error,
        mean_reference_replica_l2_gap=float(np.mean(reference_gaps)),maximum_reference_replica_l2_gap=float(max(reference_gaps)),
        summaries=summaries,paired=paired,source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','rows.json','audits.json','run_audit.json']},scope=p['scope'])
    out=inp.parent/'analysis';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    xx=np.arange(len(names));bias=np.array([r['truncation_excess'] for r in summaries]);var=np.array([r['expected_sampling_excess'] for r in summaries]);ac=np.array([r['actual_excess'] for r in summaries])
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.0),layout='constrained');labels=['ALM','Adam60','PC','No dual','Direct4096']
    axes[0].bar(xx,bias,label='Missing-region mean bias',color='#277d88');axes[0].bar(xx,var,bottom=bias,label='Expected 2048-particle variance',color='#d5a847')
    axes[0].scatter(xx,ac,label='Frozen two-repeat actual risk',color='#333333',zorder=3)
    axes[0].set_xticks(xx,labels);axes[0].set_ylabel('Conditional excess risk over full posterior mean');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    selected=[0,4]
    for side,(key,label,color) in enumerate([('truncation_excess','Missing-region bias','#277d88'),('expected_sampling_excess','Sampling variance','#d5a847'),('actual_excess','Frozen actual','#555555')]):
        values=[summaries[j][key] for j in selected];axes[1].bar(np.arange(2)+(side-1)*.23,values,.22,label=label,color=color)
    axes[1].set_xticks([0,1],['ALM','Direct4096']);axes[1].set(ylabel='Conditional excess risk (zoomed)',title='Predeclared strongest simple comparator');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Current 16 supports: four independent reference replicas; numerical volume and grid',fontsize=12)
    file=out/'conditional_risk.png';fig.savefig(file,dpi=170);plt.close(fig)
    print(json.dumps(dict(passed=True,checks=checks,output=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

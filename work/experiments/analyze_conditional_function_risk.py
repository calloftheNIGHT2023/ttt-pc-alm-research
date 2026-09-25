"""Replay conditional-risk decompositions and report every frozen comparator."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'risks.json').read_text())
    names=[c['name'] for c in protocol['configs']];seeds=protocol['seeds'];assert len(rows)==96
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    lookup={(r['seed'],r['method']):r for r in rows};max_replay=0.;replica_gap=[]
    for seed in seeds:
        audit=json.loads((a.input/f'audit_{seed}.json').read_text());path=a.input/audit['curve_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['curve_sha256']
        with np.load(path) as z:
            q=z['q'];w=z['weights'];mu=z['region_means'];second=z['region_second_moments'];actual=z['actual_predictions'];masks=z['method_masks']
            assert np.array_equal(z['methods'],np.array(names));full=np.einsum('k,rkq->rq',w,mu);fullsecond=np.einsum('k,rkq->rq',w,second)
            replica_gap.extend(float(np.trapezoid((full[left]-full[right])**2,x=q)) for left,right in protocol['pairs'])
            for j,method in enumerate(names):
                ww=w*masks[j];ww/=ww.sum();truncated=np.einsum('k,rkq->rq',ww,mu);s2=np.einsum('k,rkq->rq',ww,second)
                estimates={k:[] for k in ['actual_excess','truncation_excess','expected_sampling_excess','bayes_risk']}
                for left,right in protocol['pairs']:
                    estimates['actual_excess'].append(np.trapezoid((actual[j]-full[left])*(actual[j]-full[right]),x=q))
                    estimates['truncation_excess'].append(np.trapezoid((truncated[left]-full[left])*(truncated[right]-full[right]),x=q))
                    estimates['expected_sampling_excess'].append(np.trapezoid(((s2[left]+s2[right])/2-truncated[left]*truncated[right])/512,x=q))
                    estimates['bayes_risk'].append(np.trapezoid((fullsecond[left]+fullsecond[right])/2-full[left]*full[right],x=q))
                for k,vals in estimates.items():
                    error=abs(float(np.mean(vals))-lookup[seed,method][k]);max_replay=max(max_replay,error);assert error<1e-15
    summary=[]
    for name in names:
        rr=[lookup[s,name] for s in seeds];fields=['actual_excess','truncation_excess','expected_sampling_excess','bayes_risk']
        mm={k:float(np.mean([r[k] for r in rr])) for k in fields}
        summary.append(dict(method=name,**mm,
            ideal_512_expected_excess=mm['truncation_excess']+mm['expected_sampling_excess'],
            conditional_total_risk=mm['bayes_risk']+mm['actual_excess'],
            maximum_abs_grid_change_actual=max(abs(r['grid_refinement_actual_difference']) for r in rr),
            maximum_abs_grid_change_truncation=max(abs(r['grid_refinement_truncation_difference']) for r in rr),
            aggregate_actual_replica_pair_difference=float(np.mean([r['cross_replica_estimates'][0]['actual_excess']-r['cross_replica_estimates'][1]['actual_excess'] for r in rr]))))
    rng=np.random.default_rng(124113);ids=rng.integers(0,16,(20000,16));paired=[]
    for name in names[1:]:
        for metric in ['actual_excess','truncation_excess']:
            delta=np.array([lookup[s,names[0]][metric]-lookup[s,name][metric] for s in seeds])
            paired.append(dict(comparison=names[0]+' minus '+name,metric=metric,mean_difference=float(delta.mean()),
                descriptive_ci95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),
                lower_tasks=int((delta<-1e-12).sum()),equal_tasks=int((np.abs(delta)<=1e-12).sum()),higher_tasks=int((delta>1e-12).sum())))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),curve_hashes=16,replayed_metrics=384,
        maximum_metric_replay_difference=max_replay,mean_reference_replica_l2_gap=float(np.mean(replica_gap)),maximum_reference_replica_l2_gap=float(max(replica_gap))),
        summary=summary,paired=paired,scope='Conditional risk estimated with numerical region volumes and 1025-point input quadrature; four independent reference batches; old tasks; no query answers or tuning.')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));xx=np.arange(len(names));tr=np.array([r['truncation_excess'] for r in summary]);mc=np.array([r['expected_sampling_excess'] for r in summary]);ac=np.array([r['actual_excess'] for r in summary])
    axes[0].bar(xx,tr,label='Missing-mode mean bias',color='#247f80');axes[0].bar(xx,mc,bottom=tr,label='Expected 512-sample variance',color='#d3a441');axes[0].scatter(xx,ac,color='#333333',s=25,label='Frozen actual 512-sample state',zorder=3)
    axes[0].set_xticks(xx,['ALM20','Adam60','Adam240','Direct128','No dual','PC80'],rotation=20);axes[0].set_ylabel('Mean conditional excess risk over Bayes');axes[0].legend(fontsize=7);axes[0].grid(axis='y',alpha=.2)
    left=np.array([lookup[s,'adam240_prior256']['truncation_excess'] for s in seeds]);right=np.array([lookup[s,'alm20_prior256']['truncation_excess'] for s in seeds]);lim=max(left.max(),right.max())*1.07
    axes[1].plot([0,lim],[0,lim],'--',color='#aab1b7',linewidth=1);axes[1].scatter(left,right,color='#247f80',s=30)
    axes[1].set_xlabel('Adam240 missing-mode conditional excess risk');axes[1].set_ylabel('ALM20 missing-mode conditional excess risk');axes[1].grid(alpha=.2)
    fig.suptitle('First-write functional error — observed contexts only, no query answers');fig.tight_layout();fig.savefig(a.out/'conditional_function_risk.png',dpi=165);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

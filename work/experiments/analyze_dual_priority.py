"""Independent state/query/order audit and plots for same-pool ranking."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dual_priority_proposals as proposal
import conflict_feedback_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(delta):
    index=np.random.default_rng(472189).integers(0,len(delta),(20000,len(delta)))
    return np.quantile(delta[index].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    inp=args.project/'results/dual_priority/diagnostic';out=args.project/'results/dual_priority/analysis'
    protocol=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['complete'] and run['actual_first_write_readouts']==784
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());lookup={(r['seed'],r['scorer'],r['budget']):r for r in rows};assert len(lookup)==len(rows)==784
    pools=json.loads((inp/'pools.json').read_text());state_checks=0;orders=0;full_states=0;all_pools=[]
    for item in pools:
        path=inp/item['file'];assert sha(path)==item['sha256'];pool=json.loads(path.read_text());all_pools.append(pool)
        seed=pool['seed'];rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)[:4];q=rng.uniform(0,1,2048)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)[:4];target=model.base.forward(q,truth)
        keys=[bytes.fromhex(k) for k in pool['extra_keys']];features={bytes.fromhex(k):z for k,z in pool['features'].items()};full=None
        for scorer in protocol['scorers']:
            assert [k.hex() for k in proposal.rank(keys,features,scorer)]==pool['orders'][scorer];orders+=1
            for budget in protocol['budgets']:
                row=lookup[seed,scorer,budget];assert [k.hex() for k in proposal.canonical_select(keys,features,scorer,budget)]==row['selected_keys']
                path=inp/row['state_file'];assert sha(path)==row['state_sha256']
                with np.load(path) as z:anchor=z['anchor'];samples=z['samples']
                pred=model.old.old.old.old.posterior.make_predict(samples)
                assert float(np.mean((pred(q)-target)**2))==row['query_mse'];assert float(np.max(np.abs(pred(x)-v)))==row['support_max_error'];state_checks+=1
                assert not row['lost_modes']
                if budget is None:
                    if full is None:full=(anchor.copy(),samples.copy())
                    else:assert np.array_equal(anchor,full[0]) and np.array_equal(samples,full[1]);full_states+=1
    summary=[];paired=[]
    for budget in protocol['budgets']:
        primary=np.array([lookup[s,'dual_gain',budget]['query_mse'] for s in protocol['seeds']]);pce=np.array([lookup[s,'dual_gain',budget]['conditional_excess'] for s in protocol['seeds']])
        for scorer in protocol['scorers']:
            rr=[lookup[s,scorer,budget] for s in protocol['seeds']]
            summary.append(dict(scorer=scorer,budget=budget,mean_query_mse=float(np.mean([r['query_mse'] for r in rr])),
                mean_conditional_excess=float(np.mean([r['conditional_excess'] for r in rr])),mean_ideal_truncation=float(np.mean([r['ideal_truncation'] for r in rr])),
                mean_mass=float(np.mean([r['mass'] for r in rr])),new_modes=sum(len(r['gained_modes']) for r in rr),tasks_with_new_modes=sum(bool(r['gained_modes']) for r in rr),
                extra_geometry_calls=sum(r['extra_geometry_calls'] for r in rr)))
            if scorer!='dual_gain':
                mse=np.array([r['query_mse'] for r in rr]);ce=np.array([r['conditional_excess'] for r in rr]);delta=primary-mse;dc=pce-ce
                paired.append(dict(budget=budget,comparator=scorer,query_delta=float(delta.mean()),query_ci95=ci(delta),conditional_excess_delta=float(dc.mean()),conditional_ci95=ci(dc),
                    better_tasks=int((delta<-1e-15).sum()),equal_tasks=int((np.abs(delta)<=1e-15).sum()),worse_tasks=int((delta>1e-15).sum()),
                    selected_sets_equal=sum(lookup[s,'dual_gain',budget]['selected_keys']==lookup[s,scorer,budget]['selected_keys'] for s in protocol['seeds'])))
    costs={key:sum(p['counts'][key] for p in all_pools) for key in all_pools[0]['counts']}
    costs['routing_seconds']=sum(p['routing']['routing_seconds'] for p in all_pools)
    result=dict(analysis_source_sha256=sha(Path(__file__)),input_sha256={k:sha(inp/k) for k in ['protocol.json','run_audit.json','rows.json','pools.json']},
        audit=dict(states_and_queries=state_checks,score_orders=orders,full_states_equal=full_states,source_hashes=len(protocol['source_sha256']),all_old_modes_preserved=True),
        summary=summary,paired=paired,shared_diagnostic_costs=costs,
        scope='16 old tasks; paired descriptive intervals with no multiplicity correction; support-only ranking, first-write only, shared diagnostic not deployment timing')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,5));positions=np.arange(len(protocol['budgets']));labels=[str(k) if k is not None else 'Full' for k in protocol['budgets']]
    colors=['#177e89','#dd9b29','#a33b52','#61428c','#6b8f3c','#2b6ea6','#777777']
    for scorer,color in zip(protocol['scorers'],colors):
        rr=[next(r for r in summary if r['scorer']==scorer and r['budget']==k) for k in protocol['budgets']]
        for ax,metric in zip(axes,['mean_query_mse','mean_conditional_excess','new_modes']):
            ax.plot(positions,[r[metric] for r in rr],marker='o',label=scorer,color=color,lw=2 if scorer=='dual_gain' else 1.2,alpha=.85)
    for ax,title in zip(axes,['Actual first-write query MSE (lower is better)','First-write conditional excess (lower is better)','New feasible modes across 16 tasks']):
        ax.set_xticks(positions,labels);ax.set_xlabel('Extra geometry budget per task');ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[2].legend(fontsize=8)
    fig.suptitle('Same candidate pool, different priority: no query labels in ranking');fig.tight_layout();fig.savefig(out/'dual_priority.png',dpi=170);plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),figure_sha256=sha(out/'dual_priority.png'))
    (out/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(audit=result['audit'],summary=summary,paired=paired),indent=2),flush=True)


if __name__=='__main__':main()

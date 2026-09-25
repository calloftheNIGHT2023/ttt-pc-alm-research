"""Complete development analysis and saved-state replay, no solver retuning."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stateful_posterior_memory as model


def bootstrap(delta):
    rng=np.random.default_rng(199);means=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'episodes.json').read_text())
    names=[c['name'] for c in protocol['configs']];seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']));stages=protocol['stages']
    assert len(rows)==len(names)*len(seeds)*len(stages)==1152
    lookup={(r['method'],r['seed'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    max_replay_difference=0.;replay_count=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for name in names:
            for n in stages:
                r=lookup[name,seed,n];path=a.input/r['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['state_sha256']
                with np.load(path) as z:
                    assert np.array_equal(z['anchor'],np.array(r['anchor_output']))
                    assert z['samples'].nbytes+z['anchor'].nbytes==r['persistent_state_bytes']
                    pred=model.posterior.make_predict(z['samples']);mse=float(np.mean((pred(q)-target)**2))
                    gap=abs(mse-r['raw_query_mse']);max_replay_difference=max(max_replay_difference,gap)
                    assert gap<1e-15
                    assert abs(float(np.max(np.abs(pred(x[:n])-v[:n])))-r['support_max_error'])<1e-15
                    assert len(z['samples'])==r['query_read_samples']
                replay_count+=1
        # Every pool and the switch ablation must share its own generator's first stage.
        for family in ['alm20','adam60','adam240','direct128','nodual20','pc80']:
            group=[lookup[name,seed,4] for name in names if name.startswith(family+'_')]
            assert all(r['raw_query_mse']==group[0]['raw_query_mse'] and r['positive_cell_set_sha256']==group[0]['positive_cell_set_sha256'] for r in group)
            assert all(r['effective_pool']=='prior256' for r in group)
    summary=[];vectors={};costs={};later_vectors={}
    for name in names:
        rr=[[lookup[name,s,n] for n in stages] for s in seeds]
        error=np.array([[r['query_mse'] for r in row] for row in rr]);elapsed=np.array([[r['adaptation_seconds']+r['read_queries_seconds'] for r in row] for row in rr])
        vectors[name]=error.mean(1);later_vectors[name]=error[:,1:].mean(1);costs[name]=elapsed.sum(1)
        summary.append(dict(method=name,trajectory_mean_mse=float(error.mean()),later_mean_mse=float(error[:,1:].mean()),
            mse_by_stage=error.mean(0).tolist(),median_full_stream_seconds=float(np.median(elapsed.sum(1))),
            median_per_stage_seconds=np.median(elapsed,axis=0).tolist(),
            feasible_tasks_by_stage=[sum(lookup[name,s,n]['support_feasible'] for s in seeds) for n in stages],
            mean_positive_regions_by_stage=[float(np.mean([lookup[name,s,n]['positive_volume_regions'] for s in seeds])) for n in stages],
            minimum_persistent_state_bytes=min(r['persistent_state_bytes'] for row in rr for r in row),
            maximum_persistent_state_bytes=max(r['persistent_state_bytes'] for row in rr for r in row),
            maximum_old_plus_new_state_bytes=max(r['persistent_state_bytes']+r['previous_state_bytes'] for row in rr for r in row),
            maximum_pool_parameter_bytes=max(r['pool_parameter_bytes'] for row in rr for r in row),
            maximum_pool_signature_bytes=max(r['pool_signature_bytes'] for row in rr for r in row),
            maximum_geometry_numeric_arrays_subtotal=max(r['geometry_numeric_arrays_subtotal'] for row in rr for r in row)))
    candidate=protocol['primary_candidate'];comparisons=[]
    for name in names:
        if name==candidate:continue
        diff=vectors[candidate]-vectors[name];late=later_vectors[candidate]-later_vectors[name]
        comparisons.append(dict(candidate=candidate,comparator=name,primary=name in protocol['primary_comparators'],
            mean_mse_difference=float(diff.mean()),descriptive_ci95=bootstrap(diff),
            candidate_better_streams=int((diff<-1e-15).sum()),equal_streams=int((np.abs(diff)<=1e-15).sum()),
            later_mean_mse_difference=float(late.mean()),later_descriptive_ci95=bootstrap(late),
            median_full_stream_time_difference=float(np.median(costs[candidate]-costs[name]))))
    reuse=[]
    for family in ['alm20','adam60','adam240','direct128','nodual20','pc80']:
        left=family+'_posterior_mix';right=family+'_prior768';diff=vectors[left]-vectors[right]
        reuse.append(dict(family=family,comparison=left+' minus '+right,mean_mse_difference=float(diff.mean()),
            descriptive_ci95=bootstrap(diff),better_streams=int((diff<-1e-15).sum()),
            median_full_stream_time_difference=float(np.median(costs[left]-costs[right]))))
    output=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes=replay_count,
        all_query_predictions_replayed=True,max_replay_mse_difference=max_replay_difference,
        equal_initial_stage_across_pools_and_switches=True),summary=summary,comparisons=comparisons,reuse_comparisons=reuse,
        interpretation='Old development tasks; all confidence intervals descriptive; no independent confirmation or full native peak-memory claim.')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    colors={'alm20':'#187c79','adam60':'#dd8a32','adam240':'#b64b37','direct128':'#8994a1','nodual20':'#8865a8','pc80':'#659d42'}
    for family,color in colors.items():
        for suffix,marker in [('prior768','o'),('posterior_mix','*')]:
            r=next(s for s in summary if s['method']==family+'_'+suffix)
            axes[0].scatter(r['median_full_stream_seconds'],r['trajectory_mean_mse'],marker=marker,s=85,color=color,label=family+(' reuse' if marker=='*' else ' fresh'))
    for name,color in [('alm20_then_direct128_mix','#187c79'),('adam60_then_direct128_mix','#dd8a32')]:
        r=next(s for s in summary if s['method']==name);axes[0].scatter(r['median_full_stream_seconds'],r['trajectory_mean_mse'],marker='D',s=50,color=color,label=name.split('_then')[0]+' then direct')
    axes[0].set_xlabel('Median full stream: fit + 2048-query read (seconds)');axes[0].set_ylabel('Four-stage mean unseen-query MSE');axes[0].set_yscale('log');axes[0].grid(alpha=.2)
    for name in ['alm20_prior768','alm20_posterior_mix','adam240_posterior_mix','alm20_then_direct128_mix']:
        r=next(s for s in summary if s['method']==name);axes[1].plot(stages,r['mse_by_stage'],'o-',label=name)
    axes[1].set_xticks(stages);axes[1].set_xlabel('Observed support size');axes[1].set_ylabel('Mean unseen-query MSE');axes[1].set_yscale('log');axes[1].grid(alpha=.2);axes[1].legend(fontsize=7)
    axes[0].legend(fontsize=6.7,ncol=2);fig.suptitle('Closed-loop posterior-state reuse — all 16 old development streams')
    fig.tight_layout();fig.savefig(a.out/'stateful_online_results.png',dpi=165);plt.close(fig)
    print(json.dumps(dict(audit=output['audit'],summary=summary,comparisons=comparisons,reuse_comparisons=reuse),indent=2))


if __name__=='__main__':main()

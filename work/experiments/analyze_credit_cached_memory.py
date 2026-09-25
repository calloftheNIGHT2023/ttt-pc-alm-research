"""Output-equivalence audit and paired full-cost analysis, clustered by task."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import credit_cached_mode_memory as model


def ci(d):
    indices=np.random.default_rng(88761).integers(0,len(d),(20000,len(d)))
    return np.quantile(d[indices].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    root=a.project/'results/credit_cached_memory';inp=root/'development';out=root/'analysis';oldroot=a.project/'results/split_activity_modes/development'
    protocol=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'episodes.json').read_text());assert len(rows)==1792
    oldrows=json.loads((oldroot/'episodes.json').read_text());oldlookup={(r['seed'],r['method'],r['n_context']):r for r in oldrows}
    oldprotocol=json.loads((oldroot/'protocol.json').read_text())
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256(Path(__file__).with_name('credit_cache_development.json').read_bytes()).hexdigest()==protocol['config_sha256']
    lookup={(r['repetition'],r['seed'],r['method'],r['n_context']):r for r in rows};assert len(lookup)==len(rows)
    config={c['name']:c for c in protocol['configs']};seeds=list(range(protocol['seed0'],protocol['seed0']+protocol['count']))
    cache={};query_replays=0;state_matches=0;lower_bound_checks=0
    for r in rows:
        c=config[r['method']];key=(r['seed'],c['reference'],r['n_context']);reference=oldlookup[key]
        if key not in cache:
            path=oldroot/reference['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['state_sha256']
            with np.load(path) as z:cache[key]=(z['anchor'].copy(),z['samples'].copy())
            rng=np.random.default_rng(r['seed']);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,protocol['queries'])
            predicted=model.previous.posterior.make_predict(cache[key][1])(q);mse=float(np.mean((predicted-model.base.forward(q,truth))**2))
            assert mse==reference['raw_query_mse'];query_replays+=1
        path=inp/r['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['state_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['anchor'],cache[key][0]) and np.array_equal(z['samples'],cache[key][1]),key
        assert r['query_mse']==reference['query_mse'] and r['raw_query_mse']==reference['raw_query_mse']
        assert r['positive_mode_keys']==reference['positive_mode_keys'];state_matches+=1
        if r['n_context']!=4:assert not r['credit_cache_active'] and r['effective_generator']=='direct'
        if r['credit_cache_active']:
            assert len(r['certified_mode_keys'])==len(r['certificate_lower_bounds'])==r['certified_rejected_patterns']
            assert all(b>0 for b in r['certificate_lower_bounds'])
            assert not set(r['positive_mode_keys'])&set(r['certified_mode_keys']);lower_bound_checks+=len(r['certificate_lower_bounds'])
    summary=[];vectors={};time_arrays={}
    for c in protocol['configs']:
        name=c['name'];tt=np.array([[[lookup[rep,s,name,n]['adaptation_seconds']+lookup[rep,s,name,n]['read_queries_seconds'] for n in protocol['stages']]
            for s in seeds] for rep in range(protocol['repetitions'])])
        total=tt.sum(2);vectors[name]=total.mean(0);time_arrays[name]=tt
        first=[lookup[rep,s,name,4] for rep in range(protocol['repetitions']) for s in seeds]
        geom=[r.get('geometry_calls',r.get('initial_geometry_calls',0)+r.get('completion_geometry_calls',0)) for r in first]
        summary.append(dict(method=name,reference=c['reference'],query_mse=float(np.mean([lookup[0,s,name,n]['query_mse'] for s in seeds for n in protocol['stages']])),
            mean_full_seconds=float(vectors[name].mean()),median_task_mean_full_seconds=float(np.median(vectors[name])),
            mean_full_seconds_by_repeat=total.mean(1).tolist(),mean_first_seconds=float(tt[:,:,0].mean()),
            mean_first_geometry_calls=float(np.mean(geom)),mean_first_discovery_seconds=float(np.mean([r['discovery_seconds'] for r in first])),
            mean_first_credit_check_seconds=float(np.mean([r.get('credit_check_seconds',0) for r in first])),
            mean_certified_patterns=float(np.mean([r.get('certified_rejected_patterns',0) for r in first])),
            max_cache_key_and_lower_bytes=max(r.get('certificate_cache_key_and_lower_bytes',0) for r in first),
            maximum_old_plus_new_state_bytes=max(r['persistent_state_bytes']+r['previous_state_bytes'] for r in rows if r['method']==name)))
    candidate=protocol['primary_candidate'];paired=[]
    comparisons=[(candidate,n) for n in config if n!=candidate]+[(f'{prefix}_residual',f'{prefix}_none') for prefix in ['adam8','adam16','nodual16','pc80']]
    for left,right in comparisons:
        delta=vectors[left]-vectors[right];ratio=(vectors[left]/vectors[right])-1
        paired.append(dict(candidate=left,comparator=right,primary=left==candidate and right in protocol['primary_cost_comparators'],
            mean_full_cost_difference=float(delta.mean()),descriptive_ci95=ci(delta),
            mean_task_relative_change=float(ratio.mean()),relative_change_ci95=ci(ratio),
            faster_tasks=int((delta<0).sum()),tasks=len(delta),
            first_write_mean_difference=float((time_arrays[left][:,:,0]-time_arrays[right][:,:,0]).mean()),
            later_writes_mean_difference=float((time_arrays[left][:,:,1:]-time_arrays[right][:,:,1:]).sum(2).mean())))
    result=dict(audit=dict(source_hashes=len(protocol['source_sha256']),state_hashes_and_exact_reference_matches=state_matches,
        independently_replayed_unique_reference_states=query_replays,positive_bound_records=lower_bound_checks,all_positive_sets_unchanged=True,
        later_stages_direct_only=True),summary=summary,paired=paired,
        scope='16 old tasks; 2 timing repeats averaged per task before paired bootstrap; no independent quality confirmation')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13.5,6.5),sharey=True);pos=np.arange(len(summary))
    colors=['#167e79' if r['method'].startswith('alm') else '#d79434' if r['method'].startswith('adam') else '#778995' for r in summary]
    fields=['mean_full_seconds','mean_first_geometry_calls','mean_first_credit_check_seconds'];labels=['Mean full stream fit + query reads (s)','First-write geometry calls','Credit checking per first write (ms)']
    for ax,field,label in zip(axes,fields,labels):
        values=np.array([r[field] for r in summary])*(1000 if field.endswith('check_seconds') else 1)
        ax.barh(pos,values,color=colors);ax.set_xlim(0,float(values.max())*1.25);ax.grid(axis='x',alpha=.2);ax.set_xlabel(label)
        for y,value in zip(pos,values):ax.text(value+values.max()*.025,y,f'{value:.3f}' if field=='mean_full_seconds' else f'{value:.1f}',va='center',fontsize=7)
    axes[0].set_yticks(pos,[r['method'] for r in summary],fontsize=8);axes[0].invert_yaxis()
    fig.suptitle('Online credit cache: actual full cost, unchanged predictions, two timing repeats');fig.tight_layout()
    fig.savefig(out/'credit_cache_results.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

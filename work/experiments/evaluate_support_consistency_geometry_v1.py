"""324 sealed proposals, common exact-certified geometry, explicit 297 controls."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha,describe,certify_label

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']
POLICIES=['first_fit','uniform_state']
NAMES=[p+'/'+n for p in POLICIES for n in CHANNELS]


def run(root,out):
    started=time.perf_counter();base=root/'results/support_consistency_trigger';source=base/'development_v1'
    ss=read(source/'summary.json');assert ss['passed'];manifest=read(source/'before_geometry_manifest.json')
    assert sha(source/'before_geometry_manifest.json')==ss['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    oldroot=root/'results/branch_image_chain/geometry_v1';os=read(oldroot/'summary.json');assert os['passed']
    for n,digest in os['outputs_sha256'].items():assert sha(oldroot/n)==digest
    older=root/'results/novel_branch_continuation/development_v1';older_summary=read(older/'summary.json');assert older_summary['passed']
    for n,digest in older_summary['outputs_sha256'].items():assert sha(older/n)==digest
    raw297={r['seed']:r for r in read(older/'proposals.json')};p297=defaultdict(dict)
    for row in read(older/'scores.json'):
        if row['grid']==257:p297[row['seed']][row['method']]=set(raw297[row['seed']]['old_positive_modes'])|set(row['new_positive_modes'])
    grouped=defaultdict(list)
    for row in read(source/'rows.json'):grouped[row['seed']].append(read(source/row['file']))
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'complete_credit_mode_geometry_v1.py','evaluate_complete_credit_mode_geometry_v1.py']},
         input_summary_sha256=sha(source/'summary.json'),old321_summary_sha256=sha(oldroot/'summary.json'),old297_summary_sha256=sha(older/'summary.json'),
         methods=NAMES,query_targets_accessed=False,posterior_moments_accessed=False,resources_matched=False,global_lp_scope='Post-seal evaluator only'))
    files={};tasks=[];counts=Counter()
    for oldrow in read(oldroot/'tasks.json'):
        seed=oldrow['seed'];old=read(oldroot/oldrow['file']);x=np.array(old['x_observed']);v=np.array(old['v_observed'])
        prior321=set(old['prior_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        assert len(p297[seed])==20
        prior=prior321|set().union(*p297[seed].values());counts['prior_extra_297_pairs']+=len(prior-prior321)
        cache={}
        for folder in ['post_escape_continuation','factorized_dual_branch_search','branch_image_chain']:
            cache.update(read(root/'results'/folder/'geometry_v1'/oldrow['file'])['geometry'])
        props={n:set() for n in NAMES}
        for rec in grouped[seed]:
            assert rec['x_observed']==old['x_observed'] and rec['v_observed']==old['v_observed']
            for name in CHANNELS:props[rec['policy']+'/'+name].update(p['mode'] for p in rec['results'][name]['proposals'])
        classified={}
        for mode in sorted(set().union(*props.values())):
            if mode in cache:result=cache[mode];counts['reused_classifications']+=1
            else:result=geometry.classify_mode(x,v,mode);counts['new_classifications']+=1;counts['new_lp_calls']+=result['lp_calls']
            _,a,rhs,_,_=geometry.matrices(x,v,mode);counts['certificate_checks']+=certify_label(a,rhs,result)
            counts[result['classification']]+=1;classified[mode]=result
        methods={n:describe(props[n],classified,prior) for n in NAMES};positive={n:set(m['positive_modes']) for n,m in methods.items()}
        controls=set().union(*(positive[p+'/'+n] for p in POLICIES for n in ['residual','bp','random_sign','zero']))
        comparisons={}
        for name in NAMES:
            policy,channel=name.split('/');other=('uniform_state' if policy=='first_fit' else 'first_fit')+'/'+channel
            comparisons[name]=dict(new_vs_prior=sorted(positive[name]-prior),new_vs_prior_and_controls=sorted(positive[name]-prior-controls),
                                   new_vs_other_trigger=sorted(positive[name]-positive[other]),new_vs_prior321=sorted(positive[name]-prior321))
        rec=dict(seed=seed,x_observed=old['x_observed'],v_observed=old['v_observed'],
                 original_positive_modes=raw297[seed]['old_positive_modes'],prior_321_positive_modes=sorted(prior321),prior_positive_modes=sorted(prior),
                 prior_extra_297_modes=sorted(prior-prior321),baseline_methods_297={n:sorted(vv) for n,vv in p297[seed].items()},
                 geometry=classified,methods=methods,comparisons=comparisons)
        filename=f'{seed}_geometry.json';save(out/filename,rec);files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename]))
        if (seed-5910000+1)%8==0:print(dict(tasks=seed-5910000+1,seconds=time.perf_counter()-started),flush=True)
    allrecords=[read(out/t['file']) for t in tasks]
    aggregate={n:dict(proposal_pairs=sum(len(t['methods'][n]['proposal_modes']) for t in allrecords),positive_pairs=sum(len(t['methods'][n]['positive_modes']) for t in allrecords),
        **{field:sum(len(t['comparisons'][n][field]) for t in allrecords) for field in ['new_vs_prior','new_vs_prior_and_controls','new_vs_other_trigger','new_vs_prior321']}) for n in NAMES}
    for name,value in [('tasks.json',tasks),('aggregate.json',aggregate)]:save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,seconds=time.perf_counter()-started,outputs_sha256=files,
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise

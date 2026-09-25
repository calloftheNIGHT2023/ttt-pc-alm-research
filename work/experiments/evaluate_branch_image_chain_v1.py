"""321 post-seal common geometry, against all 319/320 pools."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha,describe,certify_label

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    begin=time.perf_counter();source=root/'results/branch_image_chain/development_v1'
    summary=read(source/'summary.json');manifest=read(source/'before_geometry_manifest.json');assert summary['passed']
    assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    priorroot=root/'results/factorized_dual_branch_search/geometry_v1';older=root/'results/post_escape_continuation/geometry_v1'
    prior_summary=read(priorroot/'summary.json');assert prior_summary['passed']
    assert read(root/'results/factorized_dual_branch_search/audit_v1/summary.json')['passed']
    for n,digest in prior_summary['outputs_sha256'].items():assert sha(priorroot/n)==digest
    older_summary=read(older/'summary.json');assert older_summary['passed']
    for n,digest in older_summary['outputs_sha256'].items():assert sha(older/n)==digest
    grouped=defaultdict(list)
    for row in read(source/'rows.json'):grouped[row['seed']].append(row)
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in
                                [Path(__file__).name,'complete_credit_mode_geometry_v1.py','evaluate_complete_credit_mode_geometry_v1.py']},
                  proposals_summary_sha256=sha(source/'summary.json'),prior_summary_sha256=sha(priorroot/'summary.json'),
                  older_summary_sha256=sha(older/'summary.json'),
                  tasks=64,channels=CHANNELS,query_targets_accessed=False,posterior_moments_accessed=False,
                  resources_matched=False,global_lp_scope='Post-seal evaluator, not online proposal generator')
    save(out/'protocol.json',protocol);counts=Counter();tasks=[];files={}
    for oldrow in read(priorroot/'tasks.json'):
        seed=oldrow['seed'];old=read(priorroot/oldrow['file']);props={n:set() for n in CHANNELS}
        prior=set(old['prior_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        cached=read(older/f'{seed}_geometry.json')['geometry'];cached.update(old['geometry'])
        for row in grouped[seed]:
            rec=read(source/row['file']);assert rec['x_observed']==old['x_observed'] and rec['v_observed']==old['v_observed']
            for name in CHANNELS:props[name].update(p['mode'] for p in rec['results'][name]['proposals'])
        classified={};x=np.array(old['x_observed']);v=np.array(old['v_observed'])
        for mode in sorted(set().union(*props.values())):
            if mode in cached:result=cached[mode];counts['reused_classifications']+=1
            else:
                result=geometry.classify_mode(x,v,mode);counts['new_classifications']+=1;counts['new_lp_calls']+=result['lp_calls']
            _,a,rhs,_,_=geometry.matrices(x,v,mode)
            counts['certificate_checks']+=certify_label(a,rhs,result);counts[result['classification']]+=1;classified[mode]=result
        methods={n:describe(props[n],classified,prior) for n in CHANNELS}
        positive={n:set(m['positive_modes']) for n,m in methods.items()};comparisons={}
        for name in CHANNELS:
            other=set().union(*(positive[n] for n in CHANNELS if n!=name))
            comparisons[name]=dict(new_vs_prior=sorted(positive[name]-prior),
                                   new_vs_prior_and_other_channels=sorted(positive[name]-prior-other),
                                   new_vs_own_row_selector=sorted(positive[name]-set(old['methods'][name]['positive_modes'])))
        candidate=positive['dual']|positive['dual_plus_residual']
        controls=set().union(*(positive[n] for n in ['residual','bp','random_sign','zero']))
        rec=dict(seed=seed,x_observed=old['x_observed'],v_observed=old['v_observed'],prior_positive_modes=sorted(prior),
                 geometry=classified,methods=methods,comparisons=comparisons,combined_candidate_exclusive=sorted(candidate-prior-controls))
        filename=f'{seed}_geometry.json';save(out/filename,rec);files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename],methods=methods,comparisons=comparisons,
                          combined_candidate_exclusive=rec['combined_candidate_exclusive']))
        if grouped[seed]:print(dict(seed=seed,modes=len(classified),seconds=time.perf_counter()-begin),flush=True)
    aggregate={n:dict(proposal_pairs=sum(len(t['methods'][n]['proposal_modes']) for t in tasks),
                      positive_pairs=sum(len(t['methods'][n]['positive_modes']) for t in tasks),
                      new_vs_prior=sum(len(t['comparisons'][n]['new_vs_prior']) for t in tasks),
                      new_vs_prior_and_other_channels=sum(len(t['comparisons'][n]['new_vs_prior_and_other_channels']) for t in tasks),
                      new_vs_own_row_selector=sum(len(t['comparisons'][n]['new_vs_own_row_selector']) for t in tasks),
                      tasks_with_new_positive=sum(bool(t['comparisons'][n]['new_vs_prior']) for t in tasks)) for n in CHANNELS}
    for name,data in [('tasks.json',tasks),('aggregate.json',aggregate)]:save(out/name,data);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,
                combined_candidate_exclusive_pairs=sum(len(t['combined_candidate_exclusive']) for t in tasks),
                seconds=time.perf_counter()-begin,outputs_sha256=files,query_targets_accessed=False,
                resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

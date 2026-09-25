"""312: common, post-seal support geometry for every predeclared configuration."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback

import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha, certify_label, describe
from paired_primal_dual_predictor_v1 import NAMES


def verified(folder):
    summary, protocol = read(folder/'summary.json'), read(folder/'protocol.json')
    assert summary.get('passed',summary.get('execution_passed')) and not (folder/'failure.json').exists()
    for n,digest in summary['outputs_sha256'].items(): assert sha(folder/n)==digest
    return summary,protocol


def comparisons(methods, old, prior):
    positive = {n:set(r['positive_modes']) for n,r in methods.items()}
    main=positive['paired_bu']
    controls=['normal_one','normal_two','bias_only','dual_only','dual_full','paired_b_zero_u',
              'paired_b_last_du','paired_b_random_du','random_db_paired_u']
    ans={f'paired_new_vs_original_and_{n}':sorted(main-old-positive[n]) for n in controls}
    ans['paired_new_vs_original_and_single_blocks']=sorted(main-old-positive['bias_only']-positive['dual_only']-positive['dual_full'])
    ans['paired_new_vs_original_and_all_308_309']=sorted(main-prior)
    ans['paired_new_vs_original_and_controls']=sorted(main-old-set().union(*(positive[n] for n in controls)))
    ans['paired_new_vs_original_and_prior_and_controls']=sorted(main-prior-set().union(*(positive[n] for n in controls)))
    return ans


def run(root,out):
    begin=time.perf_counter();folder=root/'results/paired_primal_dual_predictor/development_v1'
    support,protocol=verified(folder)
    assert support['counts']['configurations']==1572 and support['counts']['local_steps']==1703
    assert sha(folder/'before_geometry_manifest.json')==support['manifest_sha256']
    manifest=read(folder/'before_geometry_manifest.json')
    assert manifest['all_states_processed'] and not manifest['geometry_or_posterior_accessed']
    for n,digest in manifest['files_sha256'].items():assert sha(folder/n)==digest
    old_dir=root/'results/complete_credit_amplitude_events/geometry_v1'
    mask_dir=root/'results/layer_credit_interaction/geometry_v1'
    old_summary,old_protocol=verified(old_dir); mask_summary,mask_protocol=verified(mask_dir)
    old_tasks={t['seed']:t for t in read(old_dir/'tasks.json')}
    mask_tasks={t['seed']:t for t in read(mask_dir/'tasks.json')}
    source_names={Path(__file__).name}
    for pp in [protocol,old_protocol,mask_protocol]:
        for n,digest in pp['source_sha256'].items():
            assert sha(root/'work/experiments'/n)==digest;source_names.add(n)
    by_task=defaultdict(list)
    for r in read(folder/'rows.json'):by_task[r['seed']].append(r)
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(source_names)},
        support_summary_sha256=sha(folder/'summary.json'),old_summary_sha256=sha(old_dir/'summary.json'),
        mask_summary_sha256=sha(mask_dir/'summary.json'),tasks=64,states=131,methods=NAMES,
        query_targets_accessed=False,posterior_moments_accessed=False,resources_matched=False,
        numerical_volume_is_exact=False,global_lp_scope='Post-seal common evaluator only; never candidate credit.'))
    tasks,states,files,counts=[],[],{},Counter()
    for seed in sorted(old_tasks):
        old=read(old_dir/old_tasks[seed]['file']);mask=read(mask_dir/mask_tasks[seed]['file'])
        x,v=np.array(old['x_observed']),np.array(old['v_observed'])
        old_positive=set(old['original_positive_modes'])
        prior=old_positive|set().union(*(set(m['positive_modes']) for data in [old,mask] for m in data['methods'].values()))
        cache=dict(old['geometry']);cache.update(mask['geometry'])
        props={n:set() for n in NAMES}; output_props={n:set() for n in NAMES};payloads=[]
        for row in by_task[seed]:
            assert sha(folder/row['file'])==row['sha256'];payload=read(folder/row['file'])
            assert payload['seed']==seed and payload['location']==row['location']
            assert np.array_equal(x,payload['x_observed']) and np.array_equal(v,payload['v_observed'])
            assert [m['method'] for m in payload['methods']]==NAMES
            for m in payload['methods']:
                assert sorted(set(m['point_modes']))==row['methods'][m['method']]==m['proposal_modes']
                props[m['method']].update(m['proposal_modes'])
                output_props[m['method']].update(m['point_modes'][1:])
            payloads.append(payload)
        all_modes=old_positive|set().union(*props.values()); classified={}
        matrices={}
        for mode in sorted(all_modes):
            if mode in cache:
                result=cache[mode]; counts['reused_classifications']+=1
            else:
                result=geometry.classify_mode(x,v,mode);counts['new_classifications']+=1
                counts['new_LP_calls']+=result['lp_calls']
            _,a,rhs,_,_=geometry.matrices(x,v,mode);matrices[mode]=(a,rhs)
            counts['certificate_rechecks']+=certify_label(a,rhs,result)
            counts[result['classification']]+=1; classified[mode]=result
        for mode in old_positive:assert classified[mode]['positive_volume_certified']
        methods={n:describe(props[n],classified,old_positive) for n in NAMES}
        output_methods={n:describe(output_props[n],classified,old_positive) for n in NAMES}
        for n in NAMES:methods[n]['retained_pool']=sorted(old_positive|set(methods[n]['positive_modes']))
        for payload in payloads:
            sm={m['method']:describe(m['proposal_modes'],classified,old_positive) for m in payload['methods']}
            points={}
            for m in payload['methods']:
                checked=[]
                for state,mode in zip(m['states'],m['point_modes']):
                    a,rhs=matrices[mode];cert=geometry.point_certificate(a,rhs,state['b'])
                    checked.append(dict(mode=mode,closed_feasible=cert['closed_region_feasible'],
                        strict_interior=cert['strict_interior'],minimum_slack=cert['minimum_slack']))
                    counts['actual_parameter_point_checks']+=1
                    counts['actual_parameter_points_feasible']+=cert['closed_region_feasible']
                points[m['method']]=checked
            states.append(dict(seed=seed,location=payload['location'],methods=sm,
                comparisons=comparisons(sm,old_positive,prior),actual_points=points,
                metadata=payload['metadata']))
        record=dict(seed=seed,selected_states=len(payloads),methods=methods,output_only_methods=output_methods,
            comparisons=comparisons(methods,old_positive,prior),original_positive_modes=sorted(old_positive),
            all_308_309_positive_modes=sorted(prior),geometry=classified,x_observed=x.tolist(),v_observed=v.tolist())
        filename=f'{seed}_geometry.json';save(out/filename,record);files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,selected_states=len(payloads),file=filename,sha256=files[filename],
            methods=methods,output_only_methods=output_methods,comparisons=record['comparisons']))
    assert len(tasks)==64 and len(states)==131
    aggregate={}
    for n in NAMES:
        rr=[t['methods'][n] for t in tasks]
        aggregate[n]=dict(tasks_with_new_positive=sum(bool(r['new_positive_modes']) for r in rr),
            task_positive_pairs=sum(len(r['new_positive_modes']) for r in rr),
            new_pairs_vs_all_308_309=sum(len(set(t['methods'][n]['positive_modes'])-set(read(out/t['file'])['all_308_309_positive_modes'])) for t in tasks),
            unknown_pairs=sum(len(r['unknown_modes']) for r in rr),
            numeric_new_volume=sum(r['known_new_positive_numeric_volume_sum'] for r in rr))
    comparison_aggregate={n:dict(tasks=sum(bool(t['comparisons'][n]) for t in tasks),
        pairs=sum(len(t['comparisons'][n]) for t in tasks),
        state_mode_occurrences=sum(len(s['comparisons'][n]) for s in states)) for n in tasks[0]['comparisons']}
    for n,value in [('tasks.json',tasks),('states.json',states),('aggregate.json',aggregate),('comparisons.json',comparison_aggregate)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,tasks=64,states=131,methods=len(NAMES),counts=dict(counts),
        classifications_complete=counts['unresolved']==0,aggregate=aggregate,comparisons=comparison_aggregate,
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

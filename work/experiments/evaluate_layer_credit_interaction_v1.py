"""309 common post-seal geometry; evaluate every mask, not a selected winner."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback

import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha, certify_label, describe, FAMILIES
from layer_credit_interaction_v1 import masks

MASKS = [''.join(map(str, m)) for m in masks(4)]
NAMES = [f'{f}_{backend}_{mask}' for f in FAMILIES for backend in ['exact', 'float'] for mask in MASKS+['union']]


def run(root, out):
    begin = time.perf_counter()
    folder = root/'results/layer_credit_interaction/development_v1'
    summary, protocol = read(folder/'summary.json'), read(folder/'protocol.json')
    assert summary['passed'] and summary['counts']['proposals'] == 6288
    assert not (folder/'failure.json').exists()
    assert sha(folder/'before_geometry_manifest.json') == summary['manifest_sha256']
    for n, digest in summary['outputs_sha256'].items():
        assert sha(folder/n) == digest
    for n, digest in protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/n) == digest
    old_dir = root/'results/complete_credit_amplitude_events/geometry_v1'
    old_summary = read(old_dir/'summary.json')
    assert old_summary['execution_passed'] and old_summary['geometry_classification_complete']
    for n, digest in old_summary['outputs_sha256'].items():
        assert sha(old_dir/n) == digest
    old_tasks = {t['seed']: t for t in read(old_dir/'tasks.json')}
    by_task = defaultdict(list)
    for r in read(folder/'rows.json'):
        by_task[r['seed']].append(r)
    old_protocol = read(old_dir/'protocol.json')
    source_names = set(protocol['source_sha256']) | set(old_protocol['source_sha256']) | {Path(__file__).name}
    for n, digest in old_protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/n) == digest
    save(out/'protocol.json', dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(source_names)},
         support_summary_sha256=sha(folder/'summary.json'),
         old_geometry_summary_sha256=sha(old_dir/'summary.json'),
         design_sha256=sha(root/'outputs/ttt-pc-alm-research/309_layer_credit_interaction_protocol.md'),
         tasks=64, states=131, methods=NAMES, query_targets_accessed=False,
         posterior_moments_accessed=False, resources_matched=False,
         global_lp_scope='Offline common evaluator only; no access by candidate.'))
    tasks, counts, files = [], Counter(), {}
    for seed in sorted(old_tasks):
        previous = read(old_dir/old_tasks[seed]['file'])
        x, v = np.array(previous['x_observed']), np.array(previous['v_observed'])
        old_positive = set(previous['original_positive_modes'])
        old_all = old_positive | set().union(*(set(m['positive_modes']) for m in previous['methods'].values()))
        props = {name:set() for name in NAMES}
        for r in by_task[seed]:
            payload = read(folder/r['file'])
            assert sha(folder/r['file']) == r['sha256']
            assert len(payload['proposals']) == 16 and payload['family'] == r['family']
            for p in payload['proposals']:
                for backend in ['exact', 'float']:
                    mode = p[backend+'_mode']; stem = r['family']+'_'+backend+'_'
                    props[stem+p['mask']].add(mode); props[stem+'union'].add(mode)
        all_modes = set().union(*props.values())
        classified = {}
        for mode in sorted(all_modes):
            if mode in previous['geometry']:
                result = previous['geometry'][mode]
                counts['reused_classifications'] += 1
            else:
                result = geometry.classify_mode(x, v, mode)
                counts['new_classifications'] += 1
                counts['new_LP_calls'] += result['lp_calls']
                counts['new_geometry_repairs'] += len(result['geometry_repairs'])
            _, a, rhs, _, _ = geometry.matrices(x, v, mode)
            counts['certificate_rechecks'] += certify_label(a, rhs, result)
            counts[result['classification']] += 1
            classified[mode] = result
        methods = {n:describe(modes, classified, old_positive) for n,modes in props.items()}
        comparisons = {}
        for backend in ['exact', 'float']:
            unions = {f:set(methods[f+'_'+backend+'_union']['positive_modes']) for f in FAMILIES}
            for family in FAMILIES:
                positive = unions[family]
                continuous = set(previous['methods'][family+'_event_all']['positive_modes'])
                others = set().union(*(s for f,s in unions.items() if f!=family))
                comparisons[family+'_'+backend] = dict(
                    new_vs_original=sorted(positive-old_positive),
                    new_vs_original_and_own_continuous=sorted(positive-old_positive-continuous),
                    new_vs_original_and_all_308=sorted(positive-old_all),
                    exclusive_vs_all_308_and_other_masks=sorted(positive-old_all-others))
        record = dict(seed=seed, selected_states=len(by_task[seed])//3,
             original_positive_modes=sorted(old_positive), all_308_positive_modes=sorted(old_all),
             methods=methods, comparisons=comparisons, geometry=classified,
             x_observed=x.tolist(), v_observed=v.tolist())
        filename=f'{seed}_geometry.json'; save(out/filename, record); files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename],
                          selected_states=record['selected_states'],methods=methods,comparisons=comparisons))
        if (seed-5910000+1)%8==0:
            print(dict(tasks=seed-5910000+1,counts=dict(counts)), flush=True)
    aggregate = {}
    for n in NAMES:
        rr = [t['methods'][n] for t in tasks]
        aggregate[n] = dict(tasks_with_new_positive=sum(bool(r['new_positive_modes']) for r in rr),
             task_positive_pairs=sum(len(r['new_positive_modes']) for r in rr),
             unknown_pairs=sum(len(r['unknown_modes']) for r in rr),
             volume_missing_pairs=sum(len(r['positive_modes_without_numerical_volume']) for r in rr),
             numeric_new_volume=sum(r['known_new_positive_numeric_volume_sum'] for r in rr))
    comparison_aggregate = {}
    for name in tasks[0]['comparisons']:
        comparison_aggregate[name] = {field:dict(tasks=sum(bool(t['comparisons'][name][field]) for t in tasks),
             pairs=sum(len(t['comparisons'][name][field]) for t in tasks)) for field in tasks[0]['comparisons'][name]}
    for filename, value in [('tasks.json',tasks),('aggregate.json',aggregate),('comparisons.json',comparison_aggregate)]:
        save(out/filename,value); files[filename]=sha(out/filename)
    files['protocol.json']=sha(out/'protocol.json')
    final = dict(passed=True,tasks=64,states=131,methods=len(NAMES),counts=dict(counts),
        classifications_complete=counts['unresolved']==0,comparison_aggregate=comparison_aggregate,
        resources_matched=False,independent_task_gain_established=False,query_targets_accessed=False,
        seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final)
    print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()))
        raise

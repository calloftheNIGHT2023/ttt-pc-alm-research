"""321 independent exact region certificates and all six comparison pools."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
from audit_post_escape_geometry_v1 import certificates
from audit_stasis_escape_geometry_v1 import inequalities
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    start=time.perf_counter();folder=root/'results/branch_image_chain/geometry_v1';summary=read(folder/'summary.json');assert summary['passed']
    source=root/'results/branch_image_chain/development_v1';proposals=read(source/'summary.json');assert proposals['passed']
    assert sha(source/'before_geometry_manifest.json')==proposals['manifest_sha256']
    for name,digest in read(source/'before_geometry_manifest.json')['files_sha256'].items():assert sha(source/name)==digest
    for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    grouped=defaultdict(list)
    for row in read(source/'rows.json'):grouped[row['seed']].append(read(source/row['file']))
    counts=Counter();tasks=[]
    for row in read(folder/'tasks.json'):
        task=read(folder/row['file']);old=read(root/'results/factorized_dual_branch_search/geometry_v1'/row['file'])
        expected_prior=set(old['prior_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        assert sorted(expected_prior)==task['prior_positive_modes']
        props={n:set() for n in CHANNELS}
        for rec in grouped[task['seed']]:
            assert rec['x_observed']==task['x_observed'] and rec['v_observed']==task['v_observed']
            for n in CHANNELS:props[n].update(p['mode'] for p in rec['results'][n]['proposals'])
        assert set().union(*props.values())==set(task['geometry'])
        for mode,result in task['geometry'].items():
            a,rhs=inequalities(np.array(task['x_observed']),np.array(task['v_observed']),mode)
            counts['independent_certificates']+=certificates(a,rhs,result);counts[result['classification']]+=1
        positives={}
        for n in CHANNELS:
            m=task['methods'][n];assert sorted(props[n])==m['proposal_modes']
            positives[n]={mode for mode in props[n] if task['geometry'][mode]['positive_volume_certified']}
            assert sorted(positives[n])==m['positive_modes']
            assert sorted(positives[n]-expected_prior)==m['new_positive_modes'];counts['method_pools']+=1
            assert dict(Counter(task['geometry'][mode]['classification'] for mode in props[n]))==m['classifications']
        for n in CHANNELS:
            other=set().union(*(positives[k] for k in CHANNELS if k!=n))
            expected=dict(new_vs_prior=sorted(positives[n]-expected_prior),
                          new_vs_prior_and_other_channels=sorted(positives[n]-expected_prior-other),
                          new_vs_own_row_selector=sorted(positives[n]-set(old['methods'][n]['positive_modes'])))
            assert expected==task['comparisons'][n];counts['comparison_sets']+=3
        controls=set().union(*(positives[n] for n in ['residual','bp','random_sign','zero']))
        expected=(positives['dual']|positives['dual_plus_residual'])-expected_prior-controls
        assert sorted(expected)==task['combined_candidate_exclusive'];tasks.append(task)
    assert len(tasks)==64
    for n,stored in summary['aggregate'].items():
        expected=dict(proposal_pairs=sum(len(t['methods'][n]['proposal_modes']) for t in tasks),
                      positive_pairs=sum(len(t['methods'][n]['positive_modes']) for t in tasks),
                      new_vs_prior=sum(len(t['comparisons'][n]['new_vs_prior']) for t in tasks),
                      new_vs_prior_and_other_channels=sum(len(t['comparisons'][n]['new_vs_prior_and_other_channels']) for t in tasks),
                      new_vs_own_row_selector=sum(len(t['comparisons'][n]['new_vs_own_row_selector']) for t in tasks),
                      tasks_with_new_positive=sum(bool(t['comparisons'][n]['new_vs_prior']) for t in tasks))
        assert expected==stored;counts['aggregate_fields']+=len(expected)
    assert sum(len(t['combined_candidate_exclusive']) for t in tasks)==summary['combined_candidate_exclusive_pairs']
    result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-start,input_sha256=sha(folder/'summary.json'),
                source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_post_escape_geometry_v1.py','audit_stasis_escape_geometry_v1.py']},
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

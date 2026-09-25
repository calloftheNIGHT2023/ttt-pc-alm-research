"""324 independent region certificates, proposal pools and stronger old controls."""
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
POLICIES=['first_fit','uniform_state']
NAMES=[p+'/'+n for p in POLICIES for n in CHANNELS]


def run(root,out):
    start=time.perf_counter();base=root/'results/support_consistency_trigger';folder=base/'geometry_v1';source=base/'development_v1'
    summary=read(folder/'summary.json');ss=read(source/'summary.json');assert summary['passed'] and ss['passed']
    assert sha(source/'before_geometry_manifest.json')==ss['manifest_sha256']
    for n,d in read(source/'before_geometry_manifest.json')['files_sha256'].items():assert sha(source/n)==d
    for n,d in summary['outputs_sha256'].items():assert sha(folder/n)==d
    older=root/'results/novel_branch_continuation/development_v1';os=read(older/'summary.json');assert os['passed']
    for n,d in os['outputs_sha256'].items():assert sha(older/n)==d
    oldrows={r['seed']:r for r in read(older/'proposals.json')};scores=defaultdict(dict)
    for s in read(older/'scores.json'):
        if s['grid']==257:scores[s['seed']][s['method']]=s
    moment=root/'results/confirmation_conditional_risk/moments';ms=read(moment/'summary.json')
    assert sha(moment/'tasks.json')==ms['tasks_sha256'];reference={r['seed']:set(r['keys']) for r in read(moment/'tasks.json')}
    assert read(root/'results/confirmation_conditional_risk/reference_audit/summary.json')['complete_up_to_certified_zero_volume']==64
    grouped=defaultdict(list)
    for row in read(source/'rows.json'):grouped[row['seed']].append(read(source/row['file']))
    counts=Counter();alltasks=[]
    for row in read(folder/'tasks.json'):
        t=read(folder/row['file']);seed=t['seed'];old=read(root/'results/branch_image_chain/geometry_v1'/row['file'])
        prior321=set(old['prior_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        old_original=set(oldrows[seed]['old_positive_modes']);expected297={}
        for name,m in oldrows[seed]['methods'].items():
            pool=old_original|(set(m['proposal_modes'])&reference[seed]);expected297[name]=sorted(pool)
            assert pool==old_original|set(scores[seed][name]['new_positive_modes']);counts['old297_pools']+=1
        assert t['baseline_methods_297']==expected297
        prior=prior321|set().union(*(set(p) for p in expected297.values()))
        assert sorted(prior)==t['prior_positive_modes'] and prior<=reference[seed]
        assert sorted(prior321)==t['prior_321_positive_modes'] and sorted(prior-prior321)==t['prior_extra_297_modes']
        counts['prior_extra_297_pairs']+=len(prior-prior321)
        props={n:set() for n in NAMES}
        for rec in grouped[seed]:
            assert rec['x_observed']==t['x_observed'] and rec['v_observed']==t['v_observed']
            for name in CHANNELS:props[rec['policy']+'/'+name].update(p['mode'] for p in rec['results'][name]['proposals'])
        assert set().union(*props.values())==set(t['geometry'])
        for mode,value in t['geometry'].items():
            a,rhs=inequalities(np.array(t['x_observed']),np.array(t['v_observed']),mode)
            counts['independent_certificates']+=certificates(a,rhs,value);counts[value['classification']]+=1
        positives={}
        for name in NAMES:
            m=t['methods'][name];assert sorted(props[name])==m['proposal_modes']
            positives[name]={k for k in props[name] if t['geometry'][k]['positive_volume_certified']}
            assert sorted(positives[name])==m['positive_modes'] and sorted(positives[name]-prior)==m['new_positive_modes']
            assert positives[name]<=reference[seed];counts['method_pools']+=1
        controls=set().union(*(positives[p+'/'+n] for p in POLICIES for n in ['residual','bp','random_sign','zero']))
        for name in NAMES:
            policy,channel=name.split('/');other=('uniform_state' if policy=='first_fit' else 'first_fit')+'/'+channel
            expected=dict(new_vs_prior=sorted(positives[name]-prior),new_vs_prior_and_controls=sorted(positives[name]-prior-controls),
                          new_vs_other_trigger=sorted(positives[name]-positives[other]),new_vs_prior321=sorted(positives[name]-prior321))
            assert expected==t['comparisons'][name];counts['comparison_sets']+=4
        alltasks.append(t)
    for name,stored in summary['aggregate'].items():
        expected=dict(proposal_pairs=sum(len(t['methods'][name]['proposal_modes']) for t in alltasks),positive_pairs=sum(len(t['methods'][name]['positive_modes']) for t in alltasks),
            **{field:sum(len(t['comparisons'][name][field]) for t in alltasks) for field in ['new_vs_prior','new_vs_prior_and_controls','new_vs_other_trigger','new_vs_prior321']})
        assert stored==expected;counts['aggregate_fields']+=len(expected)
    assert counts['prior_extra_297_pairs']==summary['counts']['prior_extra_297_pairs']
    result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-start,input_summary_sha256=sha(folder/'summary.json'),
                source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_post_escape_geometry_v1.py','audit_stasis_escape_geometry_v1.py']},
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise

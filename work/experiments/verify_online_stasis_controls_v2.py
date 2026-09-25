"""306 functional trace equivalence followed by all 576 real untraced calls."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import online_stasis_controls_v2 as online
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import read,save,sha

PREFLIGHT=[5910000,5910001,5910008,5910016,5910032,5910048,5910053,5910063]


def identical(value, expected, label):
    assert value.shape==expected.shape and value.dtype==expected.dtype, label
    assert value.tobytes()==expected.tobytes(), label


def run(root,out):
    begin=time.perf_counter()
    gate=root/'results/online_stasis_resources/audit_v1'
    gs=read(gate/'summary.json')
    assert gs['passed'] and gs['methods']==38 and gs['failed_timing_calls']==0
    for n,d in gs['outputs_sha256'].items():assert sha(gate/n)==d
    functional=root/'results/online_stasis_memory/functional_v2'
    fs=read(functional/'summary.json');assert fs['passed'] and fs['tasks']==64
    for n,d in fs['outputs_sha256'].items():assert sha(functional/n)==d
    originals={r['seed']:r for r in read(functional/'rows.json')}
    source=root/'results/observable_trap_continuation/development_v1'
    seal=read(source/'before_reference_manifest.json')
    assert sha(source/'summary.json')=='4febdf01b4b979f163088e87ddc947fd259fa53b005cfb9ea44b99596bc01b1d'
    for n,d in seal['files_sha256'].items():assert sha(source/n)==d
    hashes=dict(read(root/'results/online_stasis_resources/calibration_v1/protocol.json')['source_sha256'])
    for n in ['online_stasis_controls_v1.py','verify_online_stasis_controls_v1.py','online_stasis_controls_v2.py','verify_online_stasis_controls_v2.py']:
        hashes[n]=sha(root/'work/experiments'/n)
    for n,d in hashes.items():assert sha(root/'work/experiments'/n)==d
    save(out/'protocol.json',dict(source_sha256=hashes,design_sha256=sha(root/'outputs/ttt-pc-alm-research/306_online_stasis_controls_protocol.md'),
        numerical_addendum_sha256=sha(root/'outputs/ttt-pc-alm-research/306_reduction_order_addendum.md'),
        resource_audit_summary_sha256=sha(gate/'summary.json'),families=online.FAMILIES,seeds=list(range(5910000,5910064)),
        preflight_seeds=PREFLIGHT,query_targets_accessed=False,read_posterior_pool_labels_only_after_full_seal=True,
        new_confirmation=False,resource_timing_comparison=False))
    counts=Counter();preflight=[];rows=[];files={}

    def call(seed,family,trace):
        with np.load(functional/originals[seed]['file'],allow_pickle=False) as z:
            x,v,q=z['x_observed'],z['v_observed'],z['q_observed']
        with discovery_box(.12):a,m=online.fit(x,v,q,seed,family=family,trace=trace)
        assert not m['query_targets_accessed'] and not m['archive_or_reference_access_in_fit']
        assert not m['selected_using_global_bp']
        assert m['global_bp_used']==(family.startswith('adam') and m['selected_states']>0)
        for k,value in a.items():
            if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),(seed,family,k)
        for k in ['prediction','point_prediction']:
            assert a[k].shape==(257,) and np.all((a[k]>=0)&(a[k]<=1))
        with np.load(functional/originals[seed]['file'],allow_pickle=False) as z:
            common=[k for k in a if k.startswith(('initial_','effective_','prefix_','anchor_','atomic_'))
                    or k in ['origins','assigned_actions','selected_b','best_bank','point_prediction']]
            if family=='alm_keep':common=[k for k in a if k in z.files]
            for k in common:identical(a[k],z[k],(seed,family,k,'unchanged'))
            counts['original_common_arrays']+=len(common)
        return a,m

    def archive(a,m,seed,family,kind):
        name=f'{kind}_{seed}_{family}.npz'
        previous=root/'results/online_stasis_controls/functional_v1'/name
        if previous.exists() and name!='preflight_5910001_alm_residual.npz':
            with np.load(previous,allow_pickle=False) as z:
                assert set(a)==set(z.files)
                for k,value in a.items():
                    identical(value,z[k],(seed,family,k,'previous_success'))
                    counts['previous_success_arrays']+=1
            counts['previous_success_calls']+=1
        with (out/name).open('xb') as stream:np.savez_compressed(stream,**a)
        files[name]=sha(out/name)
        return dict(seed=seed,method=family,kind=kind,file=name,sha256=files[name],metadata=m)

    for seed in PREFLIGHT:
        with np.load(source/f'{seed}_proposals.npz',allow_pickle=False) as z:
            mask=z['mask_forward_stasis']
            expected_common=dict(shadow_locations=z['locations'][mask],shadow_initial_b=z['initial_b'][mask],
                shadow_initial_best=z['initial_best'][mask],shadow_initial_h=z['initial_h'][:,mask],shadow_initial_u=z['initial_u'][:,mask])
            family_expected={}
            for family in online.FAMILIES:
                expected=dict(expected_common,shadow_b=z[family+'_b'][:,mask])
                if mask.any():
                    expected['shadow_effective_initial_b']=expected['shadow_b'][0]
                    if not family.startswith('adam'):
                        expected.update(shadow_effective_initial_u=z[family+'_initial_u'][:,mask],
                            shadow_final_h=z[family+'_final_h'][:,mask],shadow_final_u=z[family+'_final_u'][:,mask])
                family_expected[family]=expected
        for family in online.FAMILIES:
            a,m=call(seed,family,True)
            row=archive(a,m,seed,family,'preflight')
            # Preserve the returned arrays before checking frozen shadow identity.
            save(out/f'preflight_attempt_{seed}_{family}.json',row)
            for k,expected in family_expected[family].items():
                identical(a[k],expected,(seed,family,k,'shadow'))
                counts['shadow_arrays']+=1
            assert m['selected_states']==int(mask.sum())
            preflight.append(row)
        print(dict(phase='trace_preflight',tasks=len(preflight)//9,calls=len(preflight),total=72),flush=True)
    save(out/'preflight.json',preflight)
    preindex={(r['seed'],r['method']):r for r in preflight}
    for seed in range(5910000,5910064):
        for family in online.FAMILIES:
            a,m=call(seed,family,False)
            row=archive(a,m,seed,family,'full')
            save(out/f'full_attempt_{seed}_{family}.json',row)
            if (seed,family) in preindex:
                ref=preindex[seed,family]
                with np.load(out/ref['file'],allow_pickle=False) as z:
                    assert set(a).issubset(z.files)
                    for k,value in a.items():
                        identical(value,z[k],(seed,family,k,'trace_toggle'))
                        counts['trace_toggle_arrays']+=1
                assert m['positive_modes']==ref['metadata']['positive_modes']
            rows.append(row)
        if (seed-5910000+1)%8==0:
            print(dict(phase='full_no_trace',tasks=seed-5910000+1,calls=len(rows),total=576),flush=True)
    assert len(rows)==576 and len(preflight)==72
    assert sum(r['metadata']['selected_states'] for r in rows)==131*9
    save(out/'rows.json',rows)
    save(out/'checks.json',dict(counts=counts,passed=True))
    for n in ['protocol.json','preflight.json','rows.json','checks.json']:files[n]=sha(out/n)
    save(out/'before_pool_audit_manifest.json',dict(files_sha256=files,all_predictors_sealed=True,
        posterior_pool_labels_accessed=False,query_targets_accessed=False))
    print(dict(phase='all_real_predictors_sealed',calls=576),flush=True)
    scores={(r['seed'],r['method']):r for r in read(source/'scores.json') if r['grid']==257}
    old=root/'results/certificate_activity_attribution/development'
    old_rows={r['seed']:r for r in read(old/'rows.json') if r['method']=='credit_control_probe33'}
    poolchecks=[]
    for row in rows:
        seed,family=row['seed'],row['method']
        ref=scores[seed,f'forward_stasis__{family}_64']
        expected=set(old_rows[seed]['metadata']['positive_modes'])|set(ref['new_positive_modes'])
        assert set(row['metadata']['positive_modes'])==expected,(seed,family,'pool')
        poolchecks.append(dict(seed=seed,method=family,positive_modes=sorted(expected),passed=True))
    save(out/'pool_audit.json',poolchecks)
    for n,d in files.items():assert sha(out/n)==d
    for n,d in hashes.items():assert sha(root/'work/experiments'/n)==d
    summary=dict(passed=True,preflight_calls=72,full_calls=576,tasks=64,families=9,counts=dict(counts),
        matching_positive_pools=len(poolchecks),selected_states_all_families=131*9,
        geometry_repairs=sum(r['metadata']['geometry_repair_count'] for r in rows),
        query_targets_accessed=False,new_confirmation=False,resources_matched=False,core_research_goal_complete=False,
        seconds=time.perf_counter()-begin,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',required=True,type=Path)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_controls/functional_v2'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

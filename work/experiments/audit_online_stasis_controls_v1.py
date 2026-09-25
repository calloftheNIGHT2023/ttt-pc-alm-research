"""306 independent archive-level audit of complete online control outputs."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def same(left,right,label):
    assert left.shape==right.shape and left.dtype==right.dtype,label
    assert left.tobytes()==right.tobytes(),label


def run(root,out):
    folder=root/'results/online_stasis_controls/functional_v2'
    summary=read(folder/'summary.json');assert summary['passed'] and summary['full_calls']==576
    for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest,name
    p=read(folder/'protocol.json');assert p['seeds']==list(range(5910000,5910064))
    for name,digest in p['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest,name
    assert p['design_sha256']==sha(root/'outputs/ttt-pc-alm-research/306_online_stasis_controls_protocol.md')
    assert p['numerical_addendum_sha256']==sha(root/'outputs/ttt-pc-alm-research/306_reduction_order_addendum.md')
    seal=read(folder/'before_pool_audit_manifest.json')
    assert seal['all_predictors_sealed'] and not seal['query_targets_accessed'] and not seal['posterior_pool_labels_accessed']
    for name,digest in seal['files_sha256'].items():assert sha(folder/name)==digest,name
    rows=read(folder/'rows.json');preflight=read(folder/'preflight.json')
    assert len(rows)==576 and len(preflight)==72
    assert [(r['seed'],r['method']) for r in rows]==[(s,f) for s in p['seeds'] for f in p['families']]
    assert [(r['seed'],r['method']) for r in preflight]==[(s,f) for s in p['preflight_seeds'] for f in p['families']]
    baseline=root/'results/online_stasis_memory/functional_v2'
    source=root/'results/observable_trap_continuation/development_v1'
    original=root/'results/certificate_activity_attribution/development'
    base_rows={r['seed']:r for r in read(baseline/'rows.json')}
    old_rows={r['seed']:r for r in read(original/'rows.json') if r['method']=='credit_control_probe33'}
    shadow_files=read(source/'before_reference_manifest.json')['files_sha256']
    expected_scores={(r['seed'],r['method']):r for r in read(source/'scores.json') if r['grid']==257}
    pre={(r['seed'],r['method']):r for r in preflight};counts=Counter();details=[]
    for row in preflight+rows:
        seed,family=row['seed'],row['method'];label=(seed,family,row['kind'])
        assert row==read(folder/f'{row["kind"]}_attempt_{seed}_{family}.json')
        assert sha(folder/row['file'])==row['sha256']==seal['files_sha256'][row['file']]
        base=base_rows[seed];assert sha(baseline/base['file'])==base['sha256']
        m=row['metadata']
        assert not m['query_targets_accessed'] and not m['archive_or_reference_access_in_fit'] and not m['selected_using_global_bp']
        assert m['global_bp_used']==(family.startswith('adam') and m['selected_states']>0)
        assert m['shadow_state_steps']==64*m['selected_states']
        assert m['diagnostic_trace_enabled']==(row['kind']=='preflight')
        with np.load(folder/row['file'],allow_pickle=False) as a, np.load(baseline/base['file'],allow_pickle=False) as b:
            for key in a.files:
                value=a[key]
                if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),(label,key)
                if key in ['prediction','point_prediction']:
                    assert value.shape==(257,) and np.all((value>=0)&(value<=1))
                unchanged=key.startswith(('initial_','effective_','prefix_','anchor_','atomic_')) or key in ['origins','assigned_actions','selected_b','best_bank','point_prediction']
                if unchanged or (family=='alm_keep' and key in b.files):
                    same(value,b[key],(label,key,'original'))
                    counts['original_arrays']+=1
            if row['kind']=='preflight':
                fn=f'{seed}_proposals.npz';assert sha(source/fn)==shadow_files[fn]
                with np.load(source/fn,allow_pickle=False) as z:
                    mask=z['mask_forward_stasis'];assert m['selected_states']==int(mask.sum())
                    expected=dict(shadow_locations=z['locations'][mask],shadow_initial_b=z['initial_b'][mask],
                        shadow_initial_h=z['initial_h'][:,mask],shadow_initial_u=z['initial_u'][:,mask],
                        shadow_initial_best=z['initial_best'][mask],shadow_b=z[family+'_b'][:,mask])
                    if mask.any():
                        expected['shadow_effective_initial_b']=expected['shadow_b'][0]
                        if not family.startswith('adam'):
                            expected.update(shadow_effective_initial_u=z[family+'_initial_u'][:,mask],
                                shadow_final_h=z[family+'_final_h'][:,mask],shadow_final_u=z[family+'_final_u'][:,mask])
                    for key,value in expected.items():
                        same(a[key],value,(label,key,'shadow'))
                        counts['shadow_arrays']+=1
            elif (seed,family) in pre:
                with np.load(folder/pre[seed,family]['file'],allow_pickle=False) as z:
                    assert set(a.files).issubset(z.files)
                    for key in a.files:
                        same(a[key],z[key],(label,key,'trace_toggle'))
                        counts['trace_toggle_arrays']+=1
            if row['kind']=='full':
                expected=set(old_rows[seed]['metadata']['positive_modes'])|set(expected_scores[seed,f'forward_stasis__{family}_64']['new_positive_modes'])
                assert set(m['positive_modes'])==expected,(label,'pool')
                counts['matching_pools']+=1
                if family.startswith('adam') and m['selected_states']:
                    s=m['shadow_solver_metadata']
                    assert s['per_restart_global_jacobian_evaluations_total']==64*m['selected_states']
                    assert s['per_restart_forward_evaluations_total']==65*m['selected_states']
        counts[row['kind']+'_calls']+=1
        details.append(dict(seed=seed,family=family,kind=row['kind'],file=row['file'],sha256=row['sha256'],passed=True))
    assert counts['matching_pools']==576
    assert sum(r['metadata']['selected_states'] for r in rows)==1179
    assert summary['counts']['previous_success_calls']==15
    save(out/'details.json',details)
    result=dict(passed=True,counts=dict(counts),full_tasks=64,families=9,selected_states_all_families=1179,
        query_targets_accessed=False,new_quality_evaluation=False,core_research_goal_complete=False,
        functional_summary_sha256=sha(folder/'summary.json'),source_sha256={Path(__file__).name:sha(Path(__file__))},
        outputs_sha256={'details.json':sha(out/'details.json')})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_controls/audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

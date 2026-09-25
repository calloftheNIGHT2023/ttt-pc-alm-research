"""304 full-call support-only prototype gate, old64, no query-target evaluation."""
import argparse
from pathlib import Path
import time
import numpy as np
import online_stasis_memory_v1 as online
from posterior_confirmation_pipeline import discovery_box
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def run(root,out):
    begin=time.perf_counter()
    source=root/'results/observable_trap_continuation/development_v1'
    ss=read(source/'summary.json');assert ss['passed'] and ss['tasks']==64
    assert sha(source/'summary.json')=='4febdf01b4b979f163088e87ddc947fd259fa53b005cfb9ea44b99596bc01b1d'
    seal=read(source/'before_reference_manifest.json')
    for name,digest in seal['files_sha256'].items():assert sha(source/name)==digest,name
    props={r['seed']:r for r in read(source/'proposals.json')}
    raw=root/'results/certificate_activity_attribution/development'
    rm=read(raw/'before_evaluation_manifest.json');assert sha(raw/'rows.json')==rm['rows_sha256']
    old={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    frozen=read(root/'results/round_287_audit_v6.json')
    for n,d in frozen['source_sha256'].items():assert sha(root/'work/experiments'/n)==d,n
    names=['online_stasis_memory_v1.py','run_online_stasis_memory_v1.py','certificate_activity_attribution.py',
           'cold_stagnation_switch.py','posterior_confirmation_pipeline.py','diagnose_counterfactual_conditional_risk_v1.py']
    sources={n:sha(root/'work/experiments'/n) for n in names}
    save(out/'protocol.json',dict(source_sha256=sources,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/304_online_stasis_memory_protocol.md'),
        seeds=list(range(5910000,5910064)),compatibility_seeds=list(range(5910000,5910004)),
        shadow_source_manifest_sha256=sha(source/'before_reference_manifest.json'),
        no_query_targets=True,no_actual_risk_evaluation=True,trace_enabled=True,
        source_step='functional preflight, not resource matching or new-task confirmation'))
    rows=[];files={};compatibility=[];total_states=0;array_checks=shadow_checks=0
    for index,seed in enumerate(range(5910000,5910064)):
        path=raw/old[seed]['file'];assert sha(path)==old[seed]['sha256']
        with np.load(path,allow_pickle=False) as z:
            x,v,q=z['x_observed'],z['v_observed'],z['q_observed']
            originals={k:z[k] for k in z.files if k.startswith(('initial_','effective_','prefix_','anchor_','atomic_')) or
                k in ['origins','assigned_actions','selected_b','best_bank','point_prediction']}
            if seed<5910004:
                with discovery_box(.12):base,meta=online.fit(x,v,q,seed,rule='none',trace=True)
                for k,a in base.items():
                    assert a.shape==z[k].shape and a.dtype==z[k].dtype and a.tobytes()==z[k].tobytes(),(seed,k,'compatibility')
                compatibility.append(dict(seed=seed,arrays=len(base),metadata=meta))
        with discovery_box(.12):arrays,metadata=online.fit(x,v,q,seed,trace=True)
        for k,a in originals.items():
            assert k in arrays and a.shape==arrays[k].shape and a.dtype==arrays[k].dtype and a.tobytes()==arrays[k].tobytes(),(seed,k,'original_path')
            array_checks+=1
        assert sha(source/props[seed]['file'])==props[seed]['sha256']
        with np.load(source/props[seed]['file'],allow_pickle=False) as z:
            mask=z['mask_forward_stasis'];count=int(mask.sum())
            expected=dict(shadow_locations=z['locations'][mask],shadow_b=z['alm_keep_b'][:,mask],
                shadow_initial_b=z['initial_b'][mask],shadow_initial_best=z['initial_best'][mask],
                shadow_initial_h=z['initial_h'][:,mask],shadow_initial_u=z['initial_u'][:,mask])
            if count:
                expected.update(shadow_final_h=z['alm_keep_final_h'][:,mask],shadow_final_u=z['alm_keep_final_u'][:,mask])
        for k,a in expected.items():
            assert a.shape==arrays[k].shape and a.dtype==arrays[k].dtype and a.tobytes()==arrays[k].tobytes(),(seed,k,'shadow')
            shadow_checks+=1
        assert metadata['selected_states']==count and metadata['shadow_state_steps']==64*count
        assert not metadata['global_bp_used'] and not metadata['archive_or_reference_access_in_fit']
        total_states+=count
        arrays.update(x_observed=x,v_observed=v,q_observed=q)
        file=f'{seed}_online.npz'
        with (out/file).open('xb') as stream:np.savez_compressed(stream,**arrays)
        files[file]=sha(out/file)
        rows.append(dict(seed=seed,file=file,sha256=files[file],metadata=metadata,source_sha256=sha(path),
            original_arrays_checked=len(originals),shadow_arrays_checked=len(expected)))
        if (index+1)%8==0:print(dict(phase='full_fit_support_only',tasks=index+1,selected_states=total_states),flush=True)
    assert total_states==131 and len(rows)==64 and len(compatibility)==4
    save(out/'rows.json',rows);save(out/'compatibility.json',compatibility)
    for n in ['protocol.json','rows.json','compatibility.json']:files[n]=sha(out/n)
    save(out/'before_pool_audit_manifest.json',dict(files_sha256=files,query_targets_accessed=False,
        offline_pool_labels_accessed=False,all_predictors_sealed=True))
    print(dict(phase='all_predictors_sealed',tasks=64,selected_states=total_states),flush=True)
    # Only after real predictions have been sealed, compare offline positive pools.
    scores={r['seed']:r for r in read(source/'scores.json') if r['grid']==257 and r['method']=='forward_stasis__alm_keep_64'}
    poolchecks=[]
    for row in rows:
        seed=row['seed'];expected=set(old[seed]['metadata']['positive_modes'])|set(scores[seed]['new_positive_modes'])
        assert set(row['metadata']['positive_modes'])==expected,(seed,'positive_pool')
        poolchecks.append(dict(seed=seed,positive_modes=sorted(expected),passed=True))
    save(out/'pool_audit.json',poolchecks)
    for n,d in files.items():assert sha(out/n)==d,n
    for n,d in sources.items():assert sha(root/'work/experiments'/n)==d,n
    summary=dict(passed=True,tasks=64,compatibility_cases=4,compatibility_arrays=sum(r['arrays'] for r in compatibility),
        original_arrays_checked=array_checks,shadow_arrays_checked=shadow_checks,selected_states=131,shadow_state_steps=8384,
        matching_positive_pools=64,mean_trace_complete_seconds=float(np.mean([r['metadata']['wrapped_complete_seconds'] for r in rows])),
        mean_observer_seconds=float(np.mean([r['metadata']['observer_seconds'] for r in rows])),
        mean_shadow_solver_seconds=float(np.mean([r['metadata']['shadow_solver_seconds'] for r in rows])),
        max_extra_numeric_state_subtotal=max(r['metadata']['extra_numeric_state_subtotal'] for r in rows),
        query_targets_accessed=False,new_confirmation=False,resources_matched=False,core_research_goal_complete=False,
        seconds=time.perf_counter()-begin,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_memory/functional_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

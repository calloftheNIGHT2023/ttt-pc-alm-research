"""303: fixed support events, nine same-state continuation families, old64 diagnosis."""
import argparse
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import mode_list,residuals,tent
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha
from diagnose_stagnant_trigger_exploration_v1 import aggregate

GROUPS=['boundary_entry','forward_stasis','boundary_and_forward_stasis','primal_stasis']
FAMILIES=['alm_keep','alm_reset','nodual','pc','adam','alm_random_sign','alm_residual',
          'adam_perturb_001','adam_perturb_004']
HORIZONS=[8,32,64]
SEEDS=list(range(5910000,5910064))


def support_error(bank,x,v):
    h=np.broadcast_to(x,(len(bank),len(x)))
    for j in range(4):h=tent(h+bank[:,j,None])
    return np.max(abs(h-v),axis=1)


def new_local(b,h,u,best,x,v,method):
    state=cold.Local(b,x,v,method)
    state.h=h.copy();state.u=u.copy();state.best=best.copy()
    state.errors,state.moves=cold.base.score(best,x,v,np.zeros(4))
    return state


def run(root,out):
    started=time.perf_counter()
    events=root/'results/support_boundary_events/development_v1'
    assert sha(events/'summary.json')=='cf8f82aca2f716a24e37fdbd5a83068f3ae030d46890c9906cdbaced4ca9de2f'
    es=read(events/'summary.json');assert es['passed']
    # Verify but do not decode certificate labels, chronology or gradients.
    for name,digest in es['outputs_sha256'].items():assert sha(events/name)==digest,name
    selections={r['seed']:r for r in read(events/'selections.json')}
    raw=root/'results/certificate_activity_attribution/development'
    rm=read(raw/'before_evaluation_manifest.json')
    assert sha(raw/'rows.json')==rm['rows_sha256']
    assert sha(raw/'protocol.json')==rm['protocol_sha256']
    old={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    frozen=read(root/'results/round_287_audit_v6.json')
    for name,digest in frozen['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest,name
    names=['diagnose_observable_trap_continuation_v1.py','cold_stagnation_switch.py','batched_bp_discovery.py',
           'posterior_confirmation_pipeline.py','diagnose_gradient_flat_split_states_v1.py',
           'diagnose_counterfactual_conditional_risk_v1.py','diagnose_stagnant_trigger_exploration_v1.py']
    sources={n:sha(root/'work/experiments'/n) for n in names}
    docs=root/'outputs/ttt-pc-alm-research'
    save(out/'protocol.json',dict(seeds=SEEDS,groups=GROUPS,families=FAMILIES,horizons=HORIZONS,
        source_sha256=sources,design_sha256=sha(docs/'303_observable_trap_continuation_protocol.md'),
        execution_addendum_sha256=sha(docs/'303_execution_addendum_v1.md'),
        selection_summary_sha256=sha(events/'summary.json'),rng_seed_prefix=303911,
        random_fields=['prefix','seed','phase','step','origin','kind'],kind_multiplier_sign=0,kind_parameter_sign=1,
        perturbations=[.01,.04],box=.12,include_all_initial_proposal_modes=True,
        query_targets_accessed=False,certificate_labels_used_for_selection=False,
        reference_only_after_all_proposals_sealed=True,new_confirmation=False,resources_matched=False))
    files={};rows=[];checks=Counter();counts=Counter();empty=Counter();max_norm_gap=0.
    zero_u=zero_r=0
    for seed in SEEDS:
        sets={g:{tuple(a) for a in selections[seed]['locations'][g+'__all']} for g in GROUPS}
        union=set().union(*sets.values());loc=sorted(union);n=len(loc)
        masks={g:np.array([a in sets[g] for a in loc],dtype=bool) for g in GROUPS}
        assert np.all(np.any(np.stack(list(masks.values())),axis=0))
        assert not np.any(masks['boundary_and_forward_stasis']&~masks['forward_stasis'])
        counts['unique_union']+=n
        for g in GROUPS:counts[g]+=len(sets[g]);empty[g]+=int(not sets[g])
        source=raw/old[seed]['file'];assert sha(source)==old[seed]['sha256']==selections[seed]['source_sha256']
        with np.load(source,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            b=np.array([z[('prefix' if p==0 else 'anchor')+'_b'][t-1,o] for p,t,o in loc]).reshape(-1,4)
            best=np.array([z[('prefix' if p==0 else 'anchor')+'_best'][t-1,o] for p,t,o in loc]).reshape(-1,4)
            expected=np.array([z[('prefix' if p==0 else 'anchor')+'_b'][t,o] for p,t,o in loc]).reshape(-1,4)
            h=np.stack([z[('prefix' if p==0 else 'anchor')+'_h'][t-1,:,o] for p,t,o in loc],axis=1) if n else np.empty((4,0,len(x)))
            u=np.stack([z[('prefix' if p==0 else 'anchor')+'_u'][t-1,:,o] for p,t,o in loc],axis=1) if n else np.empty_like(h)
        init_tick=time.perf_counter()
        sign_u=np.stack([np.random.default_rng(np.random.SeedSequence([303911,seed,p,t,o,0])).choice([-1.,1.],size=(4,len(x)))
                         for p,t,o in loc],axis=1) if n else np.empty_like(u)
        sign_b=np.array([np.random.default_rng(np.random.SeedSequence([303911,seed,p,t,o,1])).choice([-1.,1.],size=4)
                         for p,t,o in loc]).reshape(-1,4)
        residual=residuals(b,h,x)
        unorm=np.sqrt(np.sum(u*u,axis=(0,2)));rnorm=np.sqrt(np.sum(residual*residual,axis=(0,2)))
        scale=np.divide(unorm,rnorm,out=np.zeros_like(unorm),where=(unorm>0)&(rnorm>0))
        ur=residual*scale[None,:,None];us=u*sign_u
        zero_u+=int(np.count_nonzero(unorm==0));zero_r+=int(np.count_nonzero(rnorm==0))
        assert np.array_equal(abs(us),abs(u))
        if n:
            valid=(unorm>0)&(rnorm>0)
            if np.any(valid):max_norm_gap=max(max_norm_gap,float(np.max((abs(np.sqrt(np.sum(ur*ur,axis=(0,2)))-unorm)/np.maximum(1,unorm))[valid])))
        assert max_norm_gap<1e-12
        bstarts={f:b.copy() for f in FAMILIES}
        bstarts['adam_perturb_001']=np.clip(b+.01*sign_b,-.12,.12)
        bstarts['adam_perturb_004']=np.clip(b+.04*sign_b,-.12,.12)
        init_seconds=time.perf_counter()-init_tick
        paths={};first_h={};final_h={};final_u={};initial_u={};timings={};solver_states={}
        with discovery_box(.12):
            for family in FAMILIES:
                tick=time.perf_counter()
                if family.startswith('adam'):
                    if n:
                        _,trace,meta=cold.run_bp(bstarts[family],x,v,'adam',64,trace=True)
                        assert trace['b'].shape==(65,n,4) and trace['roles'].tolist()==[True]*64+[False]
                        assert meta['per_restart_global_jacobian_evaluations_total']==64*n
                        assert meta['per_restart_forward_evaluations_total']==65*n
                        paths[family]=trace['b']
                        solver_states[family]=meta
                    else:paths[family]=np.empty((65,0,4));solver_states[family]=dict(empty=True)
                    timings[family]=dict(full64_trace_seconds=time.perf_counter()-tick,step_seconds=None)
                else:
                    method='alm' if family.startswith('alm_') else family
                    startu=u if family=='alm_keep' else us if family=='alm_random_sign' else ur if family=='alm_residual' else np.zeros_like(u)
                    initial_u[family]=startu.copy()
                    bb=[b.copy()];elapsed=[]
                    if n:state=new_local(b,h,startu,best,x,v,method)
                    setup_seconds=time.perf_counter()-tick
                    for t in range(64):
                        step_tick=time.perf_counter()
                        if n:
                            state.step();bb.append(state.b.copy())
                            if t==0:first_h[family]=state.h.copy()
                        else:
                            bb.append(b.copy())
                            if t==0:first_h[family]=h.copy()
                        elapsed.append(time.perf_counter()-step_tick)
                    paths[family]=np.array(bb)
                    final_h[family]=state.h.copy() if n else h.copy()
                    final_u[family]=state.u.copy() if n else startu.copy()
                    solver_states[family]=dict(named_live_state_bytes=state.numeric_state_bytes() if n else 0,
                        excludes_temporaries_and_trace_archive=True)
                    timings[family]=dict(setup_seconds=setup_seconds,step_seconds=elapsed,
                        full64_trace_seconds=time.perf_counter()-tick)
                assert np.array_equal(paths[family][0],bstarts[family])
                assert np.all(np.isfinite(paths[family])) and np.all(abs(paths[family])<=.12)
                checks['valid_initial_and_box']+=1
        assert paths['alm_keep'][1].tobytes()==expected.tobytes(),(seed,'keep1')
        assert paths['alm_reset'][1].tobytes()==paths['nodual'][1].tobytes(),(seed,'reset_nodual_b')
        assert first_h['alm_reset'].tobytes()==first_h['nodual'].tobytes(),(seed,'reset_nodual_h')
        for name in ['keep1','reset_nodual_b','reset_nodual_h']:checks[name]+=1
        errors={f:support_error(a.reshape(-1,4),x,v).reshape(65,n) if n else np.empty((65,0)) for f,a in paths.items()}
        patterns={f:np.array(mode_list(a.reshape(-1,4),x)).reshape(65,n) if n else np.empty((65,0),dtype=str) for f,a in paths.items()}
        methods={}
        for group,mask in masks.items():
            selected_count=int(mask.sum())
            for family in FAMILIES:
                previous_modes=set()
                for horizon in HORIZONS:
                    # Include the initial point, especially the perturbed BP point.
                    proposals=sorted(set(patterns[family][:horizon+1,mask].ravel().tolist()))
                    assert previous_modes<=set(proposals)
                    assert set(patterns[family][0,mask].tolist())<=set(proposals)
                    previous_modes=set(proposals);checks['pool_prefix_and_initial']+=1
                    methods[f'{group}__{family}_{horizon}']=dict(group=group,family=family,horizon=horizon,
                        modes=proposals,shadow_steps=selected_count*horizon,initial_states=selected_count,
                        visited_point_rows=selected_count*(horizon+1),
                        global_jacobian_rows=selected_count*horizon if family.startswith('adam') else 0,
                        parameter_perturbation= .01 if family=='adam_perturb_001' else .04 if family=='adam_perturb_004' else 0.)
        assert len(methods)==108
        for family in FAMILIES:
            for horizon in HORIZONS:
                left=set(methods[f'boundary_and_forward_stasis__{family}_{horizon}']['modes'])
                right=set(methods[f'forward_stasis__{family}_{horizon}']['modes'])
                assert left<=right;checks['nested_selection_pool']+=1
        arrays=dict(locations=np.array(loc,dtype=int).reshape(-1,3),initial_b=b,initial_h=h,initial_u=u,
                    initial_best=best,x_observed=x,v_observed=v,parameter_sign=sign_b,multiplier_sign=sign_u,
                    residual_direction=ur)
        arrays.update({f'mask_{g}':m for g,m in masks.items()})
        arrays.update({f'{f}_b':a for f,a in paths.items()})
        arrays.update({f'{f}_support_max_error':a for f,a in errors.items()})
        for name,mapping in [('initial_u',initial_u),('first_h',first_h),('final_h',final_h),('final_u',final_u)]:
            arrays.update({f'{f}_{name}':a for f,a in mapping.items()})
        assert all(np.all(np.isfinite(a)) for a in arrays.values())
        file=f'{seed}_proposals.npz'
        with (out/file).open('xb') as stream:np.savez_compressed(stream,**arrays)
        files[file]=sha(out/file)
        rows.append(dict(seed=seed,locations=loc,counts={g:len(sets[g]) for g in GROUPS},unique_states=n,
            source_sha256=sha(source),original_positive_modes=old[seed]['metadata']['positive_modes'],
            methods=methods,file=file,sha256=files[file],shared_direction_initialization_seconds=init_seconds,
            timings_unique_union_batch=timings,solver_state_subtotals=solver_states,
            diagnostic_archive_uncompressed_bytes=sum(a.nbytes for a in arrays.values()),
            diagnostic_support_forward_rows=9*65*n,full_call_resources_matched=False))
        if (seed-SEEDS[0]+1)%8==0:print(dict(phase='support_only',tasks=seed-SEEDS[0]+1,unique_states=counts['unique_union']),flush=True)
    assert [counts[g] for g in GROUPS]==[2984,131,117,31]
    save(out/'proposals.json',rows)
    save(out/'selftests.json',dict(passed=True,checks=dict(checks),counts=dict(counts),empty_tasks=dict(empty),
        zero_multiplier_states=zero_u,zero_residual_states=zero_r,max_norm_gap=max_norm_gap))
    for name in ['protocol.json','proposals.json','selftests.json']:files[name]=sha(out/name)
    save(out/'before_reference_manifest.json',dict(files_sha256=files,query_targets_accessed=False,
        posterior_reference_accessed=False,certificate_labels_used=False,tasks=64,new_configs=108))
    print(dict(phase='all_support_proposals_sealed',counts=dict(counts),seconds=time.perf_counter()-started),flush=True)

    cached=root/'results/counterfactual_conditional_risk/development_v2'
    assert sha(cached/'summary.json')=='95852bf236596935f7ac0dd3c6fe389118b6976b5b2f368e09cc8ac37325a3ec'
    cs=read(cached/'summary.json');assert cs['passed']
    for name,digest in cs['outputs_sha256'].items():assert sha(cached/name)==digest,name
    for name,digest in read(cached/'input_hashes.json').items():assert sha(root/name)==digest,name
    moments=root/'results/confirmation_conditional_risk/moments'
    tasks={r['seed']:r for r in read(moments/'tasks.json')}
    scores=[];baselines=[];max_scalar_gap=0.
    for row in rows:
        task=tasks[row['seed']];assert sha(moments/task['file'])==task['sha256']
        with np.load(moments/task['file'],allow_pickle=False) as z:
            volumes,means=z['volumes'],z['means'];assert np.array_equal(z['q'],np.linspace(0,1,257))
        assert np.all(volumes>0) and means.shape==(len(volumes),4,257)
        keys=task['keys'];keyset=set(keys);weights=volumes/volumes.sum()
        full=np.einsum('k,kbq->bq',weights,means)
        oldset=set(row['original_positive_modes']);assert oldset and oldset<=keyset
        def mixture(pool):
            mask=np.array([k in pool for k in keys]);mass=float(weights[mask].sum());assert mass>0
            return mass,np.einsum('k,kbq->bq',weights*mask/mass,means)
        oldmass,oldmu=mixture(oldset)
        for grid in [257,129]:
            idx=np.arange(257) if grid==257 else np.arange(0,257,2)
            pairs=[float(np.mean((oldmu[a,idx]-full[a,idx])*(oldmu[b,idx]-full[b,idx]))) for a,b in [(0,1),(2,3)]]
            baselines.append(dict(seed=row['seed'],grid=grid,mass=oldmass,policy_excess=float(np.mean(pairs)),pairs=pairs))
        for name,method in row['methods'].items():
            extra=(set(method['modes'])&keyset)-oldset;mass,mu=mixture(oldset|extra)
            for grid in [257,129]:
                idx=np.arange(257) if grid==257 else np.arange(0,257,2);estimates=[]
                for a,b in [(0,1),(2,3)]:
                    bias=float(np.mean((mu[a,idx]-full[a,idx])*(mu[b,idx]-full[b,idx])))
                    base=float(np.mean((oldmu[a,idx]-full[a,idx])*(oldmu[b,idx]-full[b,idx])))
                    scalar=math.fsum(float(c)*float(d) for c,d in zip(mu[a,idx]-full[a,idx],mu[b,idx]-full[b,idx]))/grid
                    max_scalar_gap=max(max_scalar_gap,abs(bias-scalar))
                    estimates.append(dict(pair=[a,b],policy_excess=bias,delta=bias-base))
                scores.append(dict(seed=row['seed'],method=name,scope='303_fixed_support_events',grid=grid,
                    mass=mass,mass_added=mass-oldmass,new_positive_modes=sorted(extra),shadow_steps=method['shadow_steps'],
                    initial_states=method['initial_states'],global_jacobian_rows=method['global_jacobian_rows'],
                    pairs=estimates,policy_excess=float(np.mean([e['policy_excess'] for e in estimates])),
                    delta=float(np.mean([e['delta'] for e in estimates]))))
                if not method['initial_states']:assert not extra and scores[-1]['delta']==0
    assert len(scores)==13824 and max_scalar_gap<1e-12
    aggregates=aggregate(scores);assert len(aggregates)==216
    save(out/'scores.json',scores);save(out/'aggregates.json',aggregates);save(out/'original_pool.json',baselines)
    for name,digest in files.items():assert sha(out/name)==digest,name
    for name,digest in sources.items():assert sha(root/'work/experiments'/name)==digest,name
    summary=dict(passed=True,tasks=64,configs=108,grid_rows=len(scores),counts=dict(counts),empty_tasks=dict(empty),
        checks=dict(checks),max_scalar_gap=max_scalar_gap,max_norm_gap=max_norm_gap,
        seconds=time.perf_counter()-started,query_targets_accessed=False,
        reference_only_after_all_proposals_sealed=True,certificate_labels_used=False,
        resources_matched=False,new_confirmation=False,core_research_goal_complete=False,
        cached_reference_summary_sha256=sha(cached/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/observable_trap_continuation/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

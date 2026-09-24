"""298: fixed dual/residual/sign amplitude exploration; OLD64 development only."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import math
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import mode_list
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha
from dual_amplitude_local_audit_v1 import audit_state

SEEDS=list(range(5910000,5910064))
ALPHAS=[-1.,-.5,0.,.5,1.,1.5,2.]
FAMILIES=['dual','residual','random_sign']


def make_state(b,h,u,best,x,v):
    state=cold.Local(b,x,v,'alm')
    state.h=h.copy()
    state.u=u.copy()
    state.best=best.copy()
    state.errors,state.moves=cold.base.score(best,x,v,np.zeros(4))
    return state


def run(root,out):
    started=time.perf_counter()
    fg=root/'results/counterfactual_credit_branching/first_global_support_v1'
    fs=read(fg/'summary.json')
    assert fs['passed'] and fs['tasks']==64
    for name,digest in fs['outputs_sha256'].items():
        assert sha(fg/name)==digest,name
    frows={r['seed']:r for r in read(fg/'rows.json')}
    raw=root/'results/certificate_activity_attribution/development'
    oldrows={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    frozen=read(root/'results/round_287_audit_v6.json')
    for name,digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name)==digest,name
    source_names=['diagnose_dual_amplitude_paths_v1.py','dual_amplitude_local_audit_v1.py',
                  'cold_stagnation_switch.py','diagnose_counterfactual_conditional_risk_v1.py']
    sources={n:sha(root/'work/experiments'/n) for n in source_names}
    docs=root/'outputs/ttt-pc-alm-research'
    save(out/'protocol.json',dict(seeds=SEEDS,alphas=ALPHAS,families=FAMILIES,
        random_seed_prefix=298911,per_state_random_seed_fields=['prefix','seed','phase','step','origin'],
        source_sha256=sources,design_sha256={n:sha(docs/n) for n in
        ['298_dual_amplitude_path_design.md','298_execution_protocol_addendum.md']},
        first_global_summary_sha256=sha(fg/'summary.json'),frozen_seal_sha256=sha(root/'results/round_287_audit_v6.json'),
        no_query_targets=True,reference_only_after_all_proposals_sealed=True,
        grids=[257,129],batch_pairs=[[0,1],[2,3]],resources_matched=False,new_confirmation=False))
    rows,mathrows=[],[]
    files={}
    max_norm=0.
    zero_u=zero_r=selected_states=endpoint_cases=0
    for seed in SEEDS:
        source=raw/oldrows[seed]['file']
        assert sha(source)==oldrows[seed]['sha256']==frows[seed]['source_file_sha256']
        with np.load(fg/frows[seed]['file'],allow_pickle=False) as z:
            loc=z['counterfactual_locations'].tolist()
            expected_zero=z['counterfactual_b']
        with np.load(source,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            b=np.array([z[('prefix' if p==0 else 'anchor')+'_b'][t-1,o] for p,t,o in loc])
            h=np.stack([z[('prefix' if p==0 else 'anchor')+'_h'][t-1,:,o] for p,t,o in loc],axis=1)
            u=np.stack([z[('prefix' if p==0 else 'anchor')+'_u'][t-1,:,o] for p,t,o in loc],axis=1)
            best=np.array([z[('prefix' if p==0 else 'anchor')+'_best'][t-1,o] for p,t,o in loc])
            expected_one=np.array([z[('prefix' if p==0 else 'anchor')+'_b'][t,o] for p,t,o in loc])
        selected_states+=len(b)
        residual=np.array([h[j]-cold.base.g((x if j==0 else h[j-1])+b[:,j,None]) for j in range(4)])
        unorm=np.sqrt(np.sum(u*u,axis=(0,2)))
        rnorm=np.sqrt(np.sum(residual*residual,axis=(0,2)))
        zero_u+=int(np.count_nonzero(unorm==0))
        zero_r+=int(np.count_nonzero(rnorm==0))
        scale=np.divide(unorm,rnorm,out=np.zeros_like(unorm),where=(rnorm>0)&(unorm>0))
        signs=np.stack([np.random.default_rng(np.random.SeedSequence([298911,seed,p,t,o])).choice([-1.,1.],size=(4,len(x)))
                        for p,t,o in loc],axis=1)
        directions={'dual':u,'residual':residual*scale[None,:,None],'random_sign':u*signs}
        assert np.array_equal(abs(directions['random_sign']),abs(u))
        assert np.all(directions['residual'][:,(rnorm==0)|(unorm==0)]==0)
        valid=(rnorm>0)&(unorm>0)
        gap=abs(np.sqrt(np.sum(directions['residual']**2,axis=(0,2)))-unorm)/np.maximum(1,unorm)
        if np.any(valid): max_norm=max(max_norm,float(np.max(gap[valid])))
        assert max_norm<1e-12
        paths,hpaths,methods,timings={},{},{},{}
        with discovery_box(.12):
            for family in FAMILIES:
                bb,hh=[],[]
                family_modes=set()
                family_seconds=0.
                for ai,alpha in enumerate(ALPHAS):
                    tick=time.perf_counter()
                    state=make_state(b,h,alpha*directions[family],best,x,v)
                    state.step()
                    seconds=time.perf_counter()-tick
                    bb.append(state.b.copy())
                    hh.append(state.h.copy())
                    modes=set(mode_list(state.b,x))
                    family_modes.update(modes)
                    family_seconds+=seconds
                    key=f'{family}_a{ai}'
                    methods[key]=dict(family=family,alpha=alpha,union=False,modes=sorted(modes),
                        shadow_steps=len(b),shadow_call_seconds=seconds)
                paths[family]=np.array(bb)
                hpaths[family]=np.array(hh)
                methods[family+'_union']=dict(family=family,alpha=None,union=True,modes=sorted(family_modes),
                    shadow_steps=len(b)*7,shadow_call_seconds=family_seconds)
                timings[family]=family_seconds
            for family in FAMILIES:
                assert paths[family][2].tobytes()==expected_zero.tobytes(),(seed,family,'alpha0')
                assert hpaths[family][2].tobytes()==hpaths['dual'][2].tobytes()
                endpoint_cases+=1
            assert paths['dual'][4].tobytes()==expected_one.tobytes(),(seed,'alpha1')
            endpoint_cases+=1
            def production(alpha):
                state=make_state(b[:1],h[:,:1],alpha*u[:,:1],best[:1],x,v)
                state.step()
                return state.b[0],state.h[:,0]
            mathrows.append(dict(seed=seed,location=loc[0],**audit_state(b[0],h[:,0],u[:,0],x,v,production)))
        file=f'{seed}_proposals.npz'
        with (out/file).open('xb') as stream:
            np.savez_compressed(stream,**{f'{k}_b':paths[k] for k in FAMILIES},
                **{f'{k}_h':hpaths[k] for k in FAMILIES},**{f'{k}_direction':directions[k] for k in FAMILIES})
        files[file]=sha(out/file)
        rows.append(dict(seed=seed,locations=loc,selected_states=len(b),methods=methods,
            source_sha256=oldrows[seed]['sha256'],original_positive_modes=oldrows[seed]['metadata']['positive_modes'],
            file=file,sha256=files[file],timings=timings))
        if (seed-SEEDS[0]+1)%16==0:
            print(dict(phase='support_only',tasks=seed-SEEDS[0]+1),flush=True)
    assert selected_states==2427 and endpoint_cases==256
    narrow=[w for r in mathrows for w in r['narrow'] if w['same_internal_branch_signature']]
    wide=[r for r in mathrows if r['wide_internal_signature_changes'] and r['wide_midpoint_gap']>1e-8]
    assert narrow and wide
    mathsummary=dict(passed=True,states=64,endpoint_cases=endpoint_cases,narrow_attempts=192,
        fixed_signature_cases=len(narrow),signature_changed_narrow_cases=192-len(narrow),
        max_affine_gap=max(r['max_affine_gap'] for r in narrow),
        max_independent_scalar_gap=max(r['max_scalar_gap'] for r in mathrows),
        wide_nonaffine_cases=len(wide),max_wide_midpoint_gap=max(r['wide_midpoint_gap'] for r in wide),
        max_normalization_gap=max_norm,zero_multiplier_states=zero_u,zero_residual_states=zero_r)
    save(out/'proposals.json',rows)
    save(out/'math_rows.json',mathrows)
    save(out/'math_summary.json',mathsummary)
    for name in ['protocol.json','proposals.json','math_rows.json','math_summary.json']:
        files[name]=sha(out/name)
    save(out/'before_reference_manifest.json',dict(files_sha256=files,reference_accessed=False,
        query_targets_accessed=False,tasks=64,new_configs=24))
    print(dict(phase='all_proposals_sealed',math=mathsummary),flush=True)

    cached=root/'results/counterfactual_conditional_risk/development_v2'
    cs=read(cached/'summary.json')
    assert cs['passed']
    for name,digest in cs['outputs_sha256'].items(): assert sha(cached/name)==digest,name
    for name,digest in read(cached/'input_hashes.json').items(): assert sha(root/name)==digest,name
    earlier=root/'results/novel_branch_continuation/development_v1'
    es=read(earlier/'summary.json')
    assert es['passed'] and es['methods']==20
    for name,digest in es['outputs_sha256'].items(): assert sha(earlier/name)==digest,name
    controls={r['seed']:r for r in read(earlier/'proposals.json')}
    prop=root/'results/counterfactual_branch_proposals/development_v1'
    modechange={r['seed']:r for r in read(prop/'tasks.json')}
    mom=root/'results/confirmation_conditional_risk/moments'
    tasks={r['seed']:r for r in read(mom/'tasks.json')}
    scores=[]
    for row in rows:
        seed=row['seed']
        task=tasks[seed]
        assert sha(mom/task['file'])==task['sha256']
        with np.load(mom/task['file'],allow_pickle=False) as z:
            volumes,means=z['volumes'],z['means']
        keys=task['keys']
        weights=volumes/volumes.sum()
        full=np.einsum('k,kbq->bq',weights,means)
        old=set(row['original_positive_modes'])
        mask=np.array([k in old for k in keys])
        massold=float(weights[mask].sum())
        muold=np.einsum('k,kbq->bq',weights*mask/massold,means)
        methods={name:dict(modes=r['modes'],shadow_steps=r['shadow_steps'],scope='new_amplitude')
                 for name,r in row['methods'].items()}
        methods.update({name:dict(modes=r['proposal_modes'],shadow_steps=r['proposal_state_steps'],scope='297_control')
                        for name,r in controls[seed]['methods'].items()})
        methods['original_alm']=dict(modes=[],shadow_steps=0,scope='original')
        methods['modechange']=dict(modes=modechange[seed]['rules']['changed_forward_mode']['new_positive_modes'],
            shadow_steps=modechange[seed]['rules']['changed_forward_mode']['proposals'],scope='292_pool')
        assert len(methods)==46
        for name,method in methods.items():
            extra=(set(method['modes'])&set(keys))-old
            pool=old|extra
            mask=np.array([k in pool for k in keys])
            mass=float(weights[mask].sum())
            mu=np.einsum('k,kbq->bq',weights*mask/mass,means)
            for grid in [257,129]:
                ii=np.arange(257) if grid==257 else np.arange(0,257,2)
                estimates=[]
                for a,b in [(0,1),(2,3)]:
                    bias=float(np.mean((mu[a,ii]-full[a,ii])*(mu[b,ii]-full[b,ii])))
                    base=float(np.mean((muold[a,ii]-full[a,ii])*(muold[b,ii]-full[b,ii])))
                    scalar=math.fsum(float(c)*float(d) for c,d in zip(mu[a,ii]-full[a,ii],mu[b,ii]-full[b,ii]))/grid
                    assert abs(bias-scalar)<1e-12
                    estimates.append(dict(pair=[a,b],policy_excess=bias,delta=bias-base))
                scores.append(dict(seed=seed,method=name,scope=method['scope'],grid=grid,mass=mass,mass_added=mass-massold,
                    new_positive_modes=sorted(extra),shadow_steps=method['shadow_steps'],pairs=estimates,
                    policy_excess=float(np.mean([e['policy_excess'] for e in estimates])),
                    delta=float(np.mean([e['delta'] for e in estimates]))))
    assert len(scores)==5888
    grouped=defaultdict(list)
    for r in scores: grouped[(r['method'],r['grid'])].append(r)
    aggregates=[]
    for (method,grid),group in grouped.items():
        assert len(group)==64
        aggregates.append(dict(method=method,grid=grid,scope=group[0]['scope'],tasks=64,
            policy_excess=float(np.mean([r['policy_excess'] for r in group])),
            delta=float(np.mean([r['delta'] for r in group])),
            pair_delta_means=[float(np.mean([r['pairs'][j]['delta'] for r in group])) for j in range(2)],
            mass=float(np.mean([r['mass'] for r in group])),
            shadow_steps=float(np.mean([r['shadow_steps'] for r in group])),
            new_positive_modes=sum(len(r['new_positive_modes']) for r in group),
            tasks_with_new_positive_modes=sum(bool(r['new_positive_modes']) for r in group)))
    # Every previous control must replay its frozen ideal-pool risk.
    lookup={(r['method'],r['grid']):r for r in aggregates}
    max_replay=0.
    for r in read(earlier/'aggregates.json'):
        rr=lookup[r['method'],r['grid']]
        for metric in ['policy_excess','delta','mass']:
            max_replay=max(max_replay,abs(rr[metric]-r[metric]))
    assert max_replay<1e-12
    save(out/'scores.json',scores)
    save(out/'aggregates.json',aggregates)
    for name,digest in files.items(): assert sha(out/name)==digest,name
    for name,digest in sources.items(): assert sha(root/'work/experiments'/name)==digest,name
    summary=dict(passed=True,tasks=64,configs=46,new_configs=24,grid_rows=len(scores),
        math=mathsummary,max_previous_control_replay_gap=max_replay,seconds=time.perf_counter()-started,
        query_targets_accessed=False,reference_only_after_all_proposals_sealed=True,
        resources_matched=False,new_confirmation=False,core_research_goal_complete=False,
        cached_diagnosis_sha256=sha(cached/'summary.json'),continuation_summary_sha256=sha(earlier/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    out=root/'results/dual_amplitude_paths/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()

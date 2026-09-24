"""297: seal support-only shadow paths, then OLD64 offline ideal-pool diagnosis."""
import argparse
from pathlib import Path
import time
import math
from collections import defaultdict
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import mode_list
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha

FAMILIES=['alm_reset','alm_keep','nodual','pc','adam']
HORIZONS=[1,2,4,8]
SEEDS=list(range(5910000,5910064))


def run(root,out):
    start=time.perf_counter()
    functional=root/'results/counterfactual_credit_branching/first_global_support_v1'
    fs=read(functional/'summary.json')
    assert fs['passed'] and fs['tasks']==64
    for name,digest in fs['outputs_sha256'].items():
        assert sha(functional/name)==digest,name
    firstrows={r['seed']:r for r in read(functional/'rows.json')}
    raw=root/'results/certificate_activity_attribution/development'
    oldrows={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    frozen=read(root/'results/round_287_audit_v6.json')
    for name,digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name)==digest,name
    sources={name:sha(root/'work/experiments'/name) for name in
             ['diagnose_novel_branch_continuation_v1.py','cold_stagnation_switch.py',
              'batched_bp_discovery.py','diagnose_counterfactual_conditional_risk_v1.py']}
    save(out/'protocol.json',dict(seeds=SEEDS,families=FAMILIES,horizons=HORIZONS,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/297_novel_branch_continuation_protocol.md'),
        source_sha256=sources,first_global_gate_sha256=sha(functional/'summary.json'),
        frozen_dependency_seal_sha256=sha(root/'results/round_287_audit_v6.json'),
        query_targets_accessed=False,new_confirmation=False,resources_matched=False))
    proposal_rows=[]
    files={}
    total_selected=0
    selftest_cases=0
    # Phase 1: neither reference directory nor posterior moment files are read.
    for seed in SEEDS:
        source=raw/oldrows[seed]['file']
        assert sha(source)==oldrows[seed]['sha256']==firstrows[seed]['source_file_sha256']
        firstpath=functional/firstrows[seed]['file']
        with np.load(firstpath,allow_pickle=False) as z:
            locations=z['counterfactual_locations'].tolist()
            expected_reset=z['counterfactual_b']
        with np.load(source,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            b=np.array([z[('prefix' if phase==0 else 'anchor')+'_b'][step-1,origin]
                        for phase,step,origin in locations])
            h=np.stack([z[('prefix' if phase==0 else 'anchor')+'_h'][step-1,:,origin]
                        for phase,step,origin in locations],axis=1)
            u=np.stack([z[('prefix' if phase==0 else 'anchor')+'_u'][step-1,:,origin]
                        for phase,step,origin in locations],axis=1)
            best=np.array([z[('prefix' if phase==0 else 'anchor')+'_best'][step-1,origin]
                           for phase,step,origin in locations])
            expected_keep=np.array([z[('prefix' if phase==0 else 'anchor')+'_b'][step,origin]
                                   for phase,step,origin in locations])
        assert len(b)>0
        total_selected+=len(b)
        traces,step_times,first_h={},{},{}
        with discovery_box(.12):
            for family in FAMILIES[:-1]:
                method='alm' if family.startswith('alm') else family
                state=cold.Local(b,x,v,method)
                state.h=h.copy()
                state.best=best.copy()
                state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
                state.u=u.copy() if family=='alm_keep' else np.zeros_like(u)
                path,elapsed=[],[]
                for step in range(8):
                    tick=time.perf_counter()
                    state.step()
                    elapsed.append(time.perf_counter()-tick)
                    path.append(state.b.copy())
                    if step==0:
                        first_h[family]=state.h.copy()
                traces[family]=np.array(path)
                step_times[family]=elapsed
            # This global-gradient baseline never initializes candidate states.
            tick=time.perf_counter()
            _,bp_trace,bp_meta=cold.run_bp(b,x,v,'adam',8,trace=True)
            bp_seconds=time.perf_counter()-tick
            assert bp_trace['b'].shape==(9,len(b),4)
            assert bp_trace['roles'].tolist()==[True]*8+[False]
            traces['adam']=bp_trace['b'][1:]
        assert traces['alm_reset'][0].tobytes()==expected_reset.tobytes(),(seed,'reset')
        assert traces['alm_keep'][0].tobytes()==expected_keep.tobytes(),(seed,'keep')
        assert traces['alm_reset'][0].tobytes()==traces['nodual'][0].tobytes(),(seed,'nodual_b')
        assert first_h['alm_reset'].tobytes()==first_h['nodual'].tobytes(),(seed,'nodual_h')
        selftest_cases+=4
        methods={}
        for family in FAMILIES:
            for horizon in HORIZONS:
                key=f'{family}_{horizon}'
                modes=sorted(set(mode_list(traces[family][:horizon].reshape(-1,4),x)))
                methods[key]=dict(family=family,horizon=horizon,proposal_modes=modes,
                    proposal_state_steps=len(b)*horizon,
                    local_step_seconds=math.fsum(step_times[family][:horizon]) if family!='adam' else None,
                    bp_full8_trace_seconds=bp_seconds if family=='adam' else None,
                    global_jacobian_rows=len(b)*horizon if family=='adam' else 0)
        file=f'{seed}_proposals.npz'
        with (out/file).open('xb') as stream:
            np.savez_compressed(stream,**traces)
        files[file]=sha(out/file)
        proposal_rows.append(dict(seed=seed,selected_states=len(b),source_sha256=oldrows[seed]['sha256'],
            old_positive_modes=oldrows[seed]['metadata']['positive_modes'],methods=methods,file=file,sha256=files[file]))
        if (seed-SEEDS[0]+1)%16==0:
            print(dict(phase='support_only',tasks=seed-SEEDS[0]+1),flush=True)
    assert total_selected==2427 and selftest_cases==256
    save(out/'proposals.json',proposal_rows)
    files['proposals.json']=sha(out/'proposals.json')
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_reference_manifest.json',dict(files_sha256=files,
        selftest_cases=selftest_cases,selected_states=total_selected,
        reference_accessed=False,query_targets_accessed=False))
    print(dict(phase='all_proposals_sealed',tasks=64,query_targets_accessed=False),flush=True)

    # Phase 2: expensive full posterior is diagnostic only, never a selector.
    cached=root/'results/counterfactual_conditional_risk/development_v2'
    cs=read(cached/'summary.json')
    assert cs['passed'] and cs['tasks']==64
    for name,digest in cs['outputs_sha256'].items():
        assert sha(cached/name)==digest,name
    for name,digest in read(cached/'input_hashes.json').items():
        assert sha(root/name)==digest,name
    mom=root/'results/confirmation_conditional_risk/moments'
    tasks={r['seed']:r for r in read(mom/'tasks.json')}
    scores=[]
    for r in proposal_rows:
        seed=r['seed']
        task=tasks[seed]
        assert sha(mom/task['file'])==task['sha256']
        with np.load(mom/task['file'],allow_pickle=False) as z:
            vol,means=z['volumes'],z['means']
        weights=vol/vol.sum()
        keys=task['keys']
        old=set(r['old_positive_modes'])
        assert old and old<=set(keys)
        full=np.einsum('k,kbq->bq',weights,means)
        def mixture(pool):
            mask=np.array([k in pool for k in keys])
            mass=float(weights[mask].sum())
            return mass,np.einsum('k,kbq->bq',weights*mask/mass,means)
        oldmass,oldmu=mixture(old)
        for name,method in r['methods'].items():
            selected=set(method['proposal_modes'])&set(keys)
            pool=old|selected
            mass,mu=mixture(pool)
            for grid in [257,129]:
                idx=np.arange(257) if grid==257 else np.arange(0,257,2)
                estimates=[]
                for a,b in [(0,1),(2,3)]:
                    bias=float(np.mean((mu[a,idx]-full[a,idx])*(mu[b,idx]-full[b,idx])))
                    base=float(np.mean((oldmu[a,idx]-full[a,idx])*(oldmu[b,idx]-full[b,idx])))
                    estimates.append(dict(pair=[a,b],policy_excess=bias,delta=bias-base))
                scores.append(dict(seed=seed,method=name,grid=grid,mass=mass,mass_added=mass-oldmass,
                    new_positive_modes=sorted(selected-old),proposal_state_steps=method['proposal_state_steps'],
                    pairs=estimates,policy_excess=float(np.mean([p['policy_excess'] for p in estimates])),
                    delta=float(np.mean([p['delta'] for p in estimates]))))
    grouped=defaultdict(list)
    for r in scores:
        grouped[(r['method'],r['grid'])].append(r)
    aggregate=[]
    for (method,grid),group in grouped.items():
        assert len(group)==64
        aggregate.append(dict(method=method,grid=grid,tasks=64,
            policy_excess=float(np.mean([r['policy_excess'] for r in group])),
            delta=float(np.mean([r['delta'] for r in group])),
            pair_delta_means=[float(np.mean([r['pairs'][j]['delta'] for r in group])) for j in range(2)],
            mass=float(np.mean([r['mass'] for r in group])),
            extra_state_steps=float(np.mean([r['proposal_state_steps'] for r in group])),
            new_positive_modes=sum(len(r['new_positive_modes']) for r in group),
            tasks_with_new_positive_modes=sum(bool(r['new_positive_modes']) for r in group)))
    save(out/'scores.json',scores)
    save(out/'aggregates.json',aggregate)
    for name,digest in files.items():
        assert sha(out/name)==digest,name
    for name,digest in sources.items():
        assert sha(root/'work/experiments'/name)==digest,name
    summary=dict(passed=True,tasks=64,methods=20,selftest_cases=selftest_cases,selected_states=total_selected,
        ideal_pool_grid_rows=len(scores),query_targets_accessed=False,reference_after_proposals_sealed=True,
        numerical_reference_only=True,resources_matched=False,core_research_goal_complete=False,
        seconds=time.perf_counter()-start,cached_diagnosis_sha256=sha(cached/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    out=root/'results/novel_branch_continuation/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()

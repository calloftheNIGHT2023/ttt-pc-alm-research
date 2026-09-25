"""310: full local replay then all old/current b,h,u counterfactual combinations.

Selected events are every support-certified first-arrival from the census,
not events chosen by query risk. A missing target mode is not a proof that
the alternative mode is infeasible. This script does not fit a new method.
"""
import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import time
import traceback

import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from diagnose_observable_trap_continuation_v1 import new_local
from diagnose_gradient_flat_split_states_v1 import mode_list
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha
from complete_credit_amplitude_events_v2 import rational_state
from complete_credit_rational_reference_v1 import direct_step

FAMILIES=['alm_keep','alm_reset','nodual','pc']
MASKS=list(product((0,1),repeat=3))


def compare_one_event(old, current, event, x, v, method, best):
    bb=np.array([current['b'] if m[0] else old['b'] for m in MASKS])
    hh=np.stack([current['h'] if m[1] else old['h'] for m in MASKS],axis=1)
    uu=np.stack([current['u'] if m[2] else old['u'] for m in MASKS],axis=1)
    run=new_local(bb,hh,uu,np.repeat(best[None],8,axis=0),x,v,method)
    run.step()
    modes=mode_list(run.b,x)
    outputs=[]
    for i,mask in enumerate(MASKS):
        reference=None
        if method!='pc':
            ss=rational_state(bb[i],hh[:,i],uu[:,i],x,v)
            rr=direct_step(ss,1)
            gap=max(float(np.max(abs(np.array(rr['b'],dtype=float)-run.b[i]))),
                    float(np.max(abs(np.array(rr['h'],dtype=float)-run.h[:,i]))))
            reference=dict(b=rr['b'],h=rr['h'],mode=bytes(map(int,rr['mode'])).hex(),max_float_gap=gap)
        outputs.append(dict(mask=''.join(map(str,mask)),float_b=run.b[i].tolist(),
            float_h=run.h[:,i].tolist(),float_u_after=run.u[:,i].tolist(),float_mode=modes[i],
            reaches_same_certified_target=modes[i]==event['mode'],exact_reference=reference))
    return outputs


def run(root,out):
    begin=time.perf_counter()
    census=root/'results/successful_transition_census/development_v2'
    summary=read(census/'summary.json');assert summary['passed']
    for n,digest in summary['outputs_sha256'].items():assert sha(census/n)==digest
    sources=set(read(census/'protocol.json')['source_sha256'])|{
        Path(__file__).name,'complete_credit_rational_reference_v1.py','complete_credit_amplitude_events_v2.py'}
    for n,digest in read(census/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    all_events=read(census/'state_events.json')
    events=[e for e in all_events if e['family'] in FAMILIES]
    assert all(e['first_step']>0 for e in events)
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(sources)},
        census_summary_sha256=sha(census/'summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/310_successful_transition_census_protocol.md'),
        masks=MASKS,mask_order=['b','h','u'],families=FAMILIES,selected_events=len(events),
        query_targets_accessed=False,posterior_moments_accessed=False,
        online_deployable=False,resources_matched=False,
        scope='All original local trajectories replayed; counterfactual intervention only at every first successful state-mode event.'))
    old_folder=root/'results/observable_trap_continuation/development_v1'
    rows=read(old_folder/'proposals.json');selected=defaultdict(list)
    for e in events:selected[(e['seed'],e['family'],e['first_step'])].append(e)
    checks=Counter();records=[];files={};guarded=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP/optimizer invoked in local replay')
    try:
        for obj,n in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guarded.append((obj,n,getattr(obj,n)));setattr(obj,n,forbid)
        with discovery_box(.12):
            for archive in rows:
                seed=archive['seed'];assert sha(old_folder/archive['file'])==archive['sha256']
                with np.load(old_folder/archive['file'],allow_pickle=False) as z:
                    select=z['mask_forward_stasis'];locs=z['locations'][select]
                    if not len(locs):continue
                    x,v=z['x_observed'],z['v_observed']
                    b,h,best=z['initial_b'][select],z['initial_h'][:,select],z['initial_best'][select]
                    lookup={tuple(int(q) for q in loc):i for i,loc in enumerate(locs)}
                    for family in FAMILIES:
                        method='alm' if family.startswith('alm_') else family
                        u=z[family+'_initial_u'][:,select]
                        state=new_local(b,h,u,best,x,v,method)
                        for step in range(1,65):
                            current_events=selected[(seed,family,step)]
                            for event in current_events:
                                i=lookup[tuple(event['location'])]
                                before=dict(b=state.b[i].copy(),h=state.h[:,i].copy(),u=state.u[:,i].copy())
                                initial=dict(b=b[i].copy(),h=h[:,i].copy(),u=u[:,i].copy())
                                proposals=compare_one_event(initial,before,event,x,v,method,best[i])
                                assert np.array_equal(proposals[0]['float_b'],z[family+'_b'][1,select][i])
                                assert np.array_equal(proposals[7]['float_b'],z[family+'_b'][step,select][i])
                                assert proposals[7]['reaches_same_certified_target']
                                counts=[p['reaches_same_certified_target'] for p in proposals]
                                minimum=[p['mask'] for p in proposals if p['reaches_same_certified_target'] and not any(
                                    q['reaches_same_certified_target'] and q['mask']!=p['mask'] and
                                    all(int(a)<=int(c) for a,c in zip(q['mask'],p['mask'])) for q in proposals)]
                                filename=f"{seed}_{family}_{'_'.join(map(str,event['location']))}_{step}_{event['mode']}.json"
                                payload=dict(event=event,x_observed=x.tolist(),v_observed=v.tolist(),
                                    initial={k:a.tolist() for k,a in initial.items()},
                                    current={k:a.tolist() for k,a in before.items()},
                                    proposals=proposals,minimal_successful_block_sets=minimum,
                                    old_best=best[i].tolist(),method=method)
                                save(out/filename,payload);files[filename]=sha(out/filename)
                                records.append(dict(**event,file=filename,sha256=files[filename],
                                    minimal_successful_block_sets=minimum,
                                    successful_masks=[p['mask'] for p in proposals if p['reaches_same_certified_target']],
                                    max_exact_reference_gap=max((p['exact_reference']['max_float_gap'] for p in proposals if p['exact_reference']),default=None),
                                    exact_float_mode_discrepancies=sum(p['exact_reference'] is not None and p['exact_reference']['mode']!=p['float_mode'] for p in proposals)))
                                checks['factorial_interventions']+=8;checks['event_endpoint_rechecks']+=2
                            state.step()
                            assert np.array_equal(state.b,z[family+'_b'][step,select]),(seed,family,step)
                            checks['individual_parameter_state_replays']+=len(locs)
                            if step==1:
                                assert np.array_equal(state.h,z[family+'_first_h'][:,select])
                                checks['first_activity_arrays']+=1
                        assert np.array_equal(state.h,z[family+'_final_h'][:,select])
                        assert np.array_equal(state.u,z[family+'_final_u'][:,select])
                        checks['final_activity_and_dual_arrays']+=2
    finally:
        for obj,n,value in guarded:setattr(obj,n,value)
    assert len(records)==len(events)
    save(out/'events.json',records);files['events.json']=sha(out/'events.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_interpretation_manifest.json',dict(files_sha256=files,query_targets_accessed=False,
        all_selected_events_processed=True,scope='Support-only target-mode identities; no alternative geometry scored.'))
    aggregate={}
    for family in FAMILIES:
        rr=[r for r in records if r['family']==family]
        aggregate[family]=dict(events=len(rr),tasks=len({r['seed'] for r in rr}),
            task_modes=len({(r['seed'],r['mode']) for r in rr}),
            minimal_sets=dict(Counter(s for r in rr for s in r['minimal_successful_block_sets'])),
            successes_by_mask={''.join(map(str,m)):sum(''.join(map(str,m)) in r['successful_masks'] for r in rr) for m in MASKS})
    save(out/'aggregate.json',aggregate);files['aggregate.json']=sha(out/'aggregate.json')
    final=dict(passed=True,events=len(events),checks=dict(checks),aggregate=aggregate,
        max_exact_reference_gap=max(r['max_exact_reference_gap'] for r in records if r['max_exact_reference_gap'] is not None),
        exact_float_mode_discrepancies=sum(r['exact_float_mode_discrepancies'] for r in records),
        runtime_no_global_bp_guard_passed=True,query_targets_accessed=False,resources_matched=False,
        online_deployable=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()))
        raise

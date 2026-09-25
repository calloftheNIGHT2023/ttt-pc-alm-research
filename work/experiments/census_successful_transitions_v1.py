"""310: read-only first-arrival census of all seven stored continuation families."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha
from diagnose_gradient_flat_split_states_v1 import mode_list

FAMILIES=['alm_keep','alm_reset','nodual','pc','adam','adam_perturb_001','adam_perturb_004']


def run(root,out):
    begin=time.perf_counter()
    old=root/'results/observable_trap_continuation/development_v1'
    geo=root/'results/complete_credit_amplitude_events/geometry_v1'
    support=root/'results/complete_credit_amplitude_events/development_v2'
    os=read(old/'summary.json');gs=read(geo/'summary.json');ss=read(support/'summary.json')
    assert os['passed'] and gs['execution_passed'] and ss['support_execution_passed']
    for folder,summary in [(old,os),(geo,gs),(support,ss)]:
        for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    oldrows={r['seed']:r for r in read(old/'proposals.json')}
    tasks={r['seed']:r for r in read(geo/'tasks.json')}
    state_rows={(r['seed'],*r['location']):r for r in read(geo/'states.json')}
    sources={Path(__file__).name, 'diagnose_gradient_flat_split_states_v1.py'}
    for p in [old/'protocol.json',geo/'protocol.json',support/'protocol.json']:
        for n,digest in read(p)['source_sha256'].items():
            assert sha(root/'work/experiments'/n)==digest
            sources.add(n)
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(sources)},
        input_summaries_sha256={str(p.relative_to(root)):sha(p/'summary.json') for p in [old,geo,support]},
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/310_successful_transition_census_protocol.md'),
        families=FAMILIES,states=131,tasks=64,steps_including_initial=65,
        query_targets_accessed=False,posterior_moments_accessed=False,
        new_optimization_performed=False,resources_matched=False))
    events=[];firsts=[];checks=Counter();empty_tasks=[]
    for seed in sorted(tasks):
        data=read(geo/tasks[seed]['file']);original=set(data['original_positive_modes'])
        positive={m for m,g in data['geometry'].items() if g['positive_volume_certified']}
        row=oldrows[seed]
        with np.load(old/row['file'],allow_pickle=False) as z:
            selected=z['mask_forward_stasis'];locations=z['locations'][selected]
            x,v=z['x_observed'],z['v_observed']
            assert np.array_equal(x,data['x_observed']) and np.array_equal(v,data['v_observed'])
            if len(locations)==0:empty_tasks.append(seed)
            family_first={}
            for family in FAMILIES:
                bank=z[family+'_b'][:,selected]
                modes=np.array(mode_list(bank.reshape(-1,4),x)).reshape(65,len(locations)) if len(locations) else np.empty((65,0),dtype=str)
                for k,location in enumerate(locations):
                    loc=tuple(int(n) for n in location);known=state_rows[(seed,*loc)]
                    expected=set(known['methods']['control_'+family+'_64']['proposal_modes'])
                    assert set(modes[:,k])==expected,(seed,loc,family)
                    checks['exact_same_state_mode_pool_checks']+=1
                    assert all(m in data['geometry'] for m in modes[:,k])
                    seen=set()
                    for step,mode in enumerate(modes[:,k]):
                        checks['visited_state_rows']+=1
                        if mode in seen:continue
                        seen.add(mode)
                        if mode not in positive or mode in original:continue
                        event=dict(seed=seed,family=family,location=loc,first_step=step,mode=str(mode),
                            support_max_error=float(z[family+'_support_max_error'][step,np.flatnonzero(selected)[k]]),
                            initial_perturbation_event=step==0)
                        events.append(event)
                        pair=(family,str(mode))
                        if pair not in family_first or (step,loc)<(family_first[pair]['first_step'],tuple(family_first[pair]['location'])):
                            family_first[pair]=event
                checks['family_trajectories']+=len(locations)
            firsts.extend(family_first.values())
    aggregate={}
    for family in FAMILIES:
        rr=[e for e in events if e['family']==family]
        ff=[e for e in firsts if e['family']==family]
        aggregate[family]=dict(tasks=len({e['seed'] for e in ff}),task_mode_pairs=len(ff),
            state_mode_events=len(rr),first_step_histogram=dict(sorted(Counter(e['first_step'] for e in ff).items())),
            step_zero_pairs=sum(e['first_step']==0 for e in ff),
            step_one_pairs=sum(e['first_step']==1 for e in ff),
            later_pairs=sum(e['first_step']>1 for e in ff),
            first_parameters_in_support_band_pairs=sum(e['support_max_error']<=.001 for e in ff))
        assert len(ff)==sum(len(t['methods']['control_'+family+'_64']['new_positive_modes']) for t in tasks.values())
    assert checks['family_trajectories']==131*7
    for name,value in [('state_events.json',events),('first_task_mode_events.json',firsts),('aggregate.json',aggregate)]:save(out/name,value)
    result=dict(passed=True,checks=dict(checks),selected_states=131,tasks=64,
        tasks_without_selected_states=empty_tasks,aggregate=aggregate,
        query_targets_accessed=False,new_optimization_performed=False,resources_matched=False,
        seconds=time.perf_counter()-begin,
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k not in ['outputs_sha256','tasks_without_selected_states']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)

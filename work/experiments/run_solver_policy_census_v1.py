"""313 all-state policy census; no query truth, no policy-based optimization."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from solver_policy_trace_v1 import step,segments
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

FAMILIES=['alm_keep','alm_reset','nodual']


def run(root,out):
    begin=time.perf_counter();old=root/'results/observable_trap_continuation/development_v1'
    summary=read(old/'summary.json');assert summary['passed'] and summary['counts']['forward_stasis']==131
    for n,digest in summary['outputs_sha256'].items():assert sha(old/n)==digest
    tests=root/'results/solver_policy_recurrence/tests_v1/summary.json';tested=read(tests)
    assert tested['passed'] and not tested['real_task_data_accessed']
    prior=read(old/'protocol.json');names={Path(__file__).name,'evaluate_complete_credit_mode_geometry_v1.py'}
    for p in [prior,tested]:
        for n,digest in p['source_sha256'].items():
            assert sha(root/'work/experiments'/n)==digest;names.add(n)
    docs=root/'outputs/ttt-pc-alm-research'
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
        old_summary_sha256=sha(old/'summary.json'),test_summary_sha256=sha(tests),
        design_sha256={n:sha(docs/n) for n in ['313_solver_policy_recurrence_design.md','313_execution_protocol_v1.md']},
        tasks=64,states=131,families=FAMILIES,steps=64,query_targets_accessed=False,
        new_optimization_performed=False,success_event_labels_accessed=False,resources_matched=False,
        policy_scope='Conservative observed floating execution signature; not minimal map identity or interval proof.'))
    files,rows,trajectories,empty,counts={},[],[],[],Counter();schema=None;guarded=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP/optimizer invoked during local policy replay')
    try:
        for obj,n in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guarded.append((obj,n,getattr(obj,n)));setattr(obj,n,forbid)
        with discovery_box(.12):
            oldrows=read(old/'proposals.json');assert len(oldrows)==64
            for oldrow in oldrows:
                seed=oldrow['seed'];assert sha(old/oldrow['file'])==oldrow['sha256']
                with np.load(old/oldrow['file'],allow_pickle=False) as z:
                    selected=z['mask_forward_stasis'];locations=z['locations'][selected];r=len(locations)
                    if not r:empty.append(seed);continue
                    x,v=z['x_observed'],z['v_observed']
                    for family in FAMILIES:
                        method='alm' if family.startswith('alm_') else 'nodual'
                        b=z['initial_b'][selected].copy();h=z['initial_h'][:,selected].copy()
                        u=z[family+'_initial_u'][:,selected].copy()
                        original=cold.Local(b,x,v,method);original.h=h.copy();original.u=u.copy()
                        bh=[b.copy()];hh=[h.copy()];uh=[u.copy()];policy={};forward=[]
                        for t in range(1,65):
                            result=step(b,h,u,x,v,method);original.step()
                            for k in ['b','h','u']:
                                assert np.array_equal(result[k],getattr(original,k)),(seed,family,t,k)
                                counts['traced_original_individual_array_checks']+=r
                            b,h,u=result['b'],result['h'],result['u']
                            assert np.array_equal(b,z[family+'_b'][t,selected]),(seed,family,t,'archive')
                            counts['archived_individual_parameter_states']+=r
                            if t==1:
                                assert np.array_equal(h,z[family+'_first_h'][:,selected]);counts['first_activity_arrays']+=1
                            if schema is None:schema=result['schema']
                            else:assert schema==result['schema']
                            for group,a in result['policies'].items():policy.setdefault(group,[]).append(a)
                            bh.append(b.copy());hh.append(h.copy());uh.append(u.copy());forward.append(result['forward_policy'])
                        for k,a in [('h',h),('u',u)]:
                            assert np.array_equal(a,z[family+'_final_'+k][:,selected]);counts['final_activity_dual_arrays']+=1
                        arrays={g:np.stack(a) for g,a in policy.items()}
                        full=np.concatenate(list(arrays.values()),axis=2);fw=np.stack(forward)
                        for i,loc in enumerate(locations):
                            trajectories.append(dict(seed=seed,family=family,location=loc.tolist(),
                                full_segments=segments(full[:,i]),forward_segments=segments(fw[:,i]),
                                group_segments={g:segments(a[:,i]) for g,a in arrays.items()},
                                full_policy_width=full.shape[2]))
                        filename=f'{seed}_{family}.npz'
                        np.savez_compressed(out/filename,b=np.stack(bh),h=np.stack(hh),u=np.stack(uh),
                            forward_policy=fw,locations=locations,x_observed=x,v_observed=v,
                            **{'policy_'+g:a for g,a in arrays.items()})
                        files[filename]=sha(out/filename)
                        rows.append(dict(seed=seed,family=family,method=method,file=filename,sha256=files[filename],
                            states=r,policy_width=full.shape[2],archive_bytes=(out/filename).stat().st_size))
                    print(dict(seed=seed,trajectories=len(trajectories),archived_states=counts['archived_individual_parameter_states']),flush=True)
    finally:
        for obj,n,value in guarded:setattr(obj,n,value)
    assert len(trajectories)==393 and counts['archived_individual_parameter_states']==25152 and len(empty)==45
    for n,value in [('rows.json',rows),('trajectories.json',trajectories),('policy_schema.json',schema)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_event_analysis_manifest.json',dict(files_sha256=files,states=131,tasks=64,trajectories=393,
        all_selected_states_processed=True,query_targets_accessed=False,success_event_labels_accessed=False))
    final=dict(passed=True,tasks=64,states=131,trajectories=393,steps=64,counts=dict(counts),
        empty_tasks=empty,runtime_no_global_bp_guard_passed=True,resources_matched=False,
        query_targets_accessed=False,success_event_labels_accessed=False,
        independent_task_gain_established=False,seconds=time.perf_counter()-begin,
        manifest_sha256=sha(out/'before_event_analysis_manifest.json'),outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k not in ['outputs_sha256','empty_tasks']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

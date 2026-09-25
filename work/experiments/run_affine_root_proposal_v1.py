"""316 current-state-only root proposals, exact drift witnesses and validation."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from affine_root_certificate_v1 import propose, system, dot
from affine_root_validation_v1 import validate, state, forward_metrics
from effective_affine_map_v1 import build, pack, dump, evaluate
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha


def run(root,out):
    begin=time.perf_counter();source=root/'results/solver_policy_recurrence/development_v1'
    tested=read(root/'results/affine_root_proposal/tests_v1/summary.json');assert tested['passed']
    audited=root/'results/effective_affine_map/audit_v1/summary.json';assert read(audited)['passed']
    oldsummary=read(source/'summary.json');assert oldsummary['passed']
    names={Path(__file__).name,'evaluate_complete_credit_mode_geometry_v1.py'}
    for document in [tested,read(source/'protocol.json')]:
        for name,digest in document['source_sha256'].items():
            assert sha(root/'work/experiments'/name)==digest;names.add(name)
    for name,digest in oldsummary['outputs_sha256'].items():assert sha(source/name)==digest
    design=root/'outputs/ttt-pc-alm-research/316_affine_root_proposal_protocol_v1.md'
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
                  tests_sha256=sha(root/'results/affine_root_proposal/tests_v1/summary.json'),
                  previous_audit_sha256=sha(audited),old_summary_sha256=sha(source/'summary.json'),
                  design_sha256=sha(design),states=131,cases=393,tasks=64,empty_tasks=oldsummary['empty_tasks'],
                  information='Only current input, observed x/v and one current local sweep; no future policy maps.',
                  free_coordinate_rule='Retain current values in nonpivot columns; ascending exact pivots.',
                  nodual_rule='Eliminate all u coordinates at zero before solving.',
                  query_targets_accessed=False,success_event_labels_accessed=False,resources_matched=False)
    save(out/'protocol.json',protocol)
    records=[];total=Counter();seconds=Counter();files={}
    with discovery_box(.12):
        for old in read(source/'rows.json'):
            with np.load(source/old['file'],allow_pickle=False) as z:
                # No future iteration or policy contributes to the candidate.
                x,v=z['x_observed'],z['v_observed'];locations=z['locations']
                bs,hs,us=z['b'][0],z['h'][0],z['u'][0]
                next_b,next_h,next_u=z['b'][1],z['h'][1],z['u'][1]
            d,n=bs.shape[-1],len(x);dim=d+d*n;method=old['method']
            if old['family'] in ['alm_reset','nodual']:assert not np.any(us)
            for i,location in enumerate(locations):
                timer=Counter();tick=time.perf_counter()
                current=step(bs[i:i+1],hs[:,i:i+1],us[:,i:i+1],x,v,method)
                timer['current_local_sweep']=time.perf_counter()-tick
                assert np.array_equal(current['b'][0],next_b[i])
                assert np.array_equal(current['h'][:,0],next_h[:,i])
                assert np.array_equal(current['u'][:,0],next_u[:,i])
                policies={g:a[0] for g,a in current['policies'].items()}
                tick=time.perf_counter();rows=build(policies,current['schema'],x,v,method)
                timer['formula_construction']=time.perf_counter()-tick
                point=pack(bs[i],hs[:,i],us[:,i]);predicted=evaluate(rows,point)
                actual=pack(next_b[i],next_h[:,i],next_u[:,i])
                gap=max(float(abs(a-b)) for a,b in zip(predicted,actual));assert gap<=1e-10
                active=None if method=='alm' else list(range(dim))
                fixed=None if method=='alm' else {j:F(0) for j in range(dim,len(rows))}
                tick=time.perf_counter();solved=propose(rows,point,active,fixed)
                timer['linear_solve_and_certificate_checks']=time.perf_counter()-tick
                before=forward_metrics(state(point,d,x,v))
                record=dict(seed=old['seed'],family=old['family'],method=method,location=location.tolist(),
                            x=list(map(F,x)),v=list(map(F,v)),point=point,formula=dump(rows),solution=solved,
                            original_metrics=before,current_replay_max_gap=gap)
                if solved['consistent']:
                    candidate=solved['full_root'];assert evaluate(rows,candidate)==candidate
                    tick=time.perf_counter();checked=validate(candidate,d,x,v,method)
                    timer['independent_exact_root_validation']=time.perf_counter()-tick
                    record['validation']=checked
                    record['root_distance_max']=max(abs(a-b) for a,b in zip(candidate,point))
                    record['bias_changed']=candidate[:d]!=point[:d]
                    if checked['domain_valid']:
                        tick=time.perf_counter();floatpoint=np.array(candidate,float)
                        fs=step(floatpoint[:d][None],floatpoint[d:dim].reshape(d,1,n),
                                floatpoint[dim:].reshape(d,1,n),x,v,method)
                        timer['floating_root_validation']=time.perf_counter()-tick
                        actualf=np.r_[fs['b'].ravel(),fs['h'].ravel(),fs['u'].ravel()]
                        record['float_candidate_fixed_gap']=float(np.max(abs(actualf-floatpoint)))
                    record['status']=('actual_fixed_point' if checked['actual_fixed_point'] else
                                      'root_wrong_actual_update' if checked['domain_valid'] else 'root_outside_domain')
                else:
                    # Since w A=0, w(c-A s)=1 at every algebraic state.
                    a,rhs=system(rows,solved['active'],fixed)
                    selected=[point[j] for j in solved['active']]
                    assert dot(solved['certificate'],[c-dot(row,selected) for row,c in zip(a,rhs)])==1
                    record['status']='no_fixed_point_in_selected_formula'
                    record['w_has_multiplier_component']=any(w and j>=dim for j,w in zip(solved['active'],solved['certificate']))
                record['seconds']=dict(timer);seconds.update(timer)
                file=f"{old['seed']}_{old['family']}_{i:03}.json"
                save(out/file,record);files[file]=sha(out/file)
                records.append(dict(seed=old['seed'],family=old['family'],location=location.tolist(),
                                    file=file,sha256=files[file],status=record['status']))
                total[record['status']]+=1;total['cases']+=1
            print(dict(seed=old['seed'],family=old['family'],processed=total['cases'],seconds=time.perf_counter()-begin),flush=True)
    assert total['cases']==393 and len({(r['seed'],*r['location']) for r in records})==131
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    assert sha(design)==protocol['design_sha256']
    save(out/'rows.json',records);files['rows.json']=sha(out/'rows.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_aggregation_manifest.json',dict(files_sha256=files,all_cases_processed=True,
                                                   query_targets_accessed=False,success_event_labels_accessed=False))
    final=dict(passed=True,counts=dict(total),stage_seconds=dict(seconds),seconds=time.perf_counter()-begin,
               manifest_sha256=sha(out/'before_aggregation_manifest.json'),query_targets_accessed=False,
               independent_task_gain_established=False,resources_matched=False)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

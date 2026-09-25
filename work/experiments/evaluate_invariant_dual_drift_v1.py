"""317 post-seal phase 0/1 checks; independent rank and energy audit included."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
from multiplier_fixed_point_exact import rref_solve
from affine_root_validation_v1 import state,domain,exact_step
from audit_affine_root_proposal_v1 import independent_check,g
from effective_affine_map_v1 import restore,evaluate
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))


def run(root,out):
    start=time.perf_counter();source=root/'results/invariant_dual_drift/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_guard_validation_manifest.json')
    assert sha(source/'before_guard_validation_manifest.json')==summary['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    for name,digest in read(source/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    save(out/'protocol.json',dict(input_summary_sha256=sha(source/'summary.json'),
                                 source_sha256={n:sha(root/'work/experiments'/n) for n in
                                                [Path(__file__).name,'audit_affine_root_proposal_v1.py','affine_root_validation_v1.py','multiplier_fixed_point_exact.py']},
                                 phases=[0,1],purpose='Check the deterministic phase and its next discrete step, not a phase search.',
                                 query_targets_accessed=False,whole_interval_guards_established=False))
    families=['alm_keep','alm_reset','nodual'];aggregate={f:Counter() for f in families}
    checks=Counter();entries=[];files={};valid_examples=[]
    for entry in read(source/'rows.json'):
        r=read(source/entry['file']);proposal=r['proposal'];rows=restore(r['formula']);m=len(rows)
        x,v=list(map(F,r['x'])),list(map(F,r['v']));point=list(map(F,r['point']));n=len(x);depth=m//(1+2*n);dim=depth+depth*n
        a=[[F(i==j)-row.terms.get(j,F(0)) for j in range(m)] for i,row in enumerate(rows)]
        c=[row.terms.get(-1,F(0)) for row in rows];method=r['method'];group=aggregate[r['family']]
        if method=='alm':
            dual=list(range(dim,m))
            matrix=[row+[F(i==j) for j in dual] for i,row in enumerate(a)]
            matrix += [[F(0)]*m+[row[j] for j in dual] for row in a]
            rhs=c+[F(0)]*m;reference=point+[F(0)]*len(dual)
        else:
            matrix=[row[:dim] for row in a[:dim]];rhs=c[:dim];reference=point[:dim]
        solution,meta=rref_solve(matrix,rhs,reference)
        assert (solution is not None)==proposal['consistent'] and meta['rank']==proposal['rank']
        result=dict(seed=r['seed'],family=r['family'],location=r['location'],consistent=proposal['consistent'],phases=[])
        group['cases']+=1;checks['independent_linear_systems']+=1
        if solution is None:
            w=list(map(F,proposal['certificate']))
            assert dot(w,rhs)==1
            assert all(dot(w,[row[j] for row in matrix])==0 for j in range(len(reference)))
            group['no_restricted_invariant_line']+=1;checks['no_line_certificates']+=1
        else:
            assert solution==list(map(F,proposal['root']))
            q,delta=list(map(F,proposal['q'])),list(map(F,proposal['d']))
            assert not any(delta[:dim]);group['line_system_consistent']+=1;group['nonzero_drift']+=any(delta)
            if method=='nodual':assert not any(q[dim:]) and not any(delta)
            assert [b-z for b,z in zip(evaluate(rows,q),q)]==delta
            assert all(dot(row,delta)==0 for row in a)
            for t in [F(-2),F(0),F(1),F(7,3)]:
                assert evaluate(rows,[z+t*d for z,d in zip(q,delta)])==[z+(t+1)*d for z,d in zip(q,delta)]
                checks['exact_line_probes']+=1
            result['max_primal_correction']=max(abs(z-p) for z,p in zip(q[:dim],point[:dim]))
            result['max_multiplier_correction']=max(abs(z-p) for z,p in zip(q[dim:],point[dim:]))
            result['q_equals_original_state']=q==point
            for phase in [0,1]:
                current=[z+phase*d for z,d in zip(q,delta)];expected=[z+(phase+1)*d for z,d in zip(q,delta)]
                s=state(current,depth,x,v);violations=domain(s)
                # This independent checker honors input u in primal energies;
                # nodual here suppresses only its final zero-residual condition.
                valid,primal_unchanged,metrics,blocks,intervals=independent_check(current,depth,x,v,'nodual')
                assert valid==(not violations);checks['energy_blocks']+=blocks;checks['energy_intervals']+=intervals
                phase_result=dict(phase=phase,domain_valid=valid,domain_violations=violations,
                                  actual_step_on_line=False,metrics=metrics)
                if valid:
                    actual,rr=exact_step(s,method)
                    residual=[]
                    for j,b in enumerate(s['b']):
                        prev=x if j==0 else s['h'][j-1]
                        residual.extend(h-g(z+b) for h,z in zip(s['h'][j],prev))
                    expected_drift=[(F(1,2) if method=='alm' else F(0))*z for z in residual]
                    independent=primal_unchanged and expected_drift==delta[dim:]
                    assert independent==(actual==expected)
                    phase_result.update(actual_step_on_line=actual==expected,actual_step=actual,
                                        expected_drift_matches_residual=expected_drift==delta[dim:],
                                        full_state_gap=max(abs(a-b) for a,b in zip(actual,expected)))
                    checks['independent_actual_step_checks']+=1
                    group[f'phase{phase}_domain_valid']+=1
                    group[f'phase{phase}_actual_step_on_line']+=actual==expected
                result['phases'].append(phase_result)
            both=all(p['actual_step_on_line'] for p in result['phases'])
            group['both_phase_steps_on_line']+=both
            if both:
                group['both_steps_nonzero_drift']+=any(delta)
                group['both_steps_support_feasible']+=result['phases'][0]['metrics']['support_band_feasible']
                valid_examples.append(dict(seed=r['seed'],family=r['family'],location=r['location'],
                                           source_file=entry['file'],nonzero_drift=any(delta),
                                           support_feasible=result['phases'][0]['metrics']['support_band_feasible']))
        save(out/entry['file'],result);files[entry['file']]=sha(out/entry['file']);entries.append(entry)
    assert checks['independent_linear_systems']==393
    save(out/'aggregate.json',{f:dict(c) for f,c in aggregate.items()});files['aggregate.json']=sha(out/'aggregate.json')
    save(out/'valid_examples.json',valid_examples);files['valid_examples.json']=sha(out/'valid_examples.json')
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,checks=dict(checks),seconds=time.perf_counter()-start,outputs_sha256=files,
               query_targets_accessed=False,whole_interval_guards_established=False,resources_matched=False,
               independent_task_gain_established=False)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)
    print({f:dict(c) for f,c in aggregate.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

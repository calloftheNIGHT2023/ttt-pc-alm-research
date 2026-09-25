"""317 reconstruct current maps and freeze all 393 deterministic line proposals."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from invariant_dual_drift_v1 import propose_line
from affine_root_certificate_v1 import propose
from effective_affine_map_v1 import build,restore,canonical,dump
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();source=root/'results/affine_root_proposal/development_v1'
    tested=read(root/'results/invariant_dual_drift/tests_v1/summary.json');assert tested['passed']
    audit=root/'results/affine_root_proposal/audit_v1/summary.json';assert read(audit)['passed']
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_aggregation_manifest.json')
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    names={Path(__file__).name}
    for old in [tested,read(source/'protocol.json')]:
        for name,digest in old['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest;names.add(name)
    design=root/'outputs/ttt-pc-alm-research/317_invariant_dual_drift_design_v1.md'
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
                  tests_sha256=sha(root/'results/invariant_dual_drift/tests_v1/summary.json'),
                  previous_audit_sha256=sha(audit),source_summary_sha256=sha(source/'summary.json'),
                  design_sha256=sha(design),states=131,cases=393,tasks=64,
                  query_targets_accessed=False,future_solver_states_accessed=False,resources_matched=False,
                  phase_rule='q free coordinates retain current values, free drift coordinates zero.',
                  nodual_rule='u=0 and drift=0, solve b/h only.',
                  guard_validation='Only after all q/d proposals are sealed.')
    save(out/'protocol.json',protocol);entries=[];counts=Counter();seconds=Counter();files={}
    with discovery_box(.12):
        for entry in read(source/'rows.json'):
            original=read(source/entry['file']);point=list(map(F,original['point']))
            x,v=np.array(original['x'],dtype=str),np.array(original['v'],dtype=str)
            x=np.array([float(F(t)) for t in x]);v=np.array([float(F(t)) for t in v])
            n=len(x);d=len(point)//(1+2*n);dim=d+d*n;pp=np.array(point,float);method=original['method'];times=Counter()
            tick=time.perf_counter();current=step(pp[:d][None],pp[d:dim].reshape(d,1,n),pp[dim:].reshape(d,1,n),x,v,method)
            times['current_sweep']=time.perf_counter()-tick
            tick=time.perf_counter();rows=build({g:a[0] for g,a in current['policies'].items()},current['schema'],x,v,method)
            times['formula_construction']=time.perf_counter()-tick
            assert canonical(rows)==canonical(restore(original['formula']))
            tick=time.perf_counter()
            if method=='alm':
                answer=propose_line(rows,point,dim)
            else:
                answer=propose(rows,point,list(range(dim)),{j:F(0) for j in range(dim,len(point))})
                if answer['consistent']:answer.update(q=answer['full_root'],d=[F(0)]*len(point),nonzero_drift=False)
            times['line_system_solve_and_verify']=time.perf_counter()-tick
            record=dict(seed=entry['seed'],family=entry['family'],location=entry['location'],method=method,
                        point=point,x=list(map(F,x)),v=list(map(F,v)),formula=dump(rows),proposal=answer,seconds=dict(times))
            save(out/entry['file'],record);files[entry['file']]=sha(out/entry['file'])
            entries.append(dict(seed=entry['seed'],family=entry['family'],location=entry['location'],
                                file=entry['file'],sha256=files[entry['file']]))
            seconds.update(times);counts['cases']+=1;counts['line_system_consistent']+=answer['consistent']
            if counts['cases']%25==0:print(dict(counts=counts,seconds=time.perf_counter()-start),flush=True)
    assert counts['cases']==393
    for name,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    assert sha(design)==protocol['design_sha256']
    save(out/'rows.json',entries);files['rows.json']=sha(out/'rows.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_guard_validation_manifest.json',dict(files_sha256=files,all_cases_processed=True,
                                                        guard_outcomes_accessed=False,query_targets_accessed=False))
    result=dict(passed=True,counts=dict(counts),stage_seconds=dict(seconds),seconds=time.perf_counter()-start,
                manifest_sha256=sha(out/'before_guard_validation_manifest.json'),query_targets_accessed=False,
                actual_guard_validity_established=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

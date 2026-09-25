"""318 all frozen 317 lines, capped first-exit proposals before geometry."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from primal_stasis_escape_v1 import escape
from affine_root_validation_v1 import state,forward_metrics
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();source=root/'results/invariant_dual_drift/development_v1'
    tested=read(root/'results/primal_stasis_escape/tests_v1/summary.json');assert tested['passed']
    parent=read(source/'summary.json');assert parent['passed']
    parent_manifest=read(source/'before_guard_validation_manifest.json')
    for name,digest in parent_manifest['files_sha256'].items():assert sha(source/name)==digest
    names={Path(__file__).name,'posterior_confirmation_pipeline.py'}
    for document in [tested,read(source/'protocol.json')]:
        for name,digest in document['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest;names.add(name)
    design=root/'outputs/ttt-pc-alm-research/318_primal_stasis_escape_protocol_v1.md'
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
                  tests_sha256=sha(root/'results/primal_stasis_escape/tests_v1/summary.json'),
                  line_summary_sha256=sha(source/'summary.json'),design_sha256=sha(design),cap=64,cases=393,
                  phases_chosen_from_outcomes=False,query_targets_accessed=False,geometry_accessed=False,
                  line_generation_seconds_excluded_from_incremental_timing=True,resources_matched=False)
    save(out/'protocol.json',protocol);records=[];files={};counts=Counter();seconds=Counter()
    with discovery_box(.12):
        for entry in read(source/'rows.json'):
            r=read(source/entry['file']);p=r['proposal'];tick=time.perf_counter()
            if not p['consistent']:
                result=dict(applicable=False,status='no_restricted_invariant_line',query_targets_accessed=False)
            else:
                q,delta=list(map(F,p['q'])),list(map(F,p['d']));x,v=list(map(F,r['x'])),list(map(F,r['v']))
                n=len(x);depth=len(q)//(1+2*n);dim=depth+depth*n
                result=escape(q,delta,depth,x,v,r['method'],cap=64)
            elapsed=time.perf_counter()-tick;seconds['incremental_exact_escape']+=elapsed
            payload=dict(seed=r['seed'],family=r['family'],location=r['location'],source_file=entry['file'],
                         result=result,incremental_exact_seconds=elapsed)
            if result['status']=='first_primal_exit_found':
                pp=np.array(result['exit_input'],float);xx=np.array(x,float);vv=np.array(v,float)
                tick=time.perf_counter();actual=step(pp[:depth][None],pp[depth:dim].reshape(depth,1,n),
                                                    pp[dim:].reshape(depth,1,n),xx,vv,r['method'])
                payload['floating_exit_seconds']=time.perf_counter()-tick;seconds['floating_exit_validation']+=payload['floating_exit_seconds']
                fp=np.r_[actual['b'].ravel(),actual['h'].ravel(),actual['u'].ravel()]
                payload['floating_exit_output']=fp.tolist()
                payload['floating_vs_exact_max_gap']=float(np.max(abs(fp-np.array(result['exit_output'],float))))
                fm=forward_metrics(state(list(map(F,fp)),depth,x,v))
                payload['floating_exit_metrics']=fm
                payload['exact_float_forward_mode_match']=fm['mode']==result['exit_metrics']['mode']
            save(out/entry['file'],payload);files[entry['file']]=sha(out/entry['file'])
            records.append(dict(seed=r['seed'],family=r['family'],location=r['location'],file=entry['file'],
                                sha256=files[entry['file']],status=result['status']))
            counts['cases']+=1;counts[result['status']]+=1
            if counts['cases']%50==0:print(dict(cases=counts['cases'],seconds=time.perf_counter()-start),flush=True)
    assert counts['cases']==393
    for name,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    save(out/'rows.json',records);files['rows.json']=sha(out/'rows.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_geometry_manifest.json',dict(files_sha256=files,all_proposals_sealed=True,
                                                 query_targets_accessed=False,geometry_accessed=False))
    final=dict(passed=True,counts=dict(counts),stage_seconds=dict(seconds),seconds=time.perf_counter()-start,
               manifest_sha256=sha(out/'before_geometry_manifest.json'),query_targets_accessed=False,
               resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise

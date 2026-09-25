"""370 independent live ordering; reference patterns enter only post-fit audit."""
import argparse
from pathlib import Path
import time
import traceback
import os
import numpy as np
import multiplier_constraint_order_v1 as model
import run_prefix_language_join_v1 as old
from posterior_confirmation_pipeline import discovery_box

BASE='results/multiplier_constraint_order'


def hashes(root):
    h=old.hashes(root)
    for n in ['outputs/ttt-pc-alm-research/370_multiplier_constraint_order_protocol_v1.md',
        'work/experiments/multiplier_constraint_order_v1.py','work/experiments/run_multiplier_constraint_order_v1.py']:
        h[n]=old.sha(root/n)
    return h


def run(root,out,stage):
    begin=time.perf_counter();h=hashes(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    pre=root/BASE/'preflight_v1'
    if stage=='development':old.complete(pre);assert old.read(pre/'protocol.json')['source_sha256']==h
    cases=[c for c in old.load_cases(root,'development') if c['group']=='context_scaling' and c['n'] in [16,24]]
    if stage=='preflight':cases=cases[:1]
    old.save(out/'protocol.json',dict(stage=stage,source_sha256=h,primary='alm_dual',methods=model.NAMES,
        cases=[{k:v for k,v in c.items() if k not in ['x','v']} for c in cases],query_targets_accessed=False,new_blind_tasks=False))
    rows=[];checks={};grouped={}
    for c in cases:grouped.setdefault(c['seed'],[]).append(c)
    with discovery_box(.12):
        for seed,cc in grouped.items():
            jobs=[(c,name) for c in cc for name in model.NAMES]
            for j in np.random.default_rng(np.random.SeedSequence([370929,seed])).permutation(len(jobs)):
                c,name=jobs[int(j)];x,v=c['x'],c['v'];original=(x.tobytes(),v.tobytes())
                a,m=model.fit(x,v,seed,name);assert original==(x.tobytes(),v.tobytes())
                directory=out/f"{seed}_{c['n']}_{name}";directory.mkdir()
                np.savez_compressed(directory/'arrays.npz',**a);old.save(directory/'metadata.json',m)
                oldrows=old.read(root/'results/context_block_scaling/development/rows.json')
                positives=sorted(set().union(*(set(r.get('positive_mode_keys') or []) for r in oldrows if r['seed']==seed and r['n']==c['n'] and r['complete'])))
                reordered=[np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,c['n'])[:,a['support_order']].copy().tobytes().hex() for k in positives]
                counts=old.audit_arrays(a['x_observed'],a['v_observed'],a,m,reordered)
                if stage=='preflight':
                    aa,mm=model.fit(x,v,seed,name);counts['repeat_arrays']=old.same(a,aa,list(a))
                    assert m['completed']==mm['completed'] and m['expanded']==mm['expanded']
                if stage=='development' and c is cases[0]:
                    with np.load(pre/f"{seed}_{c['n']}_{name}"/'arrays.npz',allow_pickle=False) as z:
                        counts['preflight_arrays']=old.same(a,{k:z[k] for k in z.files},list(a))
                for k,n in counts.items():checks[k]=checks.get(k,0)+n
                old.save(directory/'audit.json',dict(passed=True,checks=counts,reference_positive_modes=len(positives),read_only_after_fit=True))
                rows.append(dict(seed=seed,n=c['n'],method=name,completed=m['completed'],stop_reason=m['stop_reason'],
                    seconds=m['total_component_seconds'],ordering_seconds=m['ordering_seconds'],join_seconds=m['total_seconds'],
                    expanded=m['expanded'],pair_candidate_rows=m['pair_candidate_rows'],remaining=len(a['regions']),
                    peak_states=m['peak_completed_prefix_states'],solver_state_bytes=m['solver_state_bytes'],
                    directory=str(directory.relative_to(root)),files={n:old.sha(directory/n) for n in ['arrays.npz','metadata.json','audit.json']}))
            print(dict(stage=stage,tasks=len({r['seed'] for r in rows}),calls=len(rows),completed=sum(r['completed'] for r in rows),seconds=time.perf_counter()-begin),flush=True)
    assert hashes(root)==h;old.save(out/'rows.json',rows)
    summary=dict(passed=True,calls=len(rows),completed=sum(r['completed'] for r in rows),budget_exhausted=sum(not r['completed'] for r in rows),
        checks=checks,seconds=time.perf_counter()-begin,query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={n:old.sha(out/n) for n in ['protocol.json','rows.json']})
    old.save(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','development'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/BASE/(args.stage+'_v1');out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except Exception:old.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

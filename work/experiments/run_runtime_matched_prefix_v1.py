"""405 prediction, audit or evaluation entry; each stage writes a new directory."""
from pathlib import Path
import argparse
import os
import time
import traceback
import numpy as np
import runtime_matched_prefix_io_v1 as io
import prefix_deadline_worker_v2 as model


def run(root,out):
    begin=time.perf_counter();h=io.dependencies(root)
    configurations=io.configs(root)
    assert len(configurations)==42 and io.PRIMARY in {c['name'] for c in configurations}
    expected=len(configurations)*len(io.SEEDS)*len(io.STAGES)*len(io.BUDGETS)
    assert expected==1344
    protocol=dict(**h,configs=configurations,seeds=io.SEEDS,stages=io.STAGES,budgets=io.BUDGETS,
        primary=io.PRIMARY,primary_budget=io.PRIMARY_BUDGET,expected_calls=expected,stage='run',
        old_development_tasks=True,independent_confirmation=False,query_targets_accessed=False,
        matched_official_ttt_included=True,strict_equal_peak_memory_enforced=False,
        setup_receives_support=False,task_state_reused_across_prefixes=False,
        setup_correction='portfolio runtime dependencies imported before ready',
        comparison_role='Repeated development with corrected timing boundary, not independent confirmation')
    io.save(out/'protocol.json',protocol);(out/'inputs').mkdir();(out/'sessions').mkdir()
    inputs={};q=np.linspace(0.,1.,257)
    for seed in io.SEEDS:
        x,v=io.observations(seed);inputs[seed]=(x,v)
        np.savez_compressed(out/'inputs'/f'{seed}.npz',x=x,v=v,q=q)
    io.save(out/'inputs_manifest.json',{p.name:io.sha(p) for p in (out/'inputs').iterdir()})
    rows=[];sessions=[]
    for ci,cfg,jobs in io.jobs(configurations,io.SEEDS,io.STAGES,io.BUDGETS):
        session=model.Session(cfg,root);method_rows=[]
        try:
            for seed,n,budget in jobs:
                x,v=inputs[seed]
                job=dict(config_index=ci,method=cfg['name'],seed=seed,n=n,budget=budget)
                io.save(out/'current_job.json',job)
                result=session.run(x[:n],v[:n],q,seed=seed,budget=budget)
                directory=out/'calls'/f"{seed}_n{n}_{cfg['name']}_b{int(budget*1000)}"
                row=dict(**job,**io.old.save_call(root,directory,*result))
                rows.append(row);method_rows.append(row)
                assert row['error'] is None,('Preserved worker error',job,row['error'])
                if len(rows)%16==0:
                    print(dict(stage='runtime_matched_run',calls=len(rows),expected=expected,
                        final=sum(r['selected']=='final' for r in rows),seconds=time.perf_counter()-begin),flush=True)
        finally:
            ss=dict(method=cfg['name'],**session.close());sessions.append(ss)
            io.save(out/'sessions'/f"{cfg['name']}.json",ss)
        io.save(out/f"sealed_{cfg['name']}.json",method_rows)
        io.old.verify_hashes(root,h['source_sha256']);io.old.verify_hashes(root,h['pretrained_sha256'])
    assert len(rows)==len({(r['seed'],r['n'],r['method'],r['budget']) for r in rows})==expected
    io.save(out/'rows.json',rows);io.save(out/'sessions.json',sessions)
    io.save(out/'summary.json',dict(passed=True,calls=expected,old_tasks=4,stage='run',
        final=sum(r['selected']=='final' for r in rows),fallback=sum(r['selected']=='fallback' for r in rows),
        constant=sum(r['selected']=='constant' for r in rows),all_predictions_sealed=True,
        query_targets_accessed=False,goal_complete=False,seconds=time.perf_counter()-begin,
        outputs_sha256={f:io.sha(out/f) for f in ['protocol.json','inputs_manifest.json','rows.json','sessions.json']}))
    print(io.read(out/'summary.json'),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit','evaluate']);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    folder={'run':'development_v1','audit':'audit_v1','evaluate':'evaluation_v1'}[args.action]
    out=root/io.BASE/folder
    if args.action!='run':assert io.read(root/io.BASE/'development_v1/summary.json')['passed']
    out.mkdir(parents=True,exist_ok=False)
    try:
        if args.action=='run':run(root,out)
        elif args.action=='audit':
            import audit_prefix_deadline_pilot_v1 as audit
            audit.io=io
            audit.audit(root,root/io.BASE/'development_v1',out)
        else:
            import evaluate_prefix_deadline_pilot_v1 as evaluator
            evaluator.io=io
            evaluator.evaluate(root,out)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

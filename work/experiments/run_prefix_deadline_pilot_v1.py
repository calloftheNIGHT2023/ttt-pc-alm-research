"""397 all-method prefix preflight or old-task deadline pilot; no query answers."""
from pathlib import Path
import argparse
import os
import time
import traceback
import numpy as np
import prefix_deadline_pilot_io_v1 as io
import prefix_deadline_worker_v1 as model


def run(root,out,stage):
    begin=time.perf_counter();h=io.dependencies(root)
    if stage=='run':
        pre=root/io.BASE/'preflight_v1';au=root/io.BASE/'audit_preflight_v1'
        assert io.read(pre/'summary.json')['passed'] and io.read(au/'summary.json')['passed']
        assert io.read(au/'summary.json')['prediction_summary_sha256']==io.sha(pre/'summary.json')
    configurations=io.configs(root);seeds=[5920002] if stage=='preflight' else io.SEEDS
    budgets=[20.] if stage=='preflight' else io.BUDGETS
    expected=len(configurations)*len(seeds)*len(io.STAGES)*len(budgets)
    protocol=dict(**h,configs=configurations,seeds=seeds,stages=io.STAGES,budgets=budgets,
        primary=io.PRIMARY,primary_budget=io.PRIMARY_BUDGET,expected_calls=expected,stage=stage,
        old_development_tasks=True,independent_confirmation=False,query_targets_accessed=False,
        matched_official_ttt_included=True,strict_equal_peak_memory_enforced=False,
        setup_receives_support=False,task_state_reused_across_prefixes=False)
    io.save(out/'protocol.json',protocol);(out/'inputs').mkdir();(out/'sessions').mkdir()
    inputs={};q=np.linspace(0.,1.,257)
    for seed in seeds:
        x,v=io.observations(seed);inputs[seed]=(x,v)
        np.savez_compressed(out/'inputs'/f'{seed}.npz',x=x,v=v,q=q)
    io.save(out/'inputs_manifest.json',{p.name:io.sha(p) for p in (out/'inputs').iterdir()})
    rows=[];sessions=[]
    for ci,cfg,jobs in io.jobs(configurations,seeds,io.STAGES,budgets):
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
                if stage=='preflight':assert row['selected']=='final',('Wide preflight did not complete',job)
                if len(rows)%16==0:
                    print(dict(stage=stage,calls=len(rows),expected=expected,final=sum(r['selected']=='final' for r in rows),seconds=time.perf_counter()-begin),flush=True)
        finally:
            ss=dict(method=cfg['name'],**session.close());sessions.append(ss)
            io.save(out/'sessions'/f"{cfg['name']}.json",ss)
        io.save(out/f"sealed_{cfg['name']}.json",method_rows)
        io.old.verify_hashes(root,h['source_sha256']);io.old.verify_hashes(root,h['pretrained_sha256'])
    assert len(rows)==len({(r['seed'],r['n'],r['method'],r['budget']) for r in rows})==expected
    io.save(out/'rows.json',rows);io.save(out/'sessions.json',sessions)
    io.save(out/'summary.json',dict(passed=True,calls=expected,old_tasks=len(seeds),stage=stage,
        final=sum(r['selected']=='final' for r in rows),fallback=sum(r['selected']=='fallback' for r in rows),constant=sum(r['selected']=='constant' for r in rows),
        all_predictions_sealed=True,query_targets_accessed=False,goal_complete=False,seconds=time.perf_counter()-begin,
        outputs_sha256={f:io.sha(out/f) for f in ['protocol.json','inputs_manifest.json','rows.json','sessions.json']}))
    print(io.read(out/'summary.json'),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','run'],required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/io.BASE/('preflight_v1' if args.stage=='preflight' else 'development_v1')
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

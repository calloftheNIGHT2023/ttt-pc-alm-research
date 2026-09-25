"""397 old-task descriptive risk only after all predictions and audits are sealed."""
from pathlib import Path
from collections import defaultdict
import math
import os
import traceback
import numpy as np
import prefix_deadline_pilot_io_v1 as io


def evaluate(root,out):
    source=root/io.BASE/'development_v1';audit=root/io.BASE/'audit_v1'
    ss,au=io.read(source/'summary.json'),io.read(audit/'summary.json')
    assert ss['passed'] and au['passed'] and au['prediction_summary_sha256']==io.sha(source/'summary.json')
    for folder,summary in [(source,ss),(audit,au)]:
        for f,digest in summary['outputs_sha256'].items():assert io.sha(folder/f)==digest
    protocol=io.read(source/'protocol.json');io.old.verify_hashes(root,protocol['source_sha256']);io.old.verify_hashes(root,protocol['pretrained_sha256'])
    rows=io.read(source/'rows.json');assert len(rows)==1344 and protocol['stage']=='run'
    io.save(out/'before_query.json',dict(prediction_summary_sha256=io.sha(source/'summary.json'),audit_summary_sha256=io.sha(audit/'summary.json'),
        protocol_sha256=io.sha(source/'protocol.json'),old_development_tasks=True,evaluator_sha256=io.sha(Path(__file__))))
    import streaming_branch_projection as base
    q=np.linspace(0,1,257);truth={s:base.forward(q,np.random.default_rng(s).uniform(-.12,.12,4)) for s in io.SEEDS}
    np.savez_compressed(out/'query_truth.npz',q=q,seeds=np.array(io.SEEDS),targets=np.stack([truth[s] for s in io.SEEDS]))
    risk=[];groups=defaultdict(list);task_curves=defaultdict(list)
    for row in rows:
        folder=root/row['directory'];assert io.sha(folder/'outputs.npz')==row['files']['outputs.npz']
        prediction=io.load_arrays(folder/'outputs.npz')['prediction']
        r=dict(**row,query_mse=float(np.mean((prediction-truth[row['seed']])**2)))
        risk.append(r);groups[row['method'],row['n'],row['budget']].append(r);task_curves[row['method'],row['budget'],row['seed']].append(r)
    gg=[]
    for (method,n,budget),rr in groups.items():
        assert sorted(r['seed'] for r in rr)==io.SEEDS
        gg.append(dict(method=method,n=n,budget=budget,old_tasks=4,mean_query_mse=math.fsum(r['query_mse'] for r in rr)/4,
            final=sum(r['selected']=='final' for r in rr),fallback=sum(r['selected']=='fallback' for r in rr),constant=sum(r['selected']=='constant' for r in rr),
            mean_online_seconds=math.fsum(r['online_seconds'] for r in rr)/4,
            mean_setup_seconds=math.fsum(r['setup_seconds'] for r in rr)/4,
            mean_cleanup_seconds=math.fsum(r['cleanup_seconds']+r['task_cleanup_seconds'] for r in rr)/4,
            mean_archive_seconds=math.fsum(r['archive_seconds'] for r in rr)/4,
            max_controller_overrun_seconds=max(r['controller_overrun_seconds'] for r in rr),
            max_worker_sampled_rss=max(r['worker_max_sampled_rss'] for r in rr)))
    curves={}
    for key,rr in task_curves.items():
        assert sorted(r['n'] for r in rr)==io.STAGES
        curves[key]=math.fsum(r['query_mse'] for r in rr)/4
    averaged=[];comparisons=[]
    for cfg in protocol['configs']:
        for budget in io.BUDGETS:
            values=[curves[cfg['name'],budget,s] for s in io.SEEDS]
            averaged.append(dict(method=cfg['name'],budget=budget,task_prefix_average_mse=values,mean_query_mse=math.fsum(values)/4,old_tasks=4))
            if cfg['name']!=io.PRIMARY:
                delta=[curves[io.PRIMARY,budget,s]-curves[cfg['name'],budget,s] for s in io.SEEDS]
                comparisons.append(dict(primary=io.PRIMARY,control=cfg['name'],budget=budget,
                    primary_budget=budget==io.PRIMARY_BUDGET,per_task_difference=delta,mean_paired_difference=math.fsum(delta)/4,
                    descriptive_only=True,significance_test_performed=False))
    assert len(gg)==336 and len(averaged)==84 and len(comparisons)==82
    for name,value in [('risk_rows.json',risk),('groups.json',gg),('prefix_averages.json',averaged),('comparisons.json',comparisons)]:io.save(out/name,value)
    io.save(out/'summary.json',dict(passed=True,calls=len(rows),old_tasks=4,groups=len(gg),prefix_averages=len(averaged),comparisons=len(comparisons),
        descriptive_only=True,independent_confirmation=False,core_research_goal_complete=False,
        outputs_sha256={f:io.sha(out/f) for f in ['before_query.json','query_truth.npz','risk_rows.json','groups.json','prefix_averages.json','comparisons.json']}))
    print(io.read(out/'summary.json'),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    root=Path(__file__).resolve().parents[2];out=root/io.BASE/'evaluation_v1';out.mkdir(parents=True,exist_ok=False)
    try:evaluate(root,out)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

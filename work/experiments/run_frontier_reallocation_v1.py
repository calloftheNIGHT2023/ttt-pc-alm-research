"""362 support-only component predictions; all geometry remains outside."""
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import frontier_reallocation_v1 as model
from test_frontier_reallocation_v1 import hashes
from test_region_conditioned_credit_v1 import guarded

BASE='results/frontier_reallocation'
CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    begin=time.perf_counter();frozen=hashes(root);pre=io.complete(root/BASE/'preflight_v1');assert pre['source_sha256']==frozen
    parent=root/'results/cross_region_credit/development_v1';io.complete(parent);tasks=io.read(parent/'tasks.json')
    configs=[dict(name=c+'_'+s,channel=c,strategy=s) for c in CHANNELS for s in ['frontier','uniform']]
    p=dict(source_sha256=frozen,configs=configs,seeds=[t['seed'] for t in tasks],primary='dual_frontier',
           parent_tasks_sha256=io.sha(parent/'tasks.json'),steps=128,budget_per_candidate=128,geometry_budget=8,
           query_targets_accessed=False,geometry_accessed=False,archived_input_generation_cost_excluded=True)
    io.save(out/'protocol.json',p);rows=[];seals=[]
    for task in tasks:
        seed=task['seed'];source=parent/str(seed);d=out/str(seed);d.mkdir()
        for name,digest in task['files'].items():assert io.sha(source/name)==digest
        inp=io.read(source/'input.json');x=np.array(inp['x_observed']);v=np.array(inp['v_observed'])
        with np.load(source/'pool.npz',allow_pickle=False) as z:regs=z['regions']
        records=[]
        for ci in np.random.default_rng(np.random.SeedSequence([362929,seed])).permutation(12):
            cfg=configs[int(ci)];name=cfg['name'];credit=np.array(inp['credits'][cfg['channel']])
            with guarded():a,m=model.solve(x,v,regs,credit,strategy=cfg['strategy'])
            with np.load(source/(cfg['channel']+'_independent.npz'),allow_pickle=False) as z:
                expected=np.flatnonzero(z['first_step']==0)[:8];assert a['base_selected'].tolist()==expected.tolist()
                end=m['base_processed']
                for key,oldkey in [('base_first_step','first_step'),('base_disabled','disabled'),('base_final_credit','final_credit')]:
                    assert a[key][:end].tobytes()==z[oldkey][:end].tobytes()
                assert not z['disabled'].any()
            assert int(a['response_counts'].sum())==m['total_response_pairs']<=m['budget_limit']==128*len(regs)
            np.savez_compressed(d/(name+'.npz'),**a);io.save(d/(name+'.json'),m)
            row=dict(seed=seed,method=name,order=len(records),candidates=len(regs),seconds=m['total_seconds'],
                     files={ext:io.sha(d/(name+ext)) for ext in ['.json','.npz']})
            records.append(row);rows.append(row)
        io.save(d/'commit.json',dict(seed=seed,rows=records));seals.append(dict(seed=seed,source_files=task['files'],commit_sha256=io.sha(d/'commit.json')))
        print(dict(tasks=len(seals),calls=len(rows),seconds=time.perf_counter()-begin),flush=True)
    io.save(out/'rows.json',rows);io.save(out/'input_seals.json',seals);assert hashes(root)==frozen
    result=dict(passed=True,tasks=len(tasks),calls=len(rows),seconds=time.perf_counter()-begin,query_targets_accessed=False,
                geometry_accessed=False,core_research_goal_complete=False,
                outputs_sha256={n:io.sha(out/n) for n in ['protocol.json','rows.json','input_seals.json']})
    io.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/BASE/'development_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except Exception:io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

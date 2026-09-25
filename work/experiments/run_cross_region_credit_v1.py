"""357 full necessary-language pool and common causal component comparison."""
from itertools import product,islice
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import cross_region_credit_v1 as model
import local_region_screen as screening
from branch_image_chain_dyadic_v1 import IntegerProblem
from support_language_chain_v1 import accepted_paths
from test_region_conditioned_credit_v1 import guarded
from test_cross_region_credit_v1 import hashes
from run_region_conditioned_credit_serialization_v2 import lists
from posterior_confirmation_pipeline import discovery_box

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def pool(x,v,original,visited):
    begin=time.perf_counter();d,n=original.shape;problem=IntegerProblem(x,v,np.zeros((d,n)))
    languages=[accepted_paths(problem,i) for i in range(n)];enumerator=product(*languages)
    forbidden={bytes.fromhex(k) for k in visited};kept=[];counts=dict(enumerated=0,visited=0,c5=0,c20=0)
    c5seconds=c20seconds=0.
    while True:
        block=list(islice(enumerator,1024))
        if not block:break
        regs=np.array(block,np.uint8).transpose(0,2,1).copy();counts['enumerated']+=len(regs)
        unvisited=np.array([r.tobytes() not in forbidden for r in regs]);counts['visited']+=int((~unvisited).sum())
        regs=regs[unvisited]
        if not len(regs):continue
        tick=time.perf_counter();mask=screening.contract(x,v,regs,5);c5seconds+=time.perf_counter()-tick
        counts['c5']+=int(mask.sum());regs=regs[~mask]
        if not len(regs):continue
        tick=time.perf_counter();mask=screening.contract(x,v,regs,20);c20seconds+=time.perf_counter()-tick
        counts['c20']+=int(mask.sum());kept.extend(r.tobytes() for r in regs[~mask])
    anchor=original.ravel();kept.sort(key=lambda b:(int((np.frombuffer(b,np.uint8)!=anchor).sum()),b))
    regs=np.array([np.frombuffer(b,np.uint8).reshape(d,n) for b in kept],np.uint8).reshape(-1,d,n)
    return regs,dict(counts,remaining=len(kept),single_counts=list(map(len,languages)),
                     c5_seconds=c5seconds,c20_seconds=c20seconds,total_seconds=time.perf_counter()-begin,
                     pool_array_bytes=regs.nbytes,language_counts_product=int(np.prod(list(map(len,languages)))))


def run(root,out):
    begin=time.perf_counter();pre=root/'results/cross_region_credit/preflight_v1/summary.json'
    assert io.read(pre)['passed'];frozen=hashes(root);assert frozen==io.read(pre)['source_sha256']
    parent=root/'results/region_conditioned_credit/development_v1';oldtasks=io.read(parent/'tasks.json')
    configs=[dict(name=c+('_reuse' if reuse else '_independent'),channel=c,reuse=reuse,steps=128)
             for c in CHANNELS for reuse in [False,True]]
    configs.append(dict(name='zero_independent256',channel='zero',reuse=False,steps=256))
    protocol=dict(source_sha256=frozen,preflight_sha256=io.sha(pre),parent_tasks_sha256=io.sha(parent/'tasks.json'),
        seeds=[r['seed'] for r in oldtasks],configs=configs,primary='dual_reuse',query_targets_accessed=False,
        original_parent_trajectory_cost_excluded=True,shared_generation_charged_separately=True)
    io.save(out/'protocol.json',protocol);tasks=[]
    with discovery_box(.12):
        for old in oldtasks:
            seed=old['seed'];directory=out/str(seed);directory.mkdir();inp_path=parent/str(seed)/'input.json'
            assert io.sha(inp_path)==old['files']['input.json'];inp=io.read(inp_path)
            x=np.array(inp['x_observed']);v=np.array(inp['v_observed']);credits={c:np.array(a) for c,a in inp['credits'].items()}
            common=None;parents={}
            for c in CHANNELS:
                p=(root/inp['source_files'][c]['file']).with_suffix('.json');m=io.read(p)['metadata']
                values=(m['visited_modes'],m['selected_state']['original_mode'])
                if common is None:common=values
                assert common==values;parents[c]=dict(file=str(p.relative_to(root)),sha256=io.sha(p))
            visited,original=common;original=np.frombuffer(bytes.fromhex(original),np.uint8).reshape(4,4)
            with guarded():regs,poolmeta=pool(x,v,original,visited)
            np.savez_compressed(directory/'pool.npz',regions=regs)
            io.save(directory/'input.json',dict(seed=seed,x_observed=x.tolist(),v_observed=v.tolist(),
                original=original.tolist(),visited=visited,credits={c:a.tolist() for c,a in credits.items()},
                trigger_b=inp['trigger_b'],parents=parents,parent_input_sha256=io.sha(inp_path),pool=poolmeta))
            files={n:io.sha(directory/n) for n in ['pool.npz','input.json']}
            rng=np.random.default_rng(np.random.SeedSequence([357091,seed]));order=rng.permutation(len(configs))
            for index in order:
                cfg=configs[int(index)]
                with guarded():arrays,meta=model.solve(x,v,regs,credits[cfg['channel']],propagate=cfg['reuse'],steps=cfg['steps'])
                np.savez_compressed(directory/(cfg['name']+'.npz'),**arrays)
                io.save(directory/(cfg['name']+'.json'),meta)
                files[cfg['name']+'.npz']=io.sha(directory/(cfg['name']+'.npz'));files[cfg['name']+'.json']=io.sha(directory/(cfg['name']+'.json'))
            with guarded():
                tick=time.perf_counter()
                rejected,meta=screening.pdhg(x,v,np.broadcast_to(np.array(inp['trigger_b']),(len(regs),4)).copy(),regs,60,20)
                seconds=time.perf_counter()-tick
            np.savez_compressed(directory/'pdhg60.npz',rejected=rejected)
            io.save(directory/'pdhg60.json',lists(dict(seconds=seconds,metadata=meta)))
            files['pdhg60.npz']=io.sha(directory/'pdhg60.npz');files['pdhg60.json']=io.sha(directory/'pdhg60.json')
            row=dict(seed=seed,regions=len(regs),pool=poolmeta,execution_order=[configs[int(i)]['name'] for i in order],files=files)
            io.save(directory/'commit.json',row);tasks.append(row)
            print(dict(tasks=len(tasks),remaining=sum(r['regions'] for r in tasks),
                       enumerated=sum(r['pool']['enumerated'] for r in tasks),seconds=time.perf_counter()-begin),flush=True)
    assert frozen==hashes(root);io.save(out/'tasks.json',tasks)
    summary=dict(passed=True,tasks=len(tasks),calls=len(tasks)*len(configs),baseline_pdhg_calls=len(tasks),
        regions=sum(r['regions'] for r in tasks),enumerated=sum(r['pool']['enumerated'] for r in tasks),
        seconds=time.perf_counter()-begin,query_targets_accessed=False,geometry_accessed=False,
        outputs_sha256={n:io.sha(out/n) for n in ['protocol.json','tasks.json']})
    io.save(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/cross_region_credit/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except Exception:io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

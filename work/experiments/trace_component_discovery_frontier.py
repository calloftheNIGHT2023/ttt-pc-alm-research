"""Support-only component seeding versus finite iteration budget.

The complete feasible-mode graph is applied only AFTER frozen trajectories have
finished. No oracle information enters starts, updates, retention, or stopping.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import stateful_posterior_memory as model
from diagnose_missing_mode_graph import distances
base=model.base


def trace(starts,x,v,method,steps,checkpoints):
    anchor=np.zeros(4);saved={};archive=model.interface.archived.Archive(x);score_calls=0;eval_calls=0
    original_score=base.score;original_evaluate=model.core.batched.evaluate
    def retain(b,step):
        archive.add(b)
        if step in checkpoints:saved[int(step)]=list(k.hex() for k in archive.points)
    def score(b,*args,**kwargs):
        nonlocal score_calls
        if method!='adam' and score_calls<=steps:retain(b,score_calls)
        elif method=='adam' and score_calls==0:retain(b,0)
        score_calls+=1;return original_score(b,*args,**kwargs)
    def evaluate(b,*args,**kwargs):
        nonlocal eval_calls
        retain(b,eval_calls);eval_calls+=1;return original_evaluate(b,*args,**kwargs)
    def run():
        if method=='adam':return model.core.batched.refine(starts,x,v,anchor,solver='adam',steps=steps,lr=.003)
        if method=='pc':return model.refine_pc(starts,x,v,anchor,sweeps=steps)
        return model.core.contextual.refine_local(starts,x,v,anchor,sweeps=steps,dual_rate=0. if method=='nodual' else .5)
    with model.core.pipeline.discovery_box(.12):
        try:
            base.score=score
            if method=='adam':model.core.batched.evaluate=evaluate
            final,meta=run()
        finally:base.score=original_score;model.core.batched.evaluate=original_evaluate
        begin=time.perf_counter();reference,_=run();untraced=time.perf_counter()-begin
    assert np.array_equal(final,reference)
    assert sorted(saved)==checkpoints,(method,sorted(saved),checkpoints)
    return saved,dict(final_best_exact_untraced=True,untraced_full_refinement_seconds=untraced,
        maximum_archived_patterns=len(archive.points),archive_bookkeeping_seconds=archive.seconds,
        scope='frontier captures have tracing overhead; only full untraced replay timed, not per-checkpoint latency')


def describe(keys,allkeys,weights,adjacency):
    found=np.array([k in keys for k in allkeys]);steps=distances(adjacency,found)
    return dict(visited_patterns=len(keys),positive_regions=int(found.sum()),found_mass=float(weights[found].sum()),
        positive_regions_after_one_hop=int(((steps>=0)&(steps<=1)).sum()),
        positive_regions_after_two_hops=int(((steps>=0)&(steps<=2)).sum()),
        mass_after_one_hop=float(weights[(steps>=0)&(steps<=1)].sum()),mass_after_two_hops=float(weights[(steps>=0)&(steps<=2)].sum()),
        mass_of_seeded_components=float(weights[steps>=0].sum()),unseeded_component_mass=float(weights[steps<0].sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'component_frontier'
    previous=json.loads((a.root/'missing_mode_graph/protocol.json').read_text())
    for name,h in previous['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    configs=[dict(method='alm',steps=20,checkpoints=[0,1,2,4,8,16,20]),
             dict(method='nodual',steps=20,checkpoints=[0,1,2,4,8,16,20]),
             dict(method='pc',steps=80,checkpoints=[0,1,2,4,8,16,20,32,60,80]),
             dict(method='adam',steps=240,checkpoints=[0,1,2,4,8,16,20,32,60,120,240])]
    protocol=dict(seeds=previous['seeds'],configs=configs,prior_features=256,restarts=64,
        source_sha256={**previous['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='support-only posthoc graph seeding diagnostic, not online oracle closure, no query input/answer; fixed old streams',
        direct_controls=[64,128,257],direct_control_note='R257 means anchor plus all distinct patterns in the same F256 library')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];audits=[]
    for seed in protocol['seeds']:
        data=json.loads((a.root/f'first_write_reference/reference_{seed}.json').read_text());assert data['numerical_volume_reference_complete']
        x=np.array(data['x']);v=np.array(data['v']);cells=sorted(data['positive_regions'],key=lambda r:r['pattern']);keys=[c['pattern'] for c in cells]
        patterns=np.array([np.frombuffer(bytes.fromhex(k),np.uint8) for k in keys]);distance=np.abs(patterns[:,None].astype(int)-patterns[None,:].astype(int)).sum(-1)
        adjacency=distance==1;weights=np.array([c['volume'] for c in cells]);weights/=weights.sum()
        pool=model.interface.make_pool(4,'prior256')
        for r in protocol['direct_controls']:
            starts,_=model.interface.select_pool(x,v,np.zeros(4),pool,r);found={reg.tobytes().hex() for reg in model.interface.archived.signatures(x,starts)}
            rows.append(dict(seed=seed,method=f'direct{r}',steps=0,**describe(found,keys,weights,adjacency)))
        starts,_=model.interface.select_pool(x,v,np.zeros(4),pool,64)
        for c in configs:
            snapshots,meta=trace(starts,x,v,c['method'],c['steps'],c['checkpoints'])
            audits.append(dict(seed=seed,method=c['method'],**meta));(out/f'trace_{seed}_{c["method"]}.json').write_text(json.dumps(snapshots,indent=2),encoding='utf-8')
            for step,seen in snapshots.items():rows.append(dict(seed=seed,method=c['method'],steps=step,**describe(set(seen),keys,weights,adjacency)))
        (out/'frontier.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,rows=len(rows))),flush=True)
    summary=[]
    for method,step in sorted({(r['method'],r['steps']) for r in rows}):
        rr=[r for r in rows if r['method']==method and r['steps']==step]
        summary.append(dict(method=method,steps=step,**{k:float(np.mean([r[k] for r in rr])) for k in ['visited_patterns','positive_regions','found_mass','mass_after_one_hop','mass_after_two_hops','mass_of_seeded_components','unseeded_component_mass']},
            tasks_full_component_mass=sum(r['unseeded_component_mass']<1e-10 for r in rr)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(dict(complete=True,rows=len(rows),audits=len(audits),summary=summary),indent=2))


if __name__=='__main__':main()

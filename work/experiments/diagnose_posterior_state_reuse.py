"""Support-only prior-vs-previous-state proposal audit with frozen history.

No query inputs or query targets are sampled. Shared geometry caching is solely
diagnostic reuse and is explicitly not an online timing comparison.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import posterior_state_interface as memory
import local_region_screen as screen
base=memory.base


def diagnose_pool(x,v,anchor,pool,cache):
    begin=time.perf_counter();starts,meta=memory.select_pool(x,v,anchor,pool)
    bank=memory.core.contextual.deduplicate(starts,x,v,anchor);regs=memory.archived.signatures(x,bank)
    reject=screen.contract(x,v,regs,rounds=5);selection=time.perf_counter()-begin
    positive=[];calls=0;statuses={}
    for b,reg,skip in zip(bank,regs,reject):
        if skip:continue
        key=reg.tobytes()
        if key not in cache:
            _,_,g,rhs=base.branch_polytope(x,v,b);poly,note=memory.posterior.polytope(g,rhs)
            cache[key]=dict(**note,representative=b.tolist(),pattern=key.hex());calls+=1
        note=cache[key];statuses[note['reason']]=statuses.get(note['reason'],0)+1
        if note['reason']=='positive_volume':positive.append(note)
    return dict(**meta,selected_distinct_regions=len(bank),screened_regions=int(reject.sum()),geometry_regions=int((~reject).sum()),
                actual_geometry_calls_with_shared_cache=calls,positive_regions=len(positive),positive_geometry=positive,
                discovered_prior_mass=sum(p['volume'] for p in positive)/.24**4,geometry_statuses=statuses,
                selection_and_screen_seconds=selection,scope='geometry cost shared across pools only in this diagnostic; not total online time')


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    previous=json.loads((a.original/'protocol.json').read_text());old=json.loads((a.original/'episodes.json').read_text())
    assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in previous['source_sha256'].items())
    names=['context_alm20','context_adam60','context_adam240'];configs=[c for c in previous['configs'] if c['name'] in names]
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    sources=[Path(__file__),Path(memory.__file__),Path(memory.archived.__file__),Path(screen.__file__)]
    protocol=dict(seeds=list(range(5900000,5900016)),stages=[4,8,16,24],configs=configs,pools=['prior256','prior768','posterior_mix'],
                  source_sha256={**previous['source_sha256'],**{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}},
                  audit=memory.verify(),scope='support-only proposal audit from own frozen historical state; no query data; old tasks')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    lookup={(r['method'],r['seed'],r['n_context']):r for r in old};rows=[];captures=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        for c in configs:
            anchor=np.zeros(4);state=None
            for n in protocol['stages']:
                if state is not None:
                    cache={}
                    for kind in protocol['pools']:
                        pool=memory.make_pool(4,kind,state)
                        rows.append(dict(method=c['name'],seed=seed,n_context=n,pool=kind,previous_posterior_samples=len(state),
                                         previous_posterior_numeric_bytes=state.nbytes,**diagnose_pool(x[:n],v[:n],anchor,pool,cache)))
                predict,anchor,state,meta=memory.fit_captured(x[:n],v[:n],anchor,dict(**c,posterior_samples=512))
                reference=lookup[(c['name'],seed,n)]
                assert np.array_equal(anchor,np.array(reference['anchor_output']))
                assert meta['positive_volume_regions']==reference['positive_volume_regions'] and meta['discovered_prior_mass']==reference['discovered_prior_mass']
                filename=f'{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(a.out/filename,posterior=state,anchor=anchor,x=x[:n],v=v[:n])
                captures.append(dict(method=c['name'],seed=seed,n_context=n,original_anchor_and_geometry_exact=True,posterior_samples=len(state),
                                     numeric_persistent_state_bytes=state.nbytes+anchor.nbytes,state_file=filename,state_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest()))
        (a.out/'diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(a.out/'captures.json').write_text(json.dumps(captures,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-5900000+1,total=16,diagnostics=len(rows),captures=len(captures))),flush=True)
    print(json.dumps(dict(complete=True,diagnostics=len(rows),captures=len(captures))),flush=True)


if __name__=='__main__':main()

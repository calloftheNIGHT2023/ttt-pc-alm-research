"""Frozen visited-mode archive assessed against complete conditional reference."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import stateful_posterior_memory as model
archive=model.interface.archived


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'archive_conditional_risk'
    original=json.loads((a.root/'conditional_risk/protocol.json').read_text())
    for name,h in original['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    configs=[]
    for family,generator,steps in [('alm20','alm',20),('adam60','adam',60),('adam240','adam',240)]:
        for enabled in [False,True]:configs.append(dict(name=family+('_archive' if enabled else '_original'),generator=generator,
            initialization='contextual',features=256,restarts=64,discovery_bound=.12,sweeps=steps,steps=steps,lr=.003,archive=enabled))
    protocol=dict(seeds=original['seeds'],configs=configs,primary='ALM20 archived conditional excess risk vs equally archived Adam60/240',
        source_sha256={**original['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='frozen study79 archive at first write only, complete numerical reference from study88 and same independent MC curves from study90; no query answers, no retuning')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    for seed in protocol['seeds']:
        audit=json.loads((a.root/f'conditional_risk/audit_{seed}.json').read_text());path=a.root/'conditional_risk'/audit['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['curve_sha256']
        with np.load(path) as z:
            x=z['x'];v=z['v'];q=z['q'];weights=z['weights'];keys=z['patterns'];mu=z['region_means'];full=np.einsum('k,rkq->rq',weights,mu)
            for c in configs:
                begin=time.perf_counter();bank,dm=archive.discover(x,v,np.zeros(4),c);predict,state,pm=model.materialize(x,v,np.zeros(4),bank);write=time.perf_counter()-begin
                found={r.tobytes().hex() for r in archive.signatures(x,bank)};mask=np.array([k in found for k in keys]);w=weights*mask;alpha=w.sum();w/=alpha
                assert int(mask.sum())==pm['positive_volume_regions']
                if not c['archive']:
                    method=c['name'].replace('_original','_prior256')
                    with np.load(a.root/f'online_development/{seed}_{method}_n4.npz') as old:assert np.array_equal(old['samples'],state.samples) and np.array_equal(old['anchor'],state.anchor)
                truncated=np.einsum('k,rkq->rq',w,mu);actual=predict(q);est=[]
                for left,right in original['pairs']:
                    est.append(dict(actual_excess=float(np.trapezoid((actual-full[left])*(actual-full[right]),x=q)),
                        truncation_excess=float(np.trapezoid((truncated[left]-full[left])*(truncated[right]-full[right]),x=q))))
                filename=f'{seed}_{c["name"]}.npz';np.savez_compressed(out/filename,samples=state.samples,anchor=state.anchor,found_patterns=keys[mask])
                rows.append(dict(seed=seed,method=c['name'],mass_fraction=float(alpha),positive_regions=int(mask.sum()),
                    **{k:float(np.mean([e[k] for e in est])) for k in est[0]},
                    first_write_seconds=write,archive_unique_patterns=dm['archive_unique_patterns'],archive_numeric_key_bytes=dm['archive_numeric_key_bytes'],
                    persistent_state_bytes=pm['persistent_state_bytes'],geometry_bank_patterns=dm['geometry_bank_patterns'],
                    state_file=filename,state_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest()))
        (out/'risks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed=len(rows)//6,total=16)),flush=True)
    names=[c['name'] for c in configs];summary=[];lookup={(r['seed'],r['method']):r for r in rows}
    for name in names:
        group=[r for r in rows if r['method']==name]
        summary.append(dict(method=name,**{k:float(np.mean([r[k] for r in group])) for k in ['actual_excess','truncation_excess','mass_fraction','positive_regions']},
            median_first_write_seconds=float(np.median([r['first_write_seconds'] for r in group])),
            maximum_archive_numeric_key_bytes=max(r['archive_numeric_key_bytes'] for r in group)))
    pairs=[('alm20_archive',n) for n in ['alm20_original','adam60_archive','adam240_archive']]
    pairs +=[(f'{f}_archive',f'{f}_original') for f in ['adam60','adam240']]
    rng=np.random.default_rng(558131);ids=rng.integers(0,16,(20000,16));paired=[]
    for left,right in pairs:
        for metric in ['actual_excess','truncation_excess']:
            delta=np.array([lookup[s,left][metric]-lookup[s,right][metric] for s in original['seeds']])
            paired.append(dict(comparison=left+' minus '+right,metric=metric,mean_difference=float(delta.mean()),
                descriptive_ci95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),lower_tasks=int((delta<-1e-12).sum()),equal_tasks=int((abs(delta)<=1e-12).sum())))
    result=dict(rows=len(rows),summary=summary,paired=paired,scope=protocol['scope'],baseline_states_reproduced_exactly=48)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

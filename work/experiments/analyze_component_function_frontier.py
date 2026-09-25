"""Map support-only reachable sets to the already frozen conditional moments.

This is an oracle diagnostic, not an implemented completion algorithm. It is
used only on old development tasks; actual proposal/solver costs are absent.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import stateful_posterior_memory as model
from diagnose_missing_mode_graph import distances


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'component_function_frontier';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((a.root/'component_frontier/protocol.json').read_text())
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    spec=dict(source_sha256={**protocol['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},hops=[0,1,2],
        scope='old support-conditioned oracle reachable-set risk; no hidden teacher, query answers or actual branch-completion timing')
    assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf-8');rows=[]
    for seed in protocol['seeds']:
        audit=json.loads((a.root/f'conditional_risk/audit_{seed}.json').read_text());path=a.root/'conditional_risk'/audit['curve_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['curve_sha256']
        with np.load(path) as z:
            keys=z['patterns'];pattern=np.array([np.frombuffer(bytes.fromhex(k),np.uint8) for k in keys]);adjacency=np.abs(pattern[:,None].astype(int)-pattern[None,:].astype(int)).sum(-1)==1
            w=z['weights'];mu=z['region_means'];second=z['region_second_moments'];q=z['q'];full=np.einsum('k,rkq->rq',w,mu);snapshots=[]
            for c in protocol['configs']:
                traces=json.loads((a.root/f'component_frontier/trace_{seed}_{c["method"]}.json').read_text())
                snapshots.extend((c['method'],int(step),set(seen)) for step,seen in traces.items())
            for r in protocol['direct_controls']:
                starts,_=model.interface.select_pool(z['x'],z['v'],np.zeros(4),model.interface.make_pool(4,'prior256'),r)
                snapshots.append((f'direct{r}',0,{p.tobytes().hex() for p in model.interface.archived.signatures(z['x'],starts)}))
            for method,steps,seen in snapshots:
                found=np.array([k in seen for k in keys]);distance=distances(adjacency,found)
                for hops in spec['hops']:
                    mask=(distance>=0)&(distance<=hops);ww=w*mask;mass=ww.sum();assert mass>0;ww/=mass
                    pred=np.einsum('k,rkq->rq',ww,mu);s2=np.einsum('k,rkq->rq',ww,second);bias=[];sampling=[]
                    for left,right in [(0,1),(2,3)]:
                        bias.append(float(np.trapezoid((pred[left]-full[left])*(pred[right]-full[right]),x=q)))
                        sampling.append(float(np.trapezoid(((s2[left]+s2[right])/2-pred[left]*pred[right])/512,x=q)))
                    rows.append(dict(seed=seed,method=method,steps=steps,hops=hops,positive_regions=int(mask.sum()),mass_fraction=float(mass),
                        truncation_excess=float(np.mean(bias)),expected_sampling_excess=float(np.mean(sampling)),
                        expected_512_excess=float(np.mean(bias)+np.mean(sampling))))
    summary=[]
    for method,steps,hops in sorted({(r['method'],r['steps'],r['hops']) for r in rows}):
        rr=[r for r in rows if (r['method'],r['steps'],r['hops'])==(method,steps,hops)]
        summary.append(dict(method=method,steps=steps,hops=hops,**{k:float(np.mean([r[k] for r in rr])) for k in ['mass_fraction','truncation_excess','expected_sampling_excess','expected_512_excess','positive_regions']}))
    (out/'risks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),summary=summary),indent=2))


if __name__=='__main__':main()

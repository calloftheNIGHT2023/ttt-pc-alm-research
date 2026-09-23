"""Conditional Bayes-risk decomposition from observed context only.

Independent per-mode Monte Carlo replicas make cross-products unbiased for the
conditional mean squared errors on a fixed query-input quadrature grid. Grid
and volume integration remain numerical. No query answers / teacher access.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import enumerate_support_modes as reference
model=reference.memory


def moments(bank,q):
    first=np.empty(len(q));second=np.empty_like(first)
    for start in range(0,len(q),256):
        h=np.broadcast_to(q[start:start+256],(len(bank),len(q[start:start+256])))
        for j in range(bank.shape[1]):h=model.base.g(h+bank[:,j,None])
        first[start:start+256]=h.mean(0);second[start:start+256]=np.mean(h*h,axis=0)
    return first,second


def integral(y,q):return np.trapezoid(y,x=q,axis=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--online',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=json.loads((a.reference/'protocol.json').read_text());coverage=json.loads((a.reference/'coverage.json').read_text())
    for name,h in old['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    online=json.loads((a.online/'episodes.json').read_text());lookup={(r['seed'],r['method']):r for r in online if r['n_context']==4}
    protocol=dict(seeds=old['seeds'],n_context=4,configs=old['configs'],replicas=4,samples_per_region_per_replica=2048,
        quadrature_points=1025,coarse_quadrature_points=513,rng_seed=991731,pairs=[[0,1],[2,3]],
        primary='observed-context conditional excess risk of frozen 512-sample predictor; ALM20 vs Adam60/240',
        secondary='ideal discovered-set truncation bias and expected 512-sample variance; no query answers are used',
        source_sha256={**old['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        reference_protocol_sha256=hashlib.sha256((a.reference/'protocol.json').read_bytes()).hexdigest(),
        scope='old tasks, numerical complete-volume reference and input quadrature; cross-replica estimators are unbiased only conditional on these numerical weights/grid; not an online selector')
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];names=[c['name'] for c in protocol['configs']]
    q=np.linspace(0,1,protocol['quadrature_points'])
    for item in coverage:
        begin=time.perf_counter();seed=item['seed'];source=a.reference/item['reference_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['reference_sha256'];ref=json.loads(source.read_text())
        assert ref['numerical_volume_reference_complete'];x=np.array(ref['x']);v=np.array(ref['v']);polys=sorted(ref['positive_regions'],key=lambda r:r['pattern'])
        volume=np.array([p['volume'] for p in polys]);weights=volume/volume.sum();means=np.empty((4,len(polys),len(q)));seconds=np.empty_like(means)
        for k,record in enumerate(polys):
            b=np.array(record['representative']);_,_,g,rhs=model.base.branch_polytope(x,v,b);poly,note=model.posterior.polytope(g,rhs)
            assert poly is not None and np.isclose(poly['volume'],record['volume'],rtol=1e-9,atol=1e-30)
            for r in range(4):
                rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'],seed,k,r]))
                bank=model.posterior.sample(poly,protocol['samples_per_region_per_replica'],rng)
                means[r,k],seconds[r,k]=moments(bank,q)
        full=np.einsum('k,rkq->rq',weights,means);fullsecond=np.einsum('k,rkq->rq',weights,seconds)
        actual=[];masks=[];state_hashes={}
        for method in names:
            stage=lookup[seed,method];path=a.online/stage['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==stage['state_sha256']
            state_hashes[method]=stage['state_sha256']
            with np.load(path) as z:actual.append(model.posterior.make_predict(z['samples'])(q))
            found=set(next(c for c in item['coverage'] if c['method']==method)['found_patterns']);masks.append([p['pattern'] in found for p in polys])
        actual=np.array(actual);masks=np.array(masks);task=[]
        for j,method in enumerate(names):
            w=weights*masks[j];alpha=w.sum();w/=alpha
            truncated=np.einsum('k,rkq->rq',w,means);truncsecond=np.einsum('k,rkq->rq',w,seconds)
            estimates=[];coarse=[]
            for left,right in protocol['pairs']:
                actual_integrand=(actual[j]-full[left])*(actual[j]-full[right])
                trunc_integrand=(truncated[left]-full[left])*(truncated[right]-full[right])
                sampling_integrand=((truncsecond[left]+truncsecond[right])/2-truncated[left]*truncated[right])/512
                bayes_integrand=(fullsecond[left]+fullsecond[right])/2-full[left]*full[right]
                estimates.append(dict(actual_excess=float(integral(actual_integrand,q)),truncation_excess=float(integral(trunc_integrand,q)),
                    expected_sampling_excess=float(integral(sampling_integrand,q)),bayes_risk=float(integral(bayes_integrand,q))))
                coarse.append(dict(actual_excess=float(integral(actual_integrand[::2],q[::2])),truncation_excess=float(integral(trunc_integrand[::2],q[::2]))))
            task.append(dict(seed=seed,method=method,posterior_mass_fraction=float(alpha),
                **{k:float(np.mean([e[k] for e in estimates])) for k in estimates[0]},
                cross_replica_estimates=estimates,
                grid_refinement_actual_difference=float(np.mean([e['actual_excess'] for e in estimates])-np.mean([e['actual_excess'] for e in coarse])),
                grid_refinement_truncation_difference=float(np.mean([e['truncation_excess'] for e in estimates])-np.mean([e['truncation_excess'] for e in coarse]))))
        filename=f'conditional_curves_{seed}.npz'
        np.savez_compressed(a.out/filename,x=x,v=v,q=q,weights=weights,patterns=np.array([p['pattern'] for p in polys]),
            region_means=means,region_second_moments=seconds,actual_predictions=actual,method_masks=masks,methods=np.array(names))
        rows.extend(task);(a.out/'risks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        meta=dict(seed=seed,curve_file=filename,curve_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),
            frozen_state_hashes=state_hashes,reference_sha256=item['reference_sha256'],elapsed_seconds=time.perf_counter()-begin)
        (a.out/f'audit_{seed}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=len(rows)//len(names),total=16,seed=seed,seconds=meta['elapsed_seconds'])),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()

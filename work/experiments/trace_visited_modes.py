"""Support-only, nonintervening audit of feasible modes lost by best-point retention."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import scalar_matched_batched_controls as model
import region_posterior_memory as geometry
base=model.base


def trace(x,v,anchor,cfg):
    old_score=base.score;old_evaluate=model.batched.evaluate;old_jac=base.forward_jacobian;visits=[]
    def score(b,*args,**kwargs):
        visits.append(b.copy())
        return old_score(b,*args,**kwargs)
    def evaluate(b,*args,**kwargs):
        visits.append(b.copy())
        return old_evaluate(b,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('global gradient called from local trajectory')
    try:
        base.score=score
        if cfg['generator']=='alm':
            base.forward_jacobian=forbidden;model.batched.evaluate=forbidden
        else:model.batched.evaluate=evaluate
        bank,_=model.discover(x,v,anchor,cfg)
    finally:
        base.score=old_score;model.batched.evaluate=old_evaluate;base.forward_jacobian=old_jac
    original,_=model.discover(x,v,anchor,cfg)
    assert np.array_equal(bank,original)
    return np.concatenate(visits),bank


def audit_modes(x,v,anchor,visited,bank):
    original={base.pattern(x,b).tobytes():b.copy() for b in bank}
    all_modes=set();feasible={}
    errors,_=base.score(visited,x,v,anchor)
    for b,error in zip(visited,errors):
        key=base.pattern(x,b).tobytes();all_modes.add(key)
        if error<=base.EPS+base.TOL and key not in feasible:feasible[key]=b.copy()
    extra={k:b for k,b in feasible.items() if k not in original}
    positive_original=[];positive_extra=[];rejections=[]
    for label,items in [('retained',original),('extra_direct_feasible',extra)]:
        for key,b in items.items():
            _,_,g,rhs=base.branch_polytope(x,v,b);poly,note=geometry.polytope(g,rhs)
            item=dict(pattern=key.hex(),representative=b.tolist(),**note)
            if poly is not None:(positive_original if label=='retained' else positive_extra).append(item)
            else:rejections.append(dict(group=label,**item))
    mass=sum(p['volume'] for p in positive_original)/.24**4
    gain=sum(p['volume'] for p in positive_extra)/.24**4
    return dict(visited_points=len(visited),unique_visited_patterns=len(all_modes),direct_feasible_visited_points=int(np.sum(errors<=base.EPS+base.TOL)),
                direct_feasible_patterns=len(feasible),retained_patterns=len(original),extra_direct_feasible_patterns=len(extra),
                positive_retained_patterns=len(positive_original),positive_extra_patterns=len(positive_extra),
                retained_absolute_prior_mass=mass,extra_absolute_prior_mass=gain,
                retained_fraction_of_observed_union=mass/(mass+gain) if mass+gain>0 else None,
                trace_array_bytes=visited.nbytes,positive_retained=positive_original,positive_extra=positive_extra,rejected=rejections)


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    original_protocol=json.loads((a.original/'protocol.json').read_text());original_rows=json.loads((a.original/'episodes.json').read_text())
    assert all(hashlib.sha256(Path(__file__).with_name(k).read_bytes()).hexdigest()==v for k,v in original_protocol['source_sha256'].items())
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    selected=['alm64_wide','alm64_priorbox','adam240_r64_wide','adam240_r64_priorbox']
    configs=[c for c in original_protocol['configs'] if c['name'] in selected]
    protocol=dict(seeds=list(range(5900000,5900016)),stages=[4,8],configs=configs,
                  source_sha256={**original_protocol['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='no queries generated; support-only trajectory audit; not new task confirmation or online speed benchmark')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    lookup={(r['method'],r['seed'],r['n_context']):r for r in original_rows}
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        for c in configs:
            for n in protocol['stages']:
                anchor=np.zeros(4) if n==4 else np.asarray(lookup[(c['name'],seed,4)]['anchor_output'])
                start=time.perf_counter();visited,bank=trace(x[:n],v[:n],anchor,c);result=audit_modes(x[:n],v[:n],anchor,visited,bank)
                old=lookup[(c['name'],seed,n)]
                assert result['positive_retained_patterns']==old['positive_volume_regions']
                assert abs(result['retained_absolute_prior_mass']-old['discovered_prior_mass'])<=1e-15*max(1,old['discovered_prior_mass'])
                filename=f'{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(a.out/filename,visited=visited,bank=bank,x=x[:n],v=v[:n],anchor=anchor)
                rows.append(dict(method=c['name'],seed=seed,n_context=n,original_bank_exact_equal=True,
                                 original_geometry_matches=True,trace_file=filename,trace_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),
                                 diagnostic_seconds=time.perf_counter()-start,**result))
        (a.out/'modes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-5900000+1,total=16,rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()

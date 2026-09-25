"""Independent local-minimum, domain, and frozen-old-trajectory checks."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import bounded_activity_memory as model
block=model.activity;base=model.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
    base.BOUND=.2;rng=np.random.default_rng(731619);count=160;width=8
    center=rng.normal(size=count)*2;old=rng.uniform(-1,1,count);target=rng.normal(size=(count,width))*2
    offset=rng.normal(size=(count,width));coef=rng.normal(size=(count,width));coef[::3,::3]=0
    lower=rng.uniform(-1,-.2,count);upper=rng.uniform(.2,1,count);gaps=[];descent=[]
    for bounded in [False,True]:
        answer,meta=block.solve(center,target,offset,coef,old,lower=lower if bounded else None,upper=upper if bounded else None)
        for i in range(count):
            incumbent=np.clip(old[i],lower[i],upper[i]) if bounded else old[i]
            energy=lambda value:float((value-center[i])**2+.01*(value-old[i])**2+np.sum((base.family.activation(offset[i]+coef[i]*value)-target[i])**2))
            radius=np.sqrt(energy(incumbent));lo=center[i]-radius;hi=center[i]+radius
            if bounded:lo=max(lo,lower[i]);hi=min(hi,upper[i])
            roots=[lo,hi]
            for a,c in zip(coef[i],offset[i]):
                if a!=0:
                    roots.extend((k-c)/a for k in [-1.,0.,1.] if lo<(k-c)/a<hi)
            roots=sorted(set(roots));best=min(energy(t) for t in roots)
            for left,right in zip(roots[:-1],roots[1:]):
                solved=minimize_scalar(energy,bounds=(left,right),method='bounded',options={'xatol':1e-13})
                best=min(best,float(solved.fun))
            gap=energy(answer[i])-best;gaps.append(gap);descent.append(energy(answer[i])-energy(incumbent))
            assert abs(gap)<1e-8,(bounded,i,gap)
        if bounded:assert np.all(answer>=lower-1e-12) and np.all(answer<=upper+1e-12)
    weights=base.family.make_weights(3,8);x=rng.uniform(-1,1,(17,8));lo,hi=block.interval_bounds(x,weights,.2)
    violation=0.
    for _ in range(200):
        biases=rng.uniform(-.2,.2,(3,8));h=x
        for j,weight in enumerate(weights):
            h=base.family.activation(h@weight.T+biases[j]);violation=max(violation,float(np.max(np.maximum(lo[j]-h,h-hi[j]))))
    assert violation<1e-12
    # Identical old orthogonal update, now with domain-violation instrumentation.
    task=np.random.default_rng(5400000);truth=task.uniform(-.2,.2,(3,8));x=task.uniform(-1,1,(24,8))[:8]
    v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(24400000).uniform(-base.EPS,base.EPS,x.shape)
    anchor=np.zeros_like(truth);starts,_=base.proposals(x,v,anchor,weights)
    oldbank,_=model.original.local(starts,x,v,weights,anchor,sweeps=32)
    traced,tm=model.local(starts,x,v,weights,anchor,sweeps=32,activity_mode='orthogonal')
    difference=float(np.max(np.abs(oldbank-traced)));assert difference<1e-12,difference
    # The fitting core must not call the full-chain objective/gradient interface.
    original_evaluate=base.evaluate
    def forbidden(*args,**kwargs):raise AssertionError('global gradient/evaluate entered candidate fitting')
    base.evaluate=forbidden
    timings=[]
    try:
        for mode in ['bounded_coordinate','unbounded_coordinate','clipped_orthogonal']:
            before=time.perf_counter();_,meta=model.local(starts,x,v,weights,anchor,sweeps=8,activity_mode=mode)
            timings.append(dict(mode=mode,seconds=time.perf_counter()-before,**meta))
            if mode!='unbounded_coordinate':assert meta['maximum_activity_domain_violation']<1e-9
    finally:base.evaluate=original_evaluate
    result=dict(passed=True,scalar_checks=len(gaps),maximum_absolute_energy_gap=max(abs(g) for g in gaps),
                maximum_energy_increase=max(descent),prior_domain_checks=600,domain_violation=violation,
                old_trajectory_max_parameter_gap=difference,old_domain_diagnostics=tm,local_no_global_gradient_checks=timings,
                source_sha256={Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in
                               [__file__,model.__file__,block.__file__,model.original.__file__,model.original.original.__file__,
                                model.original.first.__file__,model.original.bias_solver.__file__,base.__file__,base.family.__file__]})
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

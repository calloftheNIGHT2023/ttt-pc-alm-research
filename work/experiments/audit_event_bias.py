import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import event_bias_block as fast
import batched_bias_block as slow
import event_affine_memory as model
import affine_eliminated_memory as original
base=model.base
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists();args.out.parent.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(667057);max_energy=0.;max_bias=0.;cases=0;timings=[]
for n in [1,2,8,24,64]:
    for rep in range(8):
        c=rng.uniform(-2,2,(4,n,8));target=rng.uniform(-3,3,c.shape);old=rng.uniform(-.2,.2,(4,8))
        if rep==0:c[:,0]=np.array([-.8,-1.,-1.2,-.2,0.,.2,.8,1.2])[None]
        b1,m1=slow.solve(c,target,old);b2,m2=fast.solve(c,target,old)
        max_energy=max(max_energy,float(np.max(np.abs(m1['conditional_objective']-m2['conditional_objective']))))
        max_bias=max(max_bias,float(np.max(np.abs(b1-b2))));cases+=b1.size
        assert np.allclose(m1['conditional_objective'],m2['conditional_objective'],rtol=1e-12,atol=1e-10)
    c=rng.uniform(-2,2,(16,n,8));target=rng.uniform(-3,3,c.shape);old=rng.uniform(-.2,.2,(16,8))
    entry={'contexts':n}
    for name,solver in [('old_quadratic',slow.solve),('new_events',fast.solve)]:
        solver(c,target,old);runs=[]
        for _ in range(20):
            start=time.perf_counter();_,meta=solver(c,target,old);runs.append(time.perf_counter()-start)
        entry[name]={'median_seconds':float(np.median(runs)),'workspace_subtotal':meta['major_workspace_bytes_subtotal']}
    timings.append(entry)
weights=base.family.make_weights(3,8);rng=np.random.default_rng(5400000);truth=rng.uniform(-base.PRIOR,base.PRIOR,(3,8));x=rng.uniform(-1,1,(24,8))
v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(24400000).uniform(-base.EPS,base.EPS,x.shape);anchor=np.zeros_like(truth)
base.BOUND=.2;starts,_=base.proposals(x[:8],v[:8],anchor,weights)
one,_=original.local(starts,x[:8],v[:8],weights,anchor,sweeps=240)
two,_=model.local(starts,x[:8],v[:8],weights,anchor,sweeps=240)
bank_difference=float(np.max(np.abs(one-two)))
point1=base.select(np.concatenate([starts,one]),x[:8],v[:8],weights,anchor);point2=base.select(np.concatenate([starts,two]),x[:8],v[:8],weights,anchor)
point_difference=float(np.max(np.abs(point1-point2)));assert point_difference<1e-9
result={'passed':True,'scalar_cases':cases,'max_energy_difference':max_energy,'max_bias_difference':max_bias,
    'full_240_sweep_bank_difference':bank_difference,'selected_parameter_difference':point_difference,'timings':timings,
    'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,fast.__file__,model.__file__,original.__file__]},
    'scope':'same conditional objective; floating trajectories can differ near ties; one-stream full-path audit'}
args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import affine_eliminated_memory as m
base=m.base
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists();args.out.parent.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(665656);max_energy_gap=0.;max_bias_gap=0.;cases=0
for n in [1,2,8,24]:
    for _ in range(5):
        c=rng.uniform(-2,2,(3,n,4));target=rng.uniform(-2,2,c.shape);old=rng.uniform(-.2,.2,(3,4))
        batch,meta=m.bias_solver.solve(c,target,old)
        for ri in range(3):
            ref,refmeta=m.first.solve(m.first.prepare(c[ri],bound=.2),target[ri:ri+1],old[ri:ri+1])
            gap=float(np.max(np.abs(refmeta['conditional_objective'][0]-meta['conditional_objective'][ri])))
            max_energy_gap=max(max_energy_gap,gap);max_bias_gap=max(max_bias_gap,float(np.max(np.abs(batch[ri]-ref[0]))));cases+=4
            assert gap<1e-8
# Orthogonal local activity block: exact change of coordinates and random feasible candidates.
weights=base.family.make_weights(3,4);w=weights[1];activity_cases=0;max_identity_gap=0.
for _ in range(40):
    center,target,old=rng.uniform(-2,2,(3,1,3,4));bias=rng.uniform(-.2,.2,(1,1,4));trust=.01
    uc=center@w.T+bias;uo=old@w.T+bias;u=base.family.nonlinear_prox(uc,target,uo,trust);solution=(u-bias)@w
    def energy(h):return np.sum((h-center)**2+(base.family.activation(h@w.T+bias)-target)**2+trust*(h-old)**2)
    transformed=float(np.sum((u-uc)**2+(base.family.activation(u)-target)**2+trust*(u-uo)**2))
    gap=abs(float(energy(solution))-transformed);max_identity_gap=max(max_identity_gap,gap);assert gap<1e-10
    assert energy(solution)<=energy(old)+1e-10
    # Random directions plus local numerical optimizer independently check the objective.
    check=minimize(lambda flat:energy(flat.reshape(old.shape)),old.ravel(),method='BFGS',options={'gtol':1e-8,'maxiter':300})
    assert energy(solution)<=check.fun+1e-8;activity_cases+=1
original=base.evaluate
try:
    def forbidden(*a,**k):raise RuntimeError('global BP forbidden')
    base.evaluate=forbidden;x=rng.uniform(-1,1,(8,4));anchor=np.zeros((3,4));v=rng.uniform(-1,1,(8,4))
    out,_=m.local(anchor[None],x,v,weights,anchor,sweeps=3);assert np.isfinite(out).all()
finally:base.evaluate=original
result={'passed':True,'bias_scalar_cases':cases,'max_bias_energy_difference':max_energy_gap,'max_bias_parameter_difference':max_bias_gap,
    'activity_block_cases':activity_cases,'max_orthogonal_energy_identity_error':max_identity_gap,'candidate_global_bp_guard':True,
    'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,m.__file__,m.bias_solver.__file__]},
    'scope':'conditional block proofs and numerical checks, not whole-network optimality'}
args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

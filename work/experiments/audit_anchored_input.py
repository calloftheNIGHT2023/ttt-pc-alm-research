import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
import anchored_input_block as block
import anchored_input_memory as memory


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    assert not args.out.exists();rng=np.random.default_rng(662554);maxgap=0.;samegap=0.;cases=0
    for n in [1,2,8,24]:
        for rep in range(5):
            c=rng.uniform(-1.5,1.5,(n,4));target=rng.uniform(-2,2,(3,n,4));old=rng.uniform(-.3,.3,(3,4));cache=block.prepare(c)
            answer,meta=block.solve(cache,target,old)
            for ri in range(3):
                apart,_=block.solve(cache,target[ri:ri+1],old[ri:ri+1]);samegap=max(samegap,float(np.max(np.abs(apart-answer[ri:ri+1]))))
                for channel in range(4):
                    def objective(b):return np.sum((block.activation(c[:,channel]+b[0])-target[ri,:,channel])**2)+.01*n*(b[0]-old[ri,channel])**2
                    result=differential_evolution(objective,[(-.3,.3)],seed=872,atol=1e-11,tol=1e-11,popsize=20,maxiter=1000,polish=True)
                    gap=objective([answer[ri,channel]])-result.fun;maxgap=max(maxgap,float(gap));assert gap<1e-8
                    assert objective([answer[ri,channel]])<=objective([old[ri,channel]])+1e-9;cases+=1
    base=memory.base;original=base.evaluate
    try:
        def forbidden(*a,**k):raise RuntimeError('global BP forbidden')
        base.evaluate=forbidden;weights=base.family.make_weights(3,4);x=rng.uniform(-1,1,(8,4));v=rng.uniform(-1,1,(8,4));anchor=np.zeros((3,4));starts=np.zeros((2,3,4))
        bank,meta=memory.local(starts,x,v,weights,anchor,sweeps=3);assert np.isfinite(bank).all()
    finally:base.evaluate=original
    result={'passed':True,'scalar_cases':cases,'max_energy_excess_vs_independent_de':maxgap,'batch_vs_individual_bias_gap':samegap,
        'candidate_global_bp_guard':True,'input_affine_invariant':meta['input_affine_residual_exactly_zero'],
        'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,block.__file__,memory.__file__]},
        'scope':'random independent optimizer checks supplement the finite-piece proof; DE is not an optimality certificate'}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

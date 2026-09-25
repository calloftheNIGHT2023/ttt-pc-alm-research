"""Evaluator-only indistinguishable-task witnesses in the actual task family.

Not a fitting procedure; true parameters are used only to exhibit information
ambiguity. Two-point minimax bound is not a population Bayes-risk estimate.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import vector_interval_memory as m


def constraints(point,x,weights):
    d,w=point.shape;h=x.copy();jac=np.zeros((len(x),w,d*w));matrices=[];rhs=[]
    for layer,weight in enumerate(weights):
        z=h@weight.T+point[layer]
        jz=np.einsum('ab,nbp->nap',weight,jac,optimize=True);jz[:,np.arange(w),layer*w+np.arange(w)]+=1
        region=np.searchsorted([-1.,0.,1.],z)
        lo=np.array([-np.inf,-1.,0.,1.])[region];hi=np.array([-1.,0.,1.,np.inf])[region]
        for sign,bound in [(1.,hi),(-1.,-lo)]:
            finite=np.isfinite(bound);matrices.append(sign*jz[finite]);rhs.append(bound[finite]-sign*z[finite])
        jac=jz*m.family.deriv(z)[:,:,None];h=m.family.activation(z)
    return jac.reshape(-1,d*w),np.concatenate(matrices),np.concatenate(rhs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists();args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[];weights=m.family.make_weights(3,8)
    for seed in range(5400000,5400012):
        rng=np.random.default_rng(seed);truth=rng.uniform(-m.PRIOR,m.PRIOR,(3,8));x=rng.uniform(-1,1,(24,8));q=rng.uniform(-1,1,(512,8))
        jac,a,margin=constraints(truth,x[:8],weights);_,sv,vh=np.linalg.svd(jac,full_matrices=False);rank=int(np.sum(sv>1e-9))
        record={'seed':seed,'rank':rank,'parameter_count':24}
        if rank<24:
            direction=vh[-1];idx=int(np.flatnonzero(np.abs(direction)>1e-10)[0])
            if direction[idx]<0:direction=-direction
            coefficient=np.r_[a@direction,direction,-direction]
            upper=np.r_[margin,m.PRIOR-truth.ravel(),m.PRIOR+truth.ravel()]
            low=-np.inf;high=np.inf
            for coef,lim in zip(coefficient,upper):
                if coef>1e-12:high=min(high,lim/coef)
                elif coef<-1e-12:low=max(low,lim/coef)
                else:assert lim>=-1e-10
            assert np.isfinite(low) and np.isfinite(high) and low<0<high
            lower=truth+.95*low*direction.reshape(3,8);higher=truth+.95*high*direction.reshape(3,8)
            observed=m.forward(np.stack([lower,higher]),x[:8],weights);pred=m.forward(np.stack([lower,higher]),q,weights)
            gap=float(np.max(np.abs(observed[0]-observed[1])));assert gap<1e-10
            assert np.max(np.abs(np.stack([lower,higher])))<=m.PRIOR+1e-12
            separation=float(np.mean((pred[0]-pred[1])**2))
            record.update(null_residual=float(np.linalg.norm(jac@direction)),feasible_direction_interval=[float(low),float(high)],
                max_observed_output_difference=gap,query_function_separation=separation,two_point_minimax_lower_bound=separation/4,
                lower_task=lower.tolist(),upper_task=higher.tolist(),null_direction=direction.tolist())
        records.append(record)
    result={'records':records,'rank_deficient_tasks':sum(r['rank']<24 for r in records),
        'scope':'two tasks within prior support with identical noiseless contexts; equal mixture lower bound, not the original uniform-prior population risk',
        'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,m.__file__,m.family.__file__]}}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'rank_deficient_tasks':result['rank_deficient_tasks'],
        'witnesses':[{k:v for k,v in r.items() if k not in ['lower_task','upper_task','null_direction']} for r in records if r['rank']<24]}),flush=True)


if __name__=='__main__':main()

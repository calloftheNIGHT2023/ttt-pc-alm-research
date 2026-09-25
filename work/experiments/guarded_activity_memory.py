"""Feasible joint activity proposals with a local-energy safeguard.

Guarantee is conditional nonincrease, not global optimization or query risk.
"""
import numpy as np
import bounded_activity_memory as reference
base=reference.base
original=reference.original


def update(old,center,target,weight,bias,low,high,trust=.01,gradient=True):
    def energy(value):
        return np.sum((value-center)**2+trust*(value-old)**2,axis=-1)+np.sum(
            (base.family.activation(value@weight.T+bias)-target)**2,axis=-1)
    old_u=old@weight.T+bias;u_center=center@weight.T+bias
    u=base.family.nonlinear_prox(u_center,target,old_u,trust)
    clipped=np.clip((u-bias)@weight,low,high)
    proposals=[old,clipped]
    if gradient:
        for value in [old,clipped]:
            z=value@weight.T+bias
            partial=value-center+trust*(value-old)+(base.family.deriv(z)*(base.family.activation(z)-target))@weight
            proposals.append(np.clip(value-partial/(5+trust),low,high))
    values=np.array([energy(value) for value in proposals]);index=np.argmin(values,axis=0)
    stack=np.array(proposals);answer=np.take_along_axis(stack,index[None,...,None],axis=0)[0]
    selected=np.take_along_axis(values,index[None],axis=0)[0]
    return answer,dict(clipped_worse_than_old=int(np.sum(values[1]>values[0]+1e-10)),
                       retained_old=int(np.sum(index==0)),block_samples=index.size,
                       max_old_energy_increase=float(np.max(selected-values[0])),
                       max_clipped_energy_increase=float(np.max(selected-values[1])),
                       proposal_workspace_bytes=stack.nbytes+values.nbytes+index.nbytes)


def local(starts,x,v,weights,anchor,*,sweeps=32,gradient=True,dual_rate=.5):
    b=starts.copy();r,d,w=b.shape;n=len(x);trust=.01
    h=np.empty((d,r,n,w));a=np.zeros_like(h);previous=x
    for j,weight in enumerate(weights):h[j]=base.family.activation(previous@weight.T+b[:,j,None,:]);previous=h[j]
    low,high=reference.activity.interval_bounds(x,weights,base.BOUND)
    cache=original.first.prepare(x@weights[0].T,base.BOUND,trust)
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor)
    counts=dict(clipped_worse_than_old=0,retained_old=0,block_samples=0);workspace=0;maximum_increase=0.
    for _ in range(sweeps):
        for j in reversed(range(d)):
            previous=x if j==0 else h[j-1]
            center=base.family.activation(previous@weights[j].T+b[:,j,None,:])-a[j]
            if j==d-1:h[j]=np.clip((center+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                h[j],meta=update(h[j],center,h[j+1]+a[j+1],weights[j+1],b[:,j+1,None,:],low[j],high[j],trust,gradient)
                for key in counts:counts[key]+=meta[key]
                workspace=max(workspace,meta['proposal_workspace_bytes'])
                maximum_increase=max(maximum_increase,meta['max_old_energy_increase'],meta['max_clipped_energy_increase'])
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];c=previous@weight.T;target=h[j]+a[j]
            if j==0:b[:,j],_=original.first.solve(cache,target,b[:,j])
            else:b[:,j],_=original.bias_solver.solve(c,target,b[:,j],trust,base.BOUND)
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];a[j]+=dual_rate*(h[j]-base.family.activation(previous@weight.T+b[:,j,None,:]))
        err,move=base.score(b,x,v,weights,anchor);change=base.better(err,move,errors,moves)
        best[change]=b[change];errors[change]=err[change];moves[change]=move[change]
    return best,dict(sweeps=sweeps,gradient_proposals=gradient,dual_rate=dual_rate,**counts,
                     maximum_conditional_energy_increase=maximum_increase,proposal_workspace_bytes=workspace,
                     major_state_arrays_subtotal=sum(z.nbytes for z in [b,h,a,best,low,high]))


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights,features=cfg.get('features',256),restarts=cfg.get('restarts',16))
    bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],gradient=cfg.get('gradient',True),dual_rate=cfg.get('dual_rate',.5))
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,dict(**meta,**more,persistent_state_bytes=point.nbytes,
                                                                 anchor_output=point.tolist(),known_weights_bytes=weights.nbytes)

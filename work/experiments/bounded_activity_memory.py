"""Valid-domain local activity updates; exact coordinate vs orthogonal block control."""
import numpy as np
import event_affine_memory as original
import bounded_activity_block as activity
base = original.base


def local(starts,x,v,weights,anchor,*,sweeps=32,activity_mode='bounded_coordinate',dual_rate=.5):
    b=starts.copy();r,d,w=b.shape;n=len(x);trust=.01
    h=np.empty((d,r,n,w));a=np.zeros_like(h);previous=x
    for j,weight in enumerate(weights):
        h[j]=base.family.activation(previous@weight.T+b[:,j,None,:]);previous=h[j]
    lows,highs=activity.interval_bounds(x,weights,base.BOUND)
    first_input=x@weights[0].T;cache=original.first.prepare(first_input,base.BOUND,trust)
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor)
    bias_workspace=0;activity_workspace=0;unreachable_count=0;activity_count=0;maximum_violation=0.
    for _ in range(sweeps):
        for j in reversed(range(d)):
            previous=x if j==0 else h[j-1]
            center=base.family.activation(previous@weights[j].T+b[:,j,None,:])-a[j]
            if j==d-1:
                h[j]=np.clip((center+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            elif activity_mode in ['bounded_coordinate','unbounded_coordinate']:
                block=h[j].reshape(r*n,w);block_center=center.reshape(r*n,w)
                target=(h[j+1]+a[j+1]).reshape(r*n,w)
                next_z=block@weights[j+1].T+np.repeat(b[:,j+1],n,axis=0)
                low=np.broadcast_to(lows[j],(r,n,w)).reshape(r*n,w)
                high=np.broadcast_to(highs[j],(r,n,w)).reshape(r*n,w)
                for coordinate in range(w):
                    coef=weights[j+1][:,coordinate]
                    old=block[:,coordinate].copy();offset=next_z-old[:,None]*coef
                    value,meta=activity.solve(block_center[:,coordinate],target,offset,coef,old,trust,
                                              low[:,coordinate] if activity_mode=='bounded_coordinate' else None,
                                              high[:,coordinate] if activity_mode=='bounded_coordinate' else None)
                    block[:,coordinate]=value;next_z+=(value-old)[:,None]*coef
                    activity_workspace=max(activity_workspace,meta['working_array_bytes_subtotal'])
            else:
                target=h[j+1]+a[j+1];old_u=h[j]@weights[j+1].T+b[:,j+1,None,:]
                u_center=center@weights[j+1].T+b[:,j+1,None,:]
                u=base.family.nonlinear_prox(u_center,target,old_u,trust)
                h[j]=(u-b[:,j+1,None,:])@weights[j+1]
                if activity_mode=='clipped_orthogonal':h[j]=np.clip(h[j],lows[j],highs[j])
                elif activity_mode!='orthogonal':raise ValueError(activity_mode)
            if j<d-1:
                violation=np.maximum(lows[j]-h[j],h[j]-highs[j])
                unreachable_count+=int(np.sum(violation>1e-9));activity_count+=violation.size
                maximum_violation=max(maximum_violation,float(np.max(violation)))
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];c=previous@weight.T;target=h[j]+a[j]
            if j==0:b[:,j],_=original.first.solve(cache,target,b[:,j])
            else:
                b[:,j],meta=original.bias_solver.solve(c,target,b[:,j],trust,base.BOUND)
                bias_workspace=max(bias_workspace,meta['major_workspace_bytes_subtotal'])
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1]
            a[j]+=dual_rate*(h[j]-base.family.activation(previous@weight.T+b[:,j,None,:]))
        err,move=base.score(b,x,v,weights,anchor);update=base.better(err,move,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=move[update]
    return best,dict(sweeps=sweeps,activity_mode=activity_mode,dual_rate=dual_rate,
                     unreachable_activity_fraction=unreachable_count/max(activity_count,1),
                     maximum_activity_domain_violation=maximum_violation,
                     activity_workspace_bytes_subtotal=activity_workspace,bias_workspace_bytes_subtotal=bias_workspace,
                     domain_bounds_bytes=lows.nbytes+highs.nbytes,
                     major_state_arrays_subtotal=sum(z.nbytes for z in [b,h,a,best,lows,highs]))


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights,features=cfg.get('features',256),restarts=cfg.get('restarts',16))
    bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],activity_mode=cfg['activity_mode'],dual_rate=cfg.get('dual_rate',.5))
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,dict(**meta,**more,persistent_state_bytes=point.nbytes,
                                                                 anchor_output=point.tolist(),known_weights_bytes=weights.nbytes)

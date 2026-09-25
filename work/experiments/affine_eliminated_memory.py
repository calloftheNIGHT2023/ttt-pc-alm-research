"""Local ALM with all affine relations exact, orthogonal mixing matrices.

Only nonlinear constraints have independent activities/multipliers.
"""
import numpy as np
import vector_interval_memory as base
import anchored_input_block as first
import batched_bias_block as bias_solver


def local(starts,x,v,weights,anchor,*,sweeps=240,dual_rate=.5,gradient_activity=False,gradient_bias=False):
    b=starts.copy();r,d,w=b.shape;n=len(x);trust=.01
    h=np.empty((d,r,n,w));a=np.zeros_like(h);previous=x
    for j,weight in enumerate(weights):h[j]=base.family.activation(previous@weight.T+b[:,j,None,:]);previous=h[j]
    first_input=x@weights[0].T;cache=first.prepare(first_input,base.BOUND,trust)
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor);workspace=0;cache_bytes=0
    for _ in range(sweeps):
        for j in reversed(range(d)):
            previous=x if j==0 else h[j-1]
            center=base.family.activation(previous@weights[j].T+b[:,j,None,:])-a[j]
            if j==d-1:
                h[j]=np.clip((center+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                target=h[j+1]+a[j+1];old_u=h[j]@weights[j+1].T+b[:,j+1,None,:]
                if gradient_activity:
                    grad=h[j]-center+(base.family.deriv(old_u)*(base.family.activation(old_u)-target))@weights[j+1]
                    h[j]=h[j]-grad/(5+trust)
                else:
                    u_center=center@weights[j+1].T+b[:,j+1,None,:]
                    u=base.family.nonlinear_prox(u_center,target,old_u,trust)
                    h[j]=(u-b[:,j+1,None,:])@weights[j+1]
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];c=previous@weight.T;target=h[j]+a[j]
            if gradient_bias:
                z=c+b[:,j,None,:];grad=np.sum(base.family.deriv(z)*(base.family.activation(z)-target),axis=1)
                b[:,j]=np.clip(b[:,j]-grad/(n*(4+trust)),-base.BOUND,base.BOUND)
            elif j==0:
                b[:,j],meta=first.solve(cache,target,b[:,j]);cache_bytes=meta['cache_bytes']
            else:
                b[:,j],meta=bias_solver.solve(c,target,b[:,j],trust,base.BOUND);workspace=max(workspace,meta['major_workspace_bytes_subtotal'])
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1]
            a[j]+=dual_rate*(h[j]-base.family.activation(previous@weight.T+b[:,j,None,:]))
        err,move=base.score(b,x,v,weights,anchor);update=base.better(err,move,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=move[update]
    return best,{'sweeps':sweeps,'dual_rate':dual_rate,'gradient_activity':gradient_activity,'gradient_bias':gradient_bias,
        'affine_variables_eliminated':True,'major_state_arrays_subtotal':sum(t.nbytes for t in [b,h,a,best])+cache_bytes,
        'bias_workspace_bytes_subtotal':workspace,'first_input_cache_bytes':cache_bytes}


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights,features=cfg.get('features',256),restarts=cfg.get('restarts',16))
    bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],dual_rate=cfg.get('dual_rate',.5),
        gradient_activity=cfg.get('gradient_activity',False),gradient_bias=cfg.get('gradient_bias',False))
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,{**meta,**more,'persistent_state_bytes':point.nbytes,
        'known_weights_bytes':weights.nbytes,'anchor_output':point.tolist()}

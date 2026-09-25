"""Eliminate first affine residual exactly; retain local ALM elsewhere."""
import numpy as np
import vector_interval_memory as base
import anchored_input_block as exact


def local(starts,x,v,weights,anchor,*,sweeps=240,dual_rate=.5,gradient_first=False):
    b=starts.copy();r,d,width=b.shape;n=len(x);trust=.01
    z=np.empty((d,r,n,width));h=np.empty_like(z);p=np.zeros_like(z);a=np.zeros_like(z);previous=x
    for j,w in enumerate(weights):z[j]=previous@w.T+b[:,j,None,:];h[j]=base.family.activation(z[j]);previous=h[j]
    first_input=x@weights[0].T;cache=exact.prepare(first_input,base.BOUND,trust)
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor);cache_bytes=0;max_intervals=0
    for _ in range(sweeps):
        for j in reversed(range(d)):
            target=base.family.activation(z[j])-a[j]
            if j==d-1:h[j]=np.clip((target+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                t=z[j+1]-b[:,j+1,None,:]+p[j+1]
                h[j]=np.clip((target+t@weights[j+1]+trust*h[j])/(2+trust),-1,1)
            target=h[j]+a[j]
            if j==0:
                if gradient_first:
                    grad=np.sum(base.family.deriv(z[0])*(base.family.activation(z[0])-target),axis=1)
                    b[:,0]=np.clip(b[:,0]-grad/(n*(4+trust)),-base.BOUND,base.BOUND)
                else:
                    b[:,0],meta=exact.solve(cache,target,b[:,0]);cache_bytes=meta['cache_bytes'];max_intervals=meta['max_intervals']
                z[0]=first_input[None]+b[:,0,None,:]
            else:
                center=h[j-1]@weights[j].T+b[:,j,None,:]-p[j]
                z[j]=base.family.nonlinear_prox(center,target,z[j],trust)
        for j in range(1,d):b[:,j]=np.clip(np.mean(z[j]-h[j-1]@weights[j].T+p[j],axis=1),-base.BOUND,base.BOUND)
        for j,w in enumerate(weights):
            if j>0:p[j]+=dual_rate*(z[j]-h[j-1]@w.T-b[:,j,None,:])
            a[j]+=dual_rate*(h[j]-base.family.activation(z[j]))
        assert np.array_equal(z[0],first_input[None]+b[:,0,None,:]) and np.count_nonzero(p[0])==0
        err,move=base.score(b,x,v,weights,anchor);update=base.better(err,move,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=move[update]
    return best,{'sweeps':sweeps,'dual_rate':dual_rate,'gradient_first':gradient_first,'input_affine_residual_exactly_zero':True,
        'input_block_cache_bytes':cache_bytes,'input_block_max_intervals':max_intervals,
        'major_arrays_bytes_subtotal':sum(t.nbytes for t in [b,z,h,p,a,best])+cache_bytes}


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights)
    bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],dual_rate=cfg.get('dual_rate',.5),gradient_first=cfg.get('gradient_first',False))
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,{**meta,**more,'persistent_state_bytes':point.nbytes,
        'known_weights_bytes':weights.nbytes,'anchor_output':point.tolist()}

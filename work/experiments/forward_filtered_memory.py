"""Local or BP proposals with the SAME support-only forward segment selector.

Filtering changes the trajectory. Local activities are retained and duals use
the accepted bias; there is no hidden BP initialization or preliminary BP fit.
"""
import time
import numpy as np
import event_affine_memory as original
import guarded_activity_memory as guarded
import forward_line_solver as line
base=original.base


def filter_step(old,proposal,x,v,weights,mode,counters,history):
    before=time.perf_counter();point,alpha,meta=line.solve(old,proposal,x,v,weights,mode)
    counters['filter_seconds']+=time.perf_counter()-before
    counters['proposals']+=len(alpha);counters['zero_steps']+=int(np.sum(alpha==0));counters['partial_steps']+=int(np.sum((alpha>0)&(alpha<1)))
    counters['sum_alpha']+=float(alpha.sum());counters['filter_workspace_bytes_subtotal']=max(counters['filter_workspace_bytes_subtotal'],meta.get('filter_workspace_bytes_subtotal',0))
    counters['max_line_segments']=max(counters['max_line_segments'],meta.get('max_line_segments',0))
    if history is not None:history.append(dict(before=line.values(old,x,v,weights).tolist(),after=line.values(point,x,v,weights).tolist()))
    return point


def counters():return dict(filter_seconds=0.,proposals=0,zero_steps=0,partial_steps=0,sum_alpha=0.,filter_workspace_bytes_subtotal=0,max_line_segments=0)


def local(starts,x,v,weights,anchor,*,sweeps=32,mode='grid',guard=False,history=None):
    b=starts.copy();r,d,w=b.shape;trust=.01
    h=np.empty((d,r,len(x),w));a=np.zeros_like(h);previous=x
    for j,weight in enumerate(weights):h[j]=base.family.activation(previous@weight.T+b[:,j,None,:]);previous=h[j]
    low,high=guarded.reference.activity.interval_bounds(x,weights,base.BOUND)
    cache=original.first.prepare(x@weights[0].T,base.BOUND,trust);more=counters()
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor)
    for _ in range(sweeps):
        old=b.copy()
        for j in reversed(range(d)):
            previous=x if j==0 else h[j-1];center=base.family.activation(previous@weights[j].T+b[:,j,None,:])-a[j]
            if j==d-1:h[j]=np.clip((center+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            elif guard:h[j],_=guarded.update(h[j],center,h[j+1]+a[j+1],weights[j+1],b[:,j+1,None,:],low[j],high[j],trust,True)
            else:
                target=h[j+1]+a[j+1];old_u=h[j]@weights[j+1].T+b[:,j+1,None,:]
                u_center=center@weights[j+1].T+b[:,j+1,None,:]
                u=base.family.nonlinear_prox(u_center,target,old_u,trust);h[j]=(u-b[:,j+1,None,:])@weights[j+1]
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];c=previous@weight.T;target=h[j]+a[j]
            if j==0:b[:,j],_=original.first.solve(cache,target,b[:,j])
            else:b[:,j],_=original.bias_solver.solve(c,target,b[:,j],trust,base.BOUND)
        b=filter_step(old,b,x,v,weights,mode,more,history)
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];a[j]+=.5*(h[j]-base.family.activation(previous@weight.T+b[:,j,None,:]))
        err,move=base.score(b,x,v,weights,anchor);change=base.better(err,move,errors,moves)
        best[change]=b[change];errors[change]=err[change];moves[change]=move[change]
    return best,dict(**more,sweeps=sweeps,filter_mode=mode,guarded_activity=guard,
                     major_state_arrays_subtotal=sum(z.nbytes for z in [b,h,a,best,low,high]))


def adam(starts,x,v,weights,anchor,*,steps=64,mode='grid',history=None):
    b=starts.copy();best=b.copy();errors,moves=base.score(best,x,v,weights,anchor)
    first=np.zeros_like(b);second=first.copy();more=counters()
    def retain(params,raw):
        err=np.max(np.abs(raw),axis=(1,2));move=.5*np.sum((params-anchor)**2,axis=(1,2));change=base.better(err,move,errors,moves)
        best[change]=params[change];errors[change]=err[change];moves[change]=move[change]
    for iteration in range(1,steps+1):
        _,grad,raw=base.evaluate(b,x,v,weights);retain(b,raw)
        first=.9*first+.1*grad;second=.999*second+.001*grad**2
        proposal=np.clip(b-.01*(first/(1-.9**iteration))/(np.sqrt(second/(1-.999**iteration))+1e-8),-base.BOUND,base.BOUND)
        b=filter_step(b,proposal,x,v,weights,mode,more,history)
    retain(b,base.forward(b,x,weights)-v)
    return best,dict(**more,steps=steps,filter_mode=mode,major_state_arrays_subtotal=sum(z.nbytes for z in [b,best,first,second]))


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights,features=cfg.get('features',256),restarts=cfg.get('restarts',16))
    if cfg['method']=='filtered_local':bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],mode=cfg['filter'],guard=cfg.get('guard',False))
    else:bank,more=adam(starts,x,v,weights,anchor,steps=cfg['steps'],mode=cfg['filter'])
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,dict(**meta,**more,persistent_state_bytes=point.nbytes,
                                                                 known_weights_bytes=weights.nbytes,anchor_output=point.tolist())

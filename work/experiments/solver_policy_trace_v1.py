"""313 non-mutating instrumentation of the frozen exact local ALM sweep.

Policies are conservative execution partitions, NOT minimal affine-map IDs.
All arithmetic producing b/h/u retains the original NumPy operation order.
"""
import numpy as np
import cold_stagnation_switch as cold

base=cold.base


def regions(z):return np.searchsorted(base.KNOTS,z,side='right')


def cmp(a,b):
    # Separate equality: it matters for clipping/sorting and deterministic ties.
    return (np.asarray(a)>b).astype(np.int16)-(np.asarray(a)<b).astype(np.int16)


class Recorder:
    def __init__(self,restarts):self.restarts=restarts;self.parts={};self.schema={}
    def add(self,group,name,value,axis=0):
        arr=np.moveaxis(np.asarray(value),axis,0)
        assert arr.shape[0]==self.restarts,(group,name,arr.shape)
        arr=arr.reshape(self.restarts,-1).astype(np.int16)
        self.parts.setdefault(group,[]).append(arr)
        self.schema.setdefault(group,[]).append(dict(name=name,width=arr.shape[1]))
    def finish(self):return {k:np.concatenate(v,axis=1) for k,v in self.parts.items()},self.schema


def bias(previous,target,old,anchor,rho,trust,rec,group):
    restarts,n=previous.shape
    reg=np.searchsorted(base.KNOTS,previous-base.BOUND,side='right')
    s0=base.SLOPES[reg];c0=s0*previous+base.INTERCEPTS[reg]
    initials=[np.sum(s0*s0,axis=1),np.sum(s0*(target-c0),axis=1),np.sum((c0-target)**2,axis=1)]
    events=(base.KNOTS[None,:,None]-previous[:,None,:]).reshape(restarts,-1)
    sb,sa=base.SLOPES[:-1][None,:,None],base.SLOPES[1:][None,:,None]
    cb=sb*previous[:,None,:]+base.INTERCEPTS[:-1][None,:,None]
    ca=sa*previous[:,None,:]+base.INTERCEPTS[1:][None,:,None]
    deltas=[np.broadcast_to(sa**2-sb**2,(restarts,3,n)).reshape(restarts,-1),
        (sa*(target[:,None,:]-ca)-sb*(target[:,None,:]-cb)).reshape(restarts,-1),
        ((ca-target[:,None,:])**2-(cb-target[:,None,:])**2).reshape(restarts,-1)]
    valid=(events>-base.BOUND)&(events<base.BOUND)
    positions=np.clip(events,-base.BOUND,base.BOUND)
    order=np.argsort(positions,axis=1,kind='stable')
    positions=np.take_along_axis(positions,order,axis=1)
    coefs=[]
    for ini,delta in zip(initials,deltas):
        ordered=np.take_along_axis(np.where(valid,delta,0.),order,axis=1)
        coefs.append(np.concatenate([ini[:,None],ini[:,None]+np.cumsum(ordered,axis=1)],axis=1)/n)
    aa,bb,cc=coefs;eta=1/(rho*n)
    lo=np.concatenate([np.full((restarts,1),-base.BOUND),positions],axis=1)
    hi=np.concatenate([positions,np.full((restarts,1),base.BOUND)],axis=1)
    raw=(bb+eta*anchor+trust*old[:,None])/(np.maximum(aa,0)+eta+trust)
    candidates=np.clip(raw,lo,hi)
    cost=aa*candidates**2-2*bb*candidates+cc
    cost+=eta*(candidates-anchor)**2+trust*(candidates-old[:,None])**2
    winner=np.argmin(cost,axis=1)
    rec.add(group,'initial_regions',reg)
    rec.add(group,'event_valid',valid)
    rec.add(group,'event_vs_box_low',cmp(events,-base.BOUND))
    rec.add(group,'event_vs_box_high',cmp(events,base.BOUND))
    rec.add(group,'stable_order',order)
    rec.add(group,'sorted_adjacent_equal',positions[:,1:]==positions[:,:-1])
    rec.add(group,'curvature_vs_zero',cmp(aa,0))
    rec.add(group,'candidate_vs_low',cmp(raw,lo))
    rec.add(group,'candidate_vs_high',cmp(raw,hi))
    rec.add(group,'winner',winner)
    rec.add(group,'winner_ties',cost==cost[np.arange(restarts),winner,None])
    return candidates[np.arange(restarts),winner]


def step(b,h,u,x,v,method='alm'):
    assert method in ['alm','nodual']
    b,h,u=np.array(b,copy=True),np.array(h,copy=True),np.array(u,copy=True)
    r,d=b.shape;trust=.01;rec=Recorder(r)
    for j in reversed(range(d)):
        group=f'activity_{j}';prev=x if j==0 else h[j-1];before=h[j].copy()
        z=prev+b[:,j,None];a=base.g(z)-u[j]
        rec.add(group,'read_regions',regions(z))
        if j==d-1:
            raw=(a+trust*before)/(1+trust);lo=np.maximum(0,v-base.EPS);hi=np.minimum(1,v+base.EPS)
            h[j]=np.clip(raw,lo,hi)
            rec.add(group,'candidate_vs_low',cmp(raw,lo));rec.add(group,'candidate_vs_high',cmp(raw,hi))
        else:
            nb=b[:,j+1];lraw=np.array([-np.inf,0,.5,1.])[:,None]-nb
            hraw=np.array([0.,.5,1.,np.inf])[:,None]-nb
            lo=np.maximum(0,lraw);hi=np.minimum(1,hraw)
            slope=base.SLOPES[:,None,None]
            offset=(base.SLOPES[:,None]*nb+base.INTERCEPTS[:,None])[:,:,None];target=h[j+1]+u[j+1]
            raw=(a[None]+slope*(target[None]-offset)+trust*before[None])/(1+slope**2+trust)
            cand=np.minimum(np.maximum(raw,lo[:,:,None]),hi[:,:,None])
            energy=(cand-a)**2+(base.g(cand+nb[None,:,None])-target)**2+trust*(cand-before)**2
            energy=np.where((lo>hi)[:,:,None],np.inf,energy)
            winner=np.argmin(energy,axis=0)
            h[j]=np.take_along_axis(cand,winner[None],axis=0)[0]
            rec.add(group,'low_source',cmp(lraw,0),axis=1);rec.add(group,'high_source',cmp(hraw,1),axis=1)
            rec.add(group,'interval_valid',lo<=hi,axis=1)
            rec.add(group,'candidate_vs_low',cmp(raw,lo[:,:,None]),axis=1)
            rec.add(group,'candidate_vs_high',cmp(raw,hi[:,:,None]),axis=1)
            rec.add(group,'candidate_eval_regions',regions(cand+nb[None,:,None]),axis=1)
            rec.add(group,'winner',winner)
            rec.add(group,'winner_ties',energy==np.take_along_axis(energy,winner[None],axis=0),axis=1)
    for j in range(d):
        prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1]
        b[:,j]=bias(prev,h[j]+u[j],b[:,j].copy(),0.,float('inf'),trust,rec,f'bias_{j}')
    prev=x;rate=.5 if method=='alm' else 0.
    for j in range(d):
        z=prev+b[:,j,None];rr=h[j]-base.g(z)
        rec.add(f'residual_{j}','regions',regions(z));u[j]+=rate*rr;prev=h[j]
    policies,schema=rec.finish()
    prev=np.broadcast_to(x,(r,len(x)));forward=[]
    for j in range(d):
        z=prev+b[:,j,None];forward.append(regions(z));prev=base.g(z)
    return dict(b=b,h=h,u=u,policies=policies,schema=schema,
                forward_policy=np.concatenate(forward,axis=1).astype(np.int16))


def segments(policies):
    """Inclusive, one-based step spans; rows are consecutive solver calls."""
    policies=np.asarray(policies);assert policies.ndim==2 and len(policies)>0
    starts=np.r_[0,1+np.flatnonzero(np.any(policies[1:]!=policies[:-1],axis=1))]
    ends=np.r_[starts[1:],len(policies)]
    return [dict(start=int(a+1),end=int(b),length=int(b-a)) for a,b in zip(starts,ends)]

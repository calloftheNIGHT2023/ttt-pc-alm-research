"""Same frozen local arithmetic, with explicit shared warm primal state.

Only initialization and recording differ from split_activity_mode_trace.refine.
Historical support-best is retained, even when it differs from current b.
"""
import numpy as np
import split_activity_mode_trace as original
import batched_bp_discovery as bp
base=original.base

def retain(best,b,x,v,anchor):
    errors,moves=base.score(best,x,v,anchor);err,mov=base.score(b,x,v,anchor)
    update=base.better(err,mov,errors,moves);best[update]=b[update]

def local(starts,activities,incumbent,x,v,method,sweeps):
    b=starts.copy();h=activities.copy();u=np.zeros_like(h);r,d=b.shape;trust=.01;anchor=np.zeros(d)
    best=incumbent.copy();retain(best,b,x,v,anchor)
    bh=[b.copy()];hh=[h.copy()];uh=[u.copy()];besth=[best.copy()]
    for step in range(1,sweeps+1):
        for j in reversed(range(d)):
            prev=x if j==0 else h[j-1];before=h[j].copy()
            a=base.g(prev+b[:,j,None])-(u[j] if method!='pc' else 0)
            if j==d-1:
                h[j]=np.clip((a+trust*before)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            elif method=='pc':
                z=before+b[:,j+1,None]
                grad=before-a+base.derivative(z)*(base.g(z)-h[j+1])
                h[j]=np.clip(before-grad/(5+trust),0,1)
            else:
                nb=b[:,j+1]
                lo=np.maximum(0,np.array([-np.inf,0,.5,1.])[:,None]-nb)
                hi=np.minimum(1,np.array([0.,.5,1.,np.inf])[:,None]-nb)
                slope=base.SLOPES[:,None,None]
                offset=(base.SLOPES[:,None]*nb+base.INTERCEPTS[:,None])[:,:,None]
                target=h[j+1]+u[j+1]
                cand=(a[None]+slope*(target[None]-offset)+trust*before[None])/(1+slope**2+trust)
                cand=np.minimum(np.maximum(cand,lo[:,:,None]),hi[:,:,None])
                energy=(cand-a)**2+(base.g(cand+nb[None,:,None])-target)**2+trust*(cand-before)**2
                energy=np.where((lo>hi)[:,:,None],np.inf,energy)
                h[j]=np.take_along_axis(cand,np.argmin(energy,axis=0)[None],axis=0)[0]
        for j in range(d):
            prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1]
            if method=='pc':
                before=b[:,j].copy();z=prev+before[:,None]
                grad=np.mean((base.g(z)-h[j])*base.derivative(z),axis=1)
                b[:,j]=np.clip(before-grad/(4+trust),-base.BOUND,base.BOUND)
            else:b[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),anchor[j],float('inf'),trust)
        if method!='pc':
            prev=x
            for j in range(d):
                residual=h[j]-base.g(prev+b[:,j,None]);u[j]+=(0. if method=='nodual' else .5)*residual;prev=h[j]
        retain(best,b,x,v,anchor);bh.append(b.copy());hh.append(h.copy());uh.append(u.copy());besth.append(best.copy())
    return dict(b=np.array(bh),h=np.array(hh),u=np.array(uh),best=np.array(besth))

def baseline(starts,incumbent,x,v,solver,steps):
    bank=[];roles=[];saved=bp.evaluate
    def record(b,x,v,with_jacobian=True):
        bank.append(b.copy());roles.append(with_jacobian);return saved(b,x,v,with_jacobian)
    try:
        bp.evaluate=record;best,meta=bp.refine(starts,x,v,np.zeros(starts.shape[1]),solver=solver,steps=steps)
    finally:bp.evaluate=saved
    expected,_=bp.refine(starts,x,v,np.zeros(starts.shape[1]),solver=solver,steps=steps)
    assert best.tobytes()==expected.tobytes()
    retained=incumbent.copy();retain(retained,best,x,v,np.zeros(starts.shape[1]))
    return dict(b=np.array(bank),roles=np.array(roles),best=retained[None]),meta

def verify():
    rng=np.random.default_rng(245731);cases=0
    with original.old.core.pipeline.discovery_box(.12):
        for n in [4,8,24]:
            x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);starts=rng.uniform(-.12,.12,(8,4));h=[];prev=x
            for j in range(4):h.append(base.g(prev+starts[:,j,None]));prev=h[-1]
            for method in ['alm','nodual','pc']:
                actual=local(starts,np.array(h),starts,x,v,method,7)['best'][-1]
                expected=original.original(starts,x,v,np.zeros(4),method,7)
                assert actual.tobytes()==expected.tobytes();cases+=1
    return dict(passed=True,forward_start_original_bitwise_cases=cases)

"""Read-only diagnostic of off-forward activity modes in unchanged local paths.

No complete reference, teacher, query or query answer enters this module.
The copied update arithmetic is checked bitwise against the frozen original.
Timing here includes tracing overhead and is NOT an online speed benchmark.
"""
import numpy as np
import stateful_posterior_memory as old
base=old.base


class Observer:
    def __init__(self,x,v):
        self.x=x;self.v=v;self.forward={};self.split={};self.events=0

    def parameter(self,b,step):
        for reg in old.interface.archived.signatures(self.x,b):
            self.forward.setdefault(reg.tobytes().hex(),int(step))

    def activity(self,b,h,u,step,phase):
        d,r,n=h.shape;codes=[];residual=[];prev=np.broadcast_to(self.x,(r,n))
        for j in range(d):
            z=prev+b[:,j,None]
            codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8))
            residual.append(h[j]-base.g(z));prev=h[j]
        codes=np.array(codes).transpose(1,0,2);residual=np.array(residual)
        rms=np.sqrt(np.mean(residual**2,axis=(0,2)))
        maximum=np.max(np.abs(residual),axis=(0,2))
        dual=np.sqrt(np.mean(u**2,axis=(0,2)))
        errors,_=base.score(b,self.x,self.v,np.zeros(d))
        for i,reg in enumerate(codes):
            key=reg.tobytes().hex()
            if key not in self.split:
                self.split[key]=dict(first_step=int(step),phase=phase,restart=int(i),
                    first_residual_rms=float(rms[i]),minimum_residual_rms=float(rms[i]),
                    first_residual_max=float(maximum[i]),first_dual_rms=float(dual[i]),
                    first_forward_max_error=float(errors[i]),appearances=0)
            entry=self.split[key];entry['appearances']+=1
            entry['minimum_residual_rms']=min(entry['minimum_residual_rms'],float(rms[i]))
        self.events+=1


def refine(starts,x,v,anchor,method,sweeps,observer):
    b=starts.copy();r,d=b.shape;trust=.01
    h=np.empty((d,r,len(x)));u=np.zeros_like(h);prev=x
    for j in range(d):h[j]=base.g(prev+b[:,j,None]);prev=h[j]
    best=b.copy();errors,moves=base.score(best,x,v,anchor)
    observer.parameter(b,0);observer.activity(b,h,u,0,'initial')
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
        observer.activity(b,h,u,step,'after_activities')
        for j in range(d):
            prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1]
            if method=='pc':
                before=b[:,j].copy();z=prev+before[:,None]
                grad=np.mean((base.g(z)-h[j])*base.derivative(z),axis=1)
                b[:,j]=np.clip(before-grad/(4+trust),-base.BOUND,base.BOUND)
            else:
                b[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),anchor[j],float('inf'),trust)
        observer.activity(b,h,u,step,'after_parameters_before_dual')
        if method!='pc':
            prev=x
            for j in range(d):
                residual=h[j]-base.g(prev+b[:,j,None])
                u[j]+=(0. if method=='nodual' else .5)*residual;prev=h[j]
        err,mov=base.score(b,x,v,anchor);update=base.better(err,mov,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=mov[update]
        observer.parameter(b,step)
    return best


def original(starts,x,v,anchor,method,sweeps):
    if method=='pc':return old.refine_pc(starts,x,v,anchor,sweeps)[0]
    return old.core.contextual.refine_local(starts,x,v,anchor,sweeps=sweeps,dual_rate=0. if method=='nodual' else .5)[0]


def verify():
    rng=np.random.default_rng(182817);cases=0
    with old.core.pipeline.discovery_box(.12):
        for n in [4,8,24]:
            x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);starts=rng.uniform(-.12,.12,(8,4))
            for method in ['alm','nodual','pc']:
                observer=Observer(x,v);actual=refine(starts,x,v,np.zeros(4),method,7,observer)
                expected=original(starts,x,v,np.zeros(4),method,7)
                assert np.array_equal(actual,expected),(n,method)
                initial={k for k,r in observer.split.items() if r['first_step']==0}
                original_keys={r.tobytes().hex() for r in old.interface.archived.signatures(x,starts)}
                assert initial==original_keys
                assert all(observer.split[k]['first_residual_max']==0 for k in initial)
                assert len(observer.forward)>0;cases+=1
    return dict(passed=True,bitwise_original_solver_cases=cases,zero_residual_matches_forward=True,
                independent_queries_or_teacher_absent=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

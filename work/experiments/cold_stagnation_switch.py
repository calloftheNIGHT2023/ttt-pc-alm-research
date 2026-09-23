"""Round247: support-only one-way multiplier switch; fixed128 total sweeps."""
import time
import numpy as np
import multiplier_warm_continuation as frozen
import batched_bp_discovery as bp
base=frozen.base

PRIMARY='stagnation_switch128'

def configs():
    ans=[dict(name=PRIMARY,family='local',method='switch',steps=128)]
    ans.extend(dict(name=f'{name}{steps}',family='local',method=method,steps=steps) for name,method,steps in [
        ('alm','alm',128),('nodual','nodual',128),('pc','pc',128),('alm','alm',16),('pc','pc',80)])
    ans.extend(dict(name=name,family='bp',method=method,steps=steps) for name,method,steps in [
        ('adam240','adam',240),('gn40','gauss_newton',40),('adam60','adam',60),('gn20','gauss_newton',20)])
    ans.extend(dict(name=n,family='regression') for n in ['linear_ls','residual_linear_ls','residual_linear_ridge','residual_linear_rls','prior4096_ridge','rbf_loocv'])
    ans.extend(dict(name=n,family='meta') for n in ['meta_ridge64','meta_ridge128','meta_shallow64_5','meta_shallow64_20'])
    assert len(ans)==len({a['name'] for a in ans})==20
    return ans

def starts():return np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(16,4))]

def trigger(previous_change,current_change,residual_max,error,active):
    return (~active)&(previous_change<=1e-12)&(current_change<=1e-12)&(residual_max>1e-6)&(error>.001001)

class Local:
    def __init__(self,b,x,v,method):
        self.x=x;self.v=v;self.b=b.copy();r,d=b.shape;self.anchor=np.zeros(d);self.method=method;self.step_count=0
        self.h=np.empty((d,r,len(x)));prev=x
        for j in range(d):self.h[j]=base.g(prev+self.b[:,j,None]);prev=self.h[j]
        self.u=np.zeros_like(self.h);self.best=b.copy();self.errors,self.moves=base.score(self.best,x,v,self.anchor)
        self.active=np.full(r,method=='alm');self.first_trigger=np.full(r,-1,dtype=int);self.last_change=np.full(r,np.inf)
        self.detection_seconds=0.;self.detect_calls=0

    def step(self):
        b,h,u=self.b,self.h,self.u;x,v=self.x,self.v;r,d=b.shape;trust=.01;method=self.method
        oldb=b.copy() if method=='switch' else None;oldh=h.copy() if method=='switch' else None
        for j in reversed(range(d)):
            prev=x if j==0 else h[j-1];before=h[j].copy();a=base.g(prev+b[:,j,None])-(u[j] if method!='pc' else 0)
            if j==d-1:h[j]=np.clip((a+trust*before)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            elif method=='pc':
                z=before+b[:,j+1,None];grad=before-a+base.derivative(z)*(base.g(z)-h[j+1]);h[j]=np.clip(before-grad/(5+trust),0,1)
            else:
                nb=b[:,j+1];lo=np.maximum(0,np.array([-np.inf,0,.5,1.])[:,None]-nb);hi=np.minimum(1,np.array([0.,.5,1.,np.inf])[:,None]-nb)
                slope=base.SLOPES[:,None,None];offset=(base.SLOPES[:,None]*nb+base.INTERCEPTS[:,None])[:,:,None];target=h[j+1]+u[j+1]
                cand=(a[None]+slope*(target[None]-offset)+trust*before[None])/(1+slope**2+trust)
                cand=np.minimum(np.maximum(cand,lo[:,:,None]),hi[:,:,None]);energy=(cand-a)**2+(base.g(cand+nb[None,:,None])-target)**2+trust*(cand-before)**2
                energy=np.where((lo>hi)[:,:,None],np.inf,energy);h[j]=np.take_along_axis(cand,np.argmin(energy,axis=0)[None],axis=0)[0]
        for j in range(d):
            prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1]
            if method=='pc':
                before=b[:,j].copy();z=prev+before[:,None];grad=np.mean((base.g(z)-h[j])*base.derivative(z),axis=1);b[:,j]=np.clip(before-grad/(4+trust),-base.BOUND,base.BOUND)
            else:b[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),self.anchor[j],float('inf'),trust)
        residual=[];prev=x
        if method!='pc':
            rate=np.where(self.active,.5,0.)
            for j in range(d):
                rr=h[j]-base.g(prev+b[:,j,None]);residual.append(rr);u[j]+=rate[:,None]*rr;prev=h[j]
        err,mov=base.score(b,x,v,self.anchor);update=base.better(err,mov,self.errors,self.moves)
        self.best[update]=b[update];self.errors[update]=err[update];self.moves[update]=mov[update];self.step_count+=1
        if method=='switch':
            start=time.perf_counter();change=np.maximum(np.max(abs(b-oldb),axis=1),np.max(abs(h-oldh),axis=(0,2)))
            maximum=np.max(abs(np.array(residual)),axis=(0,2));activate=trigger(self.last_change,change,maximum,self.errors,self.active)
            assert not np.any(self.u[:,activate])
            self.active|=activate;self.first_trigger[activate]=self.step_count;self.last_change=change
            self.detection_seconds+=time.perf_counter()-start;self.detect_calls+=1

    def arrays(self):
        return dict(b=self.b.copy(),h=self.h.copy(),u=self.u.copy(),best=self.best.copy(),active=self.active.copy(),first_trigger=self.first_trigger.copy())

    def numeric_state_bytes(self):return sum(a.nbytes for a in [self.b,self.h,self.u,self.best,self.errors,self.moves,self.active,self.first_trigger,self.last_change,self.anchor])

def run_local(start,x,v,method,steps,trace=True):
    state=Local(start,x,v,method);history={k:[v] for k,v in state.arrays().items()} if trace else None
    for _ in range(steps):
        state.step()
        if trace:
            for key,value in state.arrays().items():history[key].append(value)
    arrays={k:np.array(v) for k,v in history.items()} if trace else None
    meta=dict(active_restarts=int(np.sum((state.first_trigger>=0)&(state.first_trigger<steps))),
        first_trigger=state.first_trigger.tolist(),detection_seconds=state.detection_seconds,detect_calls=state.detect_calls,
        solver_numeric_state_bytes=state.numeric_state_bytes(),numeric_state_scope='live named arrays, excludes workspaces and diagnostics')
    return state.best.copy(),arrays,meta

def run_bp(start,x,v,method,steps,trace=True):
    bank=[];roles=[];saved=bp.evaluate
    def record(b,x,v,with_jacobian=True):
        bank.append(b.copy());roles.append(with_jacobian);return saved(b,x,v,with_jacobian)
    try:
        if trace:bp.evaluate=record
        best,meta=bp.refine(start,x,v,np.zeros(4),solver=method,steps=steps)
    finally:bp.evaluate=saved
    return best,(dict(b=np.array(bank),roles=np.array(roles),best=best[None]) if trace else None),meta

def select(best,x,v):
    err,mov=base.score(best,x,v,np.zeros(4));index=0
    for i in range(1,len(best)):
        if bool(base.better(err[i],mov[i],err[index],mov[index])):index=i
    return index,best[index].copy()

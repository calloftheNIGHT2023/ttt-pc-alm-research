"""Hard noisy-write constraints for a coupled vector memory.

New constrained split-ALM prototype; not official PC-ALM or official TTT.
No posterior polytopes: this pilot isolates internal parameter discovery.
"""
import numpy as np
from scipy.optimize import minimize
import coupled_branch_memory as family

EPS=.001
TOL=1e-6
PRIOR=.2
BOUND=.3


def forward(bank,x,weights):
    h=np.broadcast_to(x,(len(bank),*x.shape))
    for j,w in enumerate(weights): h=family.activation(h@w.T+bank[:,j,None,:])
    return h


def residuals(raw):return np.sign(raw)*np.maximum(np.abs(raw)-EPS,0)


def evaluate(bank,x,v,weights,jacobian=False):
    r,d,width=bank.shape; n=len(x); h=np.broadcast_to(x,(r,*x.shape)); ds=[]
    jac=np.zeros((r,n,width,d*width)) if jacobian else None
    for j,w in enumerate(weights):
        z=h@w.T+bank[:,j,None,:]; deriv=family.deriv(z); ds.append(deriv)
        if jacobian:
            jac=np.einsum("ab,rnbp->rnap",w,jac,optimize=True)
            jac[:,:,np.arange(width),j*width+np.arange(width)]+=1.
            jac*=deriv[:,:,:,None]
        h=family.activation(z)
    raw=h-v; error=residuals(raw); value=.5*np.mean(np.sum(error**2,axis=2),axis=1)
    if jacobian:
        jac*=np.abs(raw[:,:,:,None])>EPS
        return value,error,jac,raw
    adj=error/n; grad=np.zeros_like(bank)
    for j in reversed(range(d)):
        adj*=ds[j]; grad[:,j]=adj.sum(1); adj=adj@weights[j]
    return value,grad,raw


def score(bank,x,v,weights,anchor):
    err=np.max(np.abs(forward(bank,x,weights)-v),axis=(1,2)); move=.5*np.sum((bank-anchor)**2,axis=(1,2))
    return err,move


def better(err,move,old_err,old_move):
    feasible,old=err<=EPS+TOL,old_err<=EPS+TOL
    return (feasible&~old)|(feasible&old&(move<old_move))|(~feasible&~old&(err<old_err))


def proposals(x,v,anchor,weights,features=256,restarts=16):
    bank=np.random.default_rng(731).uniform(-PRIOR,PRIOR,(features,*anchor.shape))
    h=np.broadcast_to(x,(features,*x.shape)); signatures=[]
    for j,w in enumerate(weights):
        z=h@w.T+bank[:,j,None,:]; reg=np.searchsorted([-1.,0.,1.],z).astype(np.uint8)
        bits=np.stack([reg&1,reg>>1],axis=-1).reshape(features,-1)
        signatures.append(np.packbits(bits,axis=1,bitorder="little")); h=family.activation(z)
    raw=np.mean((h-v)**2,axis=(1,2)); signature=np.ascontiguousarray(np.concatenate(signatures,axis=1))
    tokens=signature.view(np.dtype((np.void,signature.shape[1]))).reshape(-1); order=np.argsort(raw,kind="stable")
    _,first=np.unique(tokens[order],return_index=True); selected=order[np.sort(first)][:restarts-1]
    return np.concatenate([anchor[None],bank[selected]]),{"prior_tasks":features,"proposal_task_parameter_bytes":bank.nbytes,
        "proposal_support_feature_bytes":h.nbytes,"proposal_signature_bytes":signature.nbytes}


def local(starts,x,v,weights,anchor,*,sweeps=240,dual_rate=.5,gradient_nonlinearity=False):
    b=starts.copy(); r,d,width=b.shape; n=len(x); trust=.01
    z=np.empty((d,r,n,width)); h=np.empty_like(z); p=np.zeros_like(z); a=np.zeros_like(z); previous=x
    for j,w in enumerate(weights): z[j]=previous@w.T+b[:,j,None,:]; h[j]=family.activation(z[j]); previous=h[j]
    best=b.copy(); errors,moves=score(best,x,v,weights,anchor)
    for _ in range(sweeps):
        for j in reversed(range(d)):
            target=family.activation(z[j])-a[j]
            if j==d-1:
                h[j]=np.clip((target+trust*h[j])/(1+trust),np.maximum(-1,v-EPS),np.minimum(1,v+EPS))
            else:
                t=z[j+1]-b[:,j+1,None,:]+p[j+1]
                h[j]=np.clip((target+t@weights[j+1]+trust*h[j])/(2+trust),-1,1)
            previous=x if j==0 else h[j-1]; center=previous@weights[j].T+b[:,j,None,:]-p[j]
            target=h[j]+a[j]
            if gradient_nonlinearity:
                dz=z[j]-center+family.deriv(z[j])*(family.activation(z[j])-target)
                z[j]=z[j]-dz/(5+trust)
            else:z[j]=family.nonlinear_prox(center,target,z[j],trust)
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            b[:,j]=np.clip(np.mean(z[j]-previous@w.T+p[j],axis=1),-BOUND,BOUND)
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            p[j]+=dual_rate*(z[j]-previous@w.T-b[:,j,None,:]); a[j]+=dual_rate*(h[j]-family.activation(z[j]))
        err,move=score(b,x,v,weights,anchor); update=better(err,move,errors,moves)
        best[update]=b[update]; errors[update]=err[update]; moves[update]=move[update]
    return best,{"sweeps":sweeps,"dual_rate":dual_rate,"gradient_nonlinearity":gradient_nonlinearity,
        "major_arrays_bytes_subtotal":sum(t.nbytes for t in [b,z,h,p,a,best]),
        "final_affine_residual_rms":float(np.sqrt(np.mean(np.stack([z[j]-(x if j==0 else h[j-1])@weights[j].T-b[:,j,None,:] for j in range(d)])**2))),
        "final_nonlinear_residual_rms":float(np.sqrt(np.mean((h-family.activation(z))**2)))}


def bp(starts,x,v,weights,anchor,*,solver="adam",steps=240,lr=.003):
    b=starts.copy(); best=b.copy(); r,d,width=b.shape; n=len(x); pcount=d*width
    errors,moves=score(best,x,v,weights,anchor); first=np.zeros_like(b); second=first.copy(); damping=np.full(r,.01)
    def retain(params,raw):
        err=np.max(np.abs(raw),axis=(1,2)); move=.5*np.sum((params-anchor)**2,axis=(1,2)); update=better(err,move,errors,moves)
        best[update]=params[update]; errors[update]=err[update]; moves[update]=move[update]
    for it in range(1,steps+1):
        if solver=="adam":
            _,grad,raw=evaluate(b,x,v,weights); retain(b,raw)
            first=.9*first+.1*grad; second=.999*second+.001*grad**2
            b=np.clip(b-lr*(first/(1-.9**it))/(np.sqrt(second/(1-.999**it))+1e-8),-BOUND,BOUND)
        else:
            value,error,jac,raw=evaluate(b,x,v,weights,True); retain(b,raw)
            jj=jac.reshape(r,n*width,pcount); ee=error.reshape(r,n*width)
            grad=np.einsum("rni,rn->ri",jj,ee,optimize=True)/n
            hh=np.einsum("rni,rnj->rij",jj,jj,optimize=True)/n
            diag=np.maximum(np.diagonal(hh,axis1=1,axis2=2),1e-4)
            delta=np.linalg.solve(hh+damping[:,None,None]*np.eye(pcount)[None]*diag[:,None,:],-grad[:,:,None])[:,:,0]
            delta*=np.minimum(1,.1/np.maximum(np.max(np.abs(delta),axis=1),1e-30))[:,None]; delta=delta.reshape(b.shape)
            nextb=b.copy(); nextvalue=value.copy(); changed=np.zeros(r,dtype=bool)
            for alpha in [1.,.5,.25,.125,.0625]:
                trial=np.clip(b+alpha*delta,-BOUND,BOUND); raw=forward(trial,x,weights)-v; retain(trial,raw)
                val=.5*np.mean(np.sum(residuals(raw)**2,axis=2),axis=1); update=val<nextvalue
                nextb[update]=trial[update]; nextvalue[update]=val[update]; changed|=update
            b=nextb; damping=np.clip(damping*np.where(changed,.3,10.),1e-8,1e8)
    retain(b,forward(b,x,weights)-v)
    return best,{"batched_bp_solver":solver,"steps":steps,"learning_rate":lr if solver=="adam" else None,
        "major_arrays_bytes_subtotal":sum(t.nbytes for t in [b,best,first,second])}


def lbfgs(starts,x,v,weights,anchor,steps=300):
    best=starts.copy(); errors,moves=score(best,x,v,weights,anchor); calls=0
    for i,start in enumerate(starts):
        def fun(flat):
            nonlocal calls
            b=flat.reshape(start.shape); value,grad,raw=evaluate(b[None],x,v,weights); calls+=1
            err=np.max(np.abs(raw)); move=.5*np.sum((b-anchor)**2)
            if better(err,move,errors[i],moves[i]):best[i]=b; errors[i]=err; moves[i]=move
            return value[0],grad.ravel()
        minimize(fun,start.ravel(),jac=True,method="L-BFGS-B",bounds=[(-BOUND,BOUND)]*start.size,
            options={"maxiter":steps,"ftol":1e-13,"gtol":1e-9,"maxls":40})
    return best,{"solver":"lbfgs","global_gradient_calls":calls}


def select(bank,x,v,weights,anchor):
    err,move=score(bank,x,v,weights,anchor); feasible=err<=EPS+TOL
    index=int(np.argmin(np.where(feasible,move,np.inf))) if feasible.any() else int(np.argmin(err))
    return bank[index].copy()


def fit_internal(x,v,anchor,weights,cfg):
    starts,meta=proposals(x,v,anchor,weights,features=cfg.get("features",256),restarts=cfg.get("restarts",16))
    if cfg["method"]=="local":
        bank,more=local(starts,x,v,weights,anchor,sweeps=cfg["sweeps"],dual_rate=cfg.get("dual_rate",.5),gradient_nonlinearity=cfg.get("gradient",False))
    elif cfg["method"]=="lbfgs":bank,more=lbfgs(starts,x,v,weights,anchor,cfg["steps"])
    else:bank,more=bp(starts,x,v,weights,anchor,solver=cfg["method"],steps=cfg["steps"],lr=cfg.get("lr",.003))
    point=select(np.concatenate([starts,bank]),x,v,weights,anchor)
    def predict(q):return forward(point[None],q,weights)[0]
    return predict,point,{**meta,**more,"persistent_state_bytes":point.nbytes,"known_weights_bytes":weights.nbytes,
        "anchor_output":point.tolist(),"readout":"one support-selected parameter; no posterior geometric refinement"}


def prior_moments(x,v,weights,features=1024):
    d,width=weights.shape[:2]; bank=np.random.default_rng(731).uniform(-PRIOR,PRIOR,(features,d,width))
    phi=forward(bank,x,weights).reshape(features,-1); mean=phi.mean(0); centered=phi-mean
    covariance=centered.T@centered/features+np.eye(v.size)*EPS**2/3
    alpha=np.linalg.solve(covariance,v.ravel()-mean); mixing=(1+centered@alpha)/features
    def predict(q):
        answer=np.empty((len(q),width))
        for first in range(0,len(q),64):answer[first:first+64]=np.einsum("r,rnw->nw",mixing,forward(bank,q[first:first+64],weights),optimize=False)
        return answer
    return predict,{"prior_tasks":features,"persistent_state_bytes":bank.nbytes+mixing.nbytes,"known_weights_bytes":weights.nbytes,
        "support_covariance_bytes":covariance.nbytes,"noise_variance":EPS**2/3}


def tangent_features(x,weights):
    d,width=weights.shape[:2]; h=x.copy(); jac=np.zeros((len(x),width,d*width))
    for j,w in enumerate(weights):
        z=h@w.T; jac=np.einsum("ab,nbp->nap",w,jac,optimize=True)
        jac[:,np.arange(width),j*width+np.arange(width)]+=1
        jac*=family.deriv(z)[:,:,None]; h=family.activation(z)
    return h,jac


def fit_closed(x,v,weights,cfg):
    if cfg["method"]=="prior":return prior_moments(x,v,weights,cfg["features"])
    d,width=weights.shape[:2]; zero=np.zeros((1,d,width)); initial=forward(zero,x,weights)[0]; target=v-initial
    if cfg["method"]=="tangent":
        _,jac=tangent_features(x,weights); design=jac.reshape(v.size,-1)
        regularizer=(EPS**2/3)/(PRIOR**2/3)
        coef=np.linalg.solve(design.T@design+regularizer*np.eye(d*width),design.T@target.ravel())
        def predict(q):
            mean,features=tangent_features(q,weights)
            return mean+np.einsum("nwp,p->nw",features,coef,optimize=False)
        return predict,{"persistent_state_bytes":coef.nbytes,"known_weights_bytes":weights.nbytes,"ridge_parameter":regularizer}
    if cfg["method"]=="linear":
        coef=np.linalg.lstsq(np.c_[x,np.ones(len(x))],target,rcond=None)[0]
        return lambda q:forward(zero,q,weights)[0]+np.c_[q,np.ones(len(q))]@coef,{"persistent_state_bytes":coef.nbytes,"known_weights_bytes":weights.nbytes}
    # Frozen RBF residual head; bandwidth and ridge selected solely by context LOO.
    dist=np.maximum(0,np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None]-2*x@x.T)/width
    best=None
    for scale in [.1,.3,1.,3.]:
        k=np.exp(-dist/(2*scale**2))
        for reg in [1e-6,1e-4,1e-2,1.]:
            inverse=np.linalg.inv(k+reg*np.eye(len(x))); coef=inverse@target
            loo=float(np.mean((coef/np.diag(inverse)[:,None])**2))
            if best is None or loo<best[0]:best=loo,scale,reg,coef
    _,scale,reg,coef=best
    def predict(q):
        distance=np.maximum(0,np.sum(q*q,axis=1)[:,None]+np.sum(x*x,axis=1)[None]-2*q@x.T)/width
        return forward(zero,q,weights)[0]+np.exp(-distance/(2*scale**2))@coef
    return predict,{"persistent_state_bytes":x.nbytes+coef.nbytes,"known_weights_bytes":weights.nbytes,"loo_bandwidth":scale,"loo_ridge":reg}


def verify():
    rng=np.random.default_rng(118654); weights=family.make_weights(3,4); x=rng.uniform(-1,1,(7,4)); v=rng.uniform(-1,1,(7,4)); bank=rng.uniform(-.2,.2,(3,3,4))
    value,grad,raw=evaluate(bank,x,v,weights); value2,err,jac,raw2=evaluate(bank,x,v,weights,True)
    jgrad=np.einsum("rnwp,rnw->rp",jac,err,optimize=True).reshape(bank.shape)/len(x)
    assert np.allclose(grad,jgrad,atol=1e-11) and np.array_equal(raw,raw2)
    finite=[]
    for i in range(3):
        numeric=[]
        for j in range(12):
            delta=np.eye(12)[j].reshape(3,4)*1e-7
            plus=evaluate((bank[i]+delta)[None],x,v,weights)[0][0]; minus=evaluate((bank[i]-delta)[None],x,v,weights)[0][0]
            numeric.append((plus-minus)/2e-7)
        finite.append(float(np.max(np.abs(np.array(numeric)-grad[i].ravel()))))
    assert max(finite)<1e-6
    # Local exact h block audit: orthogonal W, box projection and proximal term.
    a=rng.normal(size=(4,)); t=rng.normal(size=(4,)); old=rng.normal(size=(4,)); w=weights[1]; trust=.01
    solution=np.clip((a+t@w+trust*old)/(2+trust),-1,1)
    fn=lambda h:.5*np.sum((h-a)**2)+.5*np.sum((h@w.T-t)**2)+trust*.5*np.sum((h-old)**2)
    check=minimize(fn,np.zeros(4),bounds=[(-1,1)]*4,method="L-BFGS-B",options={"ftol":1e-14,"gtol":1e-9})
    assert fn(solution)<=check.fun+1e-10
    for solver in ["adam","gauss_newton"]:
        together,_=bp(bank,x,v,weights,np.zeros((3,4)),solver=solver,steps=3)
        apart=np.concatenate([bp(b[None],x,v,weights,np.zeros((3,4)),solver=solver,steps=3)[0] for b in bank])
        assert np.allclose(together,apart,atol=1e-10)
    # Candidate executes with whole-network BP interfaces disabled.
    original=evaluate
    try:
        def forbidden(*args,**kwargs):raise RuntimeError("global BP forbidden in candidate")
        globals()["evaluate"]=forbidden
        out,_=local(bank,x,v,weights,np.zeros((3,4)),sweeps=3)
        assert np.isfinite(out).all()
    finally:globals()["evaluate"]=original
    return {"passed":True,"max_finite_difference_gradient_error":max(finite),"reverse_and_forward_chain_gradient_agree":True,
        "orthogonal_box_activity_block_exact":True,"batched_bp_independent_restarts":True,"candidate_global_bp_guard":True,
        "existing_piecewise_prox":family.verify()}


if __name__=="__main__":
    import json
    print(json.dumps(verify()))

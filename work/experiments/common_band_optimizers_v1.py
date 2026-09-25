"""409 same band objective and support-only incumbent rule across six solvers.

Local variants use only adjacent activities/derivatives and start with u=0.
Whole-chain derivatives are confined to explicitly named BP baselines.
No posterior geometry or claim of global/finite-budget superiority is made.
"""
from contextlib import contextmanager
import time
import numpy as np
import streaming_branch_projection as base

FAMILIES=('bp_adam','bp_gn','pc_grad','alm_grad','pc_block','alm_block')


def bands(v,epsilon):
    low,high=np.maximum(0.,v-epsilon),np.minimum(1.,v+epsilon)
    assert epsilon>=0 and np.all(low<=high)
    return low,high


def evaluate(bank,x,v,*,epsilon=.001,tau=1.,jacobian=False):
    assert tau>0
    r,depth=bank.shape;n=len(x)
    h=np.broadcast_to(x,(r,n));jac=np.zeros((r,n,depth)) if jacobian else None
    for layer in range(depth):
        z=h+bank[:,layer,None]
        if jacobian:
            d=base.derivative(z);jac*=d[:,:,None];jac[:,:,layer]+=d
        h=base.g(z)
    low,high=bands(v,epsilon);residual=h-np.clip(h,low,high)
    if jacobian:jac*=((h<low)|(h>high))[:,:,None]
    value=.5*np.mean(residual**2,axis=1)/tau
    return value,residual,jac,h


def forward_trace(bank,x):
    h=[];previous=np.broadcast_to(x,(len(bank),len(x)))
    for layer in range(bank.shape[1]):
        previous=base.g(previous+bank[:,layer,None]);h.append(previous)
    return np.array(h)


def output_prox(a,previous,low,high,rho,trust,tau):
    kappa=rho*(1+trust);center=(a+trust*previous)/(1+trust)
    return np.clip((kappa*tau*center+np.clip(center,low,high))/(1+kappa*tau),0.,1.)


def activity_block(a,target,next_bias,previous,trust):
    lo=np.maximum(0.,np.array([-np.inf,0.,.5,1.])[:,None]-next_bias)
    hi=np.minimum(1.,np.array([0.,.5,1.,np.inf])[:,None]-next_bias)
    slope=base.SLOPES[:,None,None]
    offset=(base.SLOPES[:,None]*next_bias+base.INTERCEPTS[:,None])[:,:,None]
    proposal=(a[None]+slope*(target[None]-offset)+trust*previous[None])/(1+slope**2+trust)
    proposal=np.minimum(np.maximum(proposal,lo[:,:,None]),hi[:,:,None])
    energy=(proposal-a)**2+(base.g(proposal+next_bias[None,:,None])-target)**2+trust*(proposal-previous)**2
    energy=np.where((lo>hi)[:,:,None],np.inf,energy)
    return np.take_along_axis(proposal,np.argmin(energy,axis=0)[None],axis=0)[0]


@contextmanager
def scoped_bias_box(bound):
    """Serial process-local call; frozen solver source is never modified."""
    scope=base.bias_solve.__globals__;before=scope['BOUND'];scope['BOUND']=bound
    try:yield
    finally:scope['BOUND']=before


def bias_block(previous,target,before,trust,bound):
    with scoped_bias_box(bound):
        # rho=inf removes the old solver's anchor regularizer. The actual AL
        # rho cancels from this block's residual and proximal stabilization.
        return base.bias_solve(previous,target,before,0.,float('inf'),trust)


def local_partials(bank,h,u,x,v,*,epsilon=.001,tau=1.,rho=1.):
    depth,restarts,n=h.shape;previous=np.broadcast_to(x,(restarts,n))
    credit=[];derivative=[]
    for j in range(depth):
        z=previous+bank[:,j,None]
        credit.append(u[j]+h[j]-base.g(z));derivative.append(base.derivative(z));previous=h[j]
    credit,derivative=np.array(credit),np.array(derivative)
    gb=(-rho*np.mean(credit*derivative,axis=2)).T
    gh=rho*credit/n
    gh[:-1]-=rho*derivative[1:]*credit[1:]/n
    low,high=bands(v,epsilon);gh[-1]+=(h[-1]-np.clip(h[-1],low,high))/(n*tau)
    return gb,gh


def augmented(bank,h,u,x,v,*,epsilon=.001,tau=1.,rho=1.):
    low,high=bands(v,epsilon);answer=.5*np.mean((h[-1]-np.clip(h[-1],low,high))**2,axis=1)/tau
    previous=np.broadcast_to(x,(len(bank),len(x)))
    for j in range(len(h)):
        residual=h[j]-base.g(previous+bank[:,j,None])
        answer+=rho*np.mean(u[j]*residual+.5*residual**2,axis=1);previous=h[j]
    return answer


class Local:
    def __init__(self,start,x,v,family,*,epsilon=.001,tau=1.,rho=1.,trust=.01,dual_rate=.5,bound=.12):
        assert family in FAMILIES[2:] and tau>0 and rho>0 and trust>0 and dual_rate>=0 and bound>0
        self.b=np.array(start,dtype=float,copy=True);self.x=x;self.v=v;self.family=family
        self.epsilon=epsilon;self.tau=tau;self.rho=rho;self.trust=trust;self.dual_rate=dual_rate;self.bound=bound
        self.h=forward_trace(self.b,x);self.u=np.zeros_like(self.h)
        self.low,self.high=bands(v,epsilon);self.sweeps=0

    def step(self):
        b,h,u=self.b,self.h,self.u;r,depth=b.shape;trust=self.trust
        gradient=self.family.endswith('_grad')
        for j in reversed(range(depth)):
            previous=self.x if j==0 else h[j-1]
            before=h[j].copy();a=base.g(previous+b[:,j,None])-u[j]
            if j==depth-1:
                h[j]=output_prox(a,before,self.low,self.high,self.rho,trust,self.tau)
            elif gradient:
                z=before+b[:,j+1,None]
                derivative=before-a+base.derivative(z)*(base.g(z)-h[j+1]-u[j+1])
                h[j]=np.clip(before-derivative/(5+trust),0.,1.)
            else:
                h[j]=activity_block(a,h[j+1]+u[j+1],b[:,j+1],before,trust)
        for j in range(depth):
            previous=np.broadcast_to(self.x,(r,len(self.x))) if j==0 else h[j-1]
            before=b[:,j].copy();target=h[j]+u[j]
            if gradient:
                z=previous+before[:,None]
                derivative=np.mean((base.g(z)-target)*base.derivative(z),axis=1)
                b[:,j]=np.clip(before-derivative/(4+trust),-self.bound,self.bound)
            else:b[:,j]=bias_block(previous,target,before,trust,self.bound)
        if self.family.startswith('alm_'):
            previous=self.x
            for j in range(depth):
                u[j]+=self.dual_rate*(h[j]-base.g(previous+b[:,j,None]));previous=h[j]
        self.sweeps+=1


def fit(start,x,v,q,*,family,steps=128,epsilon=.001,tau=1.,rho=1.,trust=.01,dual_rate=.5,bound=.12,lr=.003,trace=False):
    begin=time.perf_counter()
    assert family in FAMILIES and isinstance(steps,int) and steps>=0 and tau>0 and rho>0 and trust>0 and bound>0 and lr>0
    start,x,v,q=[np.array(a,dtype=float,copy=True) for a in (start,x,v,q)]
    assert start.ndim==2 and x.shape==v.shape and x.ndim==q.ndim==1 and len(x)>0
    assert all(np.isfinite(a).all() for a in [start,x,v,q]) and np.max(abs(start))<=bound
    bands(v,epsilon);restarts,depth=start.shape
    b=start.copy();best=b.copy();initial_jacobian=family.startswith('bp_') and steps>0
    value,residual,jac,_=evaluate(b,x,v,epsilon=epsilon,tau=tau,jacobian=initial_jacobian)
    best_value=value.copy()
    best_norm=.5*np.sum(best**2,axis=1);forward_rows=restarts;jacobian_rows=restarts*int(initial_jacobian)
    history=[]
    def retain(point,value):
        norm=.5*np.sum(point**2,axis=1)
        changed=(value<best_value)|((value==best_value)&(norm<best_norm))
        best[changed]=point[changed];best_value[changed]=value[changed];best_norm[changed]=norm[changed]
    def snapshot():
        if trace:history.append(dict(b=b.copy(),best=best.copy(),objective=best_value.copy()))
    snapshot();local=None;optimizer_state={}
    if family.startswith('bp_'):
        first=np.zeros_like(b);second=np.zeros_like(b);damping=np.full(restarts,.01)
        for step in range(1,steps+1):
            # Cache this point's already computed objective/Jacobian, so the BP
            # control does not pay a redundant pre-update forward at every step.
            grad=np.einsum('rni,rn->ri',jac,residual,optimize=False)/(len(x)*tau)
            if family=='bp_adam':
                first=.9*first+.1*grad;second=.999*second+.001*grad**2
                b=np.clip(b-lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8),-bound,bound)
            else:
                hessian=np.einsum('rni,rnj->rij',jac,jac,optimize=False)/(len(x)*tau)
                scale=np.maximum(np.diagonal(hessian,axis1=1,axis2=2),1e-4)
                system=hessian+damping[:,None,None]*np.eye(depth)[None]*scale[:,None,:]
                delta=np.linalg.solve(system,-grad[:,:,None])[:,:,0]
                delta*=np.minimum(1.,.05/np.maximum(np.max(abs(delta),axis=1),1e-30))[:,None]
                next_b=b.copy();next_value=value.copy();changed=np.zeros(restarts,bool)
                for alpha in [1.,.5,.25,.125,.0625]:
                    trial=np.clip(b+alpha*delta,-bound,bound)
                    trial_value=evaluate(trial,x,v,epsilon=epsilon,tau=tau)[0];forward_rows+=restarts
                    retain(trial,trial_value);accepted=trial_value<next_value
                    next_b[accepted]=trial[accepted];next_value[accepted]=trial_value[accepted];changed|=accepted
                b=next_b;damping=np.clip(damping*np.where(changed,.3,10.),1e-8,1e8)
            with_jacobian=step<steps
            value,residual,jac,_=evaluate(b,x,v,epsilon=epsilon,tau=tau,jacobian=with_jacobian)
            forward_rows+=restarts;jacobian_rows+=restarts*int(with_jacobian)
            retain(b,value);snapshot()
        optimizer_state=dict(adam_first=first,adam_second=second,gn_damping=damping)
    else:
        local=Local(start,x,v,family,epsilon=epsilon,tau=tau,rho=rho,trust=trust,dual_rate=dual_rate,bound=bound)
        for _ in range(steps):
            local.step();b=local.b
            value=evaluate(b,x,v,epsilon=epsilon,tau=tau)[0];forward_rows+=restarts;retain(b,value);snapshot()
    index=min(range(restarts),key=lambda i:(best_value[i],best_norm[i],i))
    selected=best[index].copy();prediction=base.forward(q,selected)
    arrays=dict(x=x,v=v,q=q,starts=start,best_bank=best,best_objective=best_value,best_squared_norm=best_norm,
                current_bank=b.copy(),selected_point=selected,prediction=prediction,**optimizer_state)
    if local is not None:arrays.update(activities=local.h.copy(),scaled_duals=local.u.copy())
    if trace:
        for key in ['b','best','objective']:arrays['trace_'+key]=np.array([row[key] for row in history])
    metadata=dict(family=family,steps=steps,epsilon=epsilon,tau=tau,rho=rho,trust=trust,dual_rate=dual_rate,bound=bound,
        selected_index=index,selection_rule='minimum common J, exact ties by squared bias norm, then restart index',
        task_objective='mean(dist(f_b(x), clipped observation band)^2)/(2*tau)',
        global_bp_used=family.startswith('bp_'),dual_initialized_from_bp=False,
        online_multiplier_initialization='zero' if local is not None else 'not used',
        query_targets_accessed=False,cross_task_state_reused=False,
        full_network_forward_rows_for_objective=forward_rows,whole_chain_jacobian_rows=jacobian_rows,
        local_activity_initialization_forward_rows=restarts if local is not None else 0,query_readout_forwards=1,
        output_readout='single common-objective selected point; no posterior union',
        named_returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        resource_scope='returned array subtotal only, not peak solver state or a matched runtime benchmark',
        total_seconds=time.perf_counter()-begin)
    return arrays,metadata

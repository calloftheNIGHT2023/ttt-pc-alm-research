"""Support-only piecewise-quadratic minimization on a parameter segment.

Global only along the supplied direction, in floating-point arithmetic. This
uses whole-network forward algebra, not a local PC operation or loss gradient.
"""
import numpy as np
import vector_interval_memory as base
SLOPES=np.array([0.,2.,-2.,0.]);INTERCEPTS=np.array([-1.,1.,1.,-1.])


def values(bank,x,v,weights):
    raw=base.forward(bank,x,weights)-v
    return .5*np.mean(np.sum(base.residuals(raw)**2,axis=-1),axis=-1)


def segments(old,new,x,weights):
    restarts,depth,width=old.shape;n=len(x);direction=new-old
    restart=np.repeat(np.arange(restarts),n);sample=np.tile(np.arange(n),restarts)
    lower=np.zeros(restarts*n);upper=np.ones_like(lower)
    slope=np.zeros((restarts*n,width));intercept=x[sample].copy();maximum=len(lower);workspace=0
    for j,weight in enumerate(weights):
        za=slope@weight.T+direction[restart,j];zb=intercept@weight.T+old[restart,j]
        with np.errstate(divide='ignore',invalid='ignore'):
            roots=(np.array([-1.,0.,1.])[None,None,:]-zb[:,:,None])/za[:,:,None]
        roots=roots.reshape(len(lower),-1)
        roots=np.where(np.isfinite(roots),np.clip(roots,lower[:,None],upper[:,None]),lower[:,None])
        cuts=np.sort(np.c_[lower,roots,upper],axis=1);left,right=cuts[:,:-1],cuts[:,1:]
        parent,column=np.nonzero(right>left);next_lower,next_upper=left[parent,column],right[parent,column]
        middle=(next_lower+next_upper)/2;region=np.searchsorted([-1.,0.,1.],za[parent]*middle[:,None]+zb[parent],side='left')
        slope=SLOPES[region]*za[parent];intercept=SLOPES[region]*zb[parent]+INTERCEPTS[region]
        restart,sample=restart[parent],sample[parent];lower,upper=next_lower,next_upper
        maximum=max(maximum,len(lower));workspace=max(workspace,sum(t.nbytes for t in [za,zb,roots,cuts,parent,column,slope,intercept,restart,sample,lower,upper,region]))
    return dict(restart=restart,sample=sample,lower=lower,upper=upper,slope=slope,intercept=intercept,
                max_segments=maximum,segment_workspace_bytes_subtotal=workspace)


def loss_pieces(state,v,n,restarts):
    width=v.shape[1]
    restart=np.repeat(state['restart'],width)
    slope=state['slope'].ravel();offset=(state['intercept']-v[state['sample']]).ravel()
    lower=np.repeat(state['lower'],width);upper=np.repeat(state['upper'],width)
    with np.errstate(divide='ignore',invalid='ignore'):
        roots=(np.array([-base.EPS,base.EPS])[None,:]-offset[:,None])/slope[:,None]
    roots=np.where(np.isfinite(roots),np.clip(roots,lower[:,None],upper[:,None]),lower[:,None])
    cuts=np.sort(np.c_[lower,roots,upper],axis=1);left,right=cuts[:,:-1],cuts[:,1:]
    parent,column=np.nonzero(right>left);lo,hi=left[parent,column],right[parent,column]
    midpoint=(lo+hi)/2;residual=slope[parent]*midpoint+offset[parent]
    sign=np.where(residual>base.EPS,1.,np.where(residual<-base.EPS,-1.,0.))
    aa=np.where(sign!=0,slope[parent],0.);bb=np.where(sign!=0,offset[parent]-sign*base.EPS,0.)
    coefficients=np.c_[aa**2,aa*bb,bb**2]*(.5/n)
    owner=restart[parent]
    event_owner=np.r_[owner,owner];event_at=np.r_[lo,hi];event_delta=np.vstack([coefficients,-coefficients])
    # Aggregate all samples and output channels for each restart. Duplicate
    # cuts have no open interval; evaluating partial sums there would be wrong.
    ordering=np.lexsort((event_at,event_owner));event_owner=event_owner[ordering]
    event_at=event_at[ordering];event_delta=event_delta[ordering]
    starts=np.searchsorted(event_owner,np.arange(restarts+1));pieces=[]
    for r in range(restarts):
        at=event_at[starts[r]:starts[r+1]];delta=event_delta[starts[r]:starts[r+1]]
        prefix=np.cumsum(delta,axis=0);keep=at[1:]>at[:-1]
        pieces.append(dict(lower=at[:-1][keep],upper=at[1:][keep],coefficients=prefix[:-1][keep]))
    workspace=sum(t.nbytes for t in [roots,cuts,parent,column,lo,hi,aa,bb,coefficients,event_owner,event_at,event_delta,ordering])
    return pieces,dict(line_event_count=len(event_at),event_workspace_bytes_subtotal=workspace)


def solve(old,new,x,v,weights,mode='exact'):
    if mode=='none':return new.copy(),np.ones(len(old)),dict(mode=mode)
    if mode=='grid':
        grid=np.array([0.,.125,.25,.5,1.])
        banks=old[None]+grid[:,None,None,None]*(new-old)[None]
        losses=values(banks.reshape(-1,*old.shape[1:]),x,v,weights).reshape(len(grid),len(old))
        index=np.argmin(losses,axis=0);alpha=grid[index]
        return old+alpha[:,None,None]*(new-old),alpha,dict(mode=mode,forward_candidates=5,filter_workspace_bytes_subtotal=banks.nbytes+losses.nbytes)
    if mode!='exact':raise ValueError(mode)
    state=segments(old,new,x,weights);pieces,meta=loss_pieces(state,v,len(x),len(old));alpha=np.zeros(len(old));minimum_curvature=np.inf
    for r,piece in enumerate(pieces):
        coeff=piece['coefficients'];aa,bb,cc=coeff.T;minimum_curvature=min(minimum_curvature,float(np.min(aa)))
        # Exact interval curvature is nonnegative. Negative roundoff is clipped;
        # direct forward validation below ensures the accepted step is safe.
        aa=np.maximum(aa,0.)
        stationary=np.divide(-bb,aa,out=piece['lower'].copy(),where=aa>0)
        point=np.clip(stationary,piece['lower'],piece['upper'])
        objective=aa*point**2+2*bb*point+cc;alpha[r]=point[np.argmin(objective)]
    analytic=old+alpha[:,None,None]*(new-old)
    candidates=np.stack([old,analytic,new]);losses=values(candidates.reshape(-1,*old.shape[1:]),x,v,weights).reshape(3,len(old))
    index=np.argmin(losses,axis=0);alpha=np.where(index==0,0.,np.where(index==2,1.,alpha))
    answer=old+alpha[:,None,None]*(new-old)
    return answer,alpha,dict(mode=mode,forward_candidates=3,max_line_segments=state['max_segments'],
                            minimum_accumulated_curvature=minimum_curvature,
                            filter_workspace_bytes_subtotal=state['segment_workspace_bytes_subtotal']+meta['event_workspace_bytes_subtotal'],**meta)

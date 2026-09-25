"""Finite lower-envelope solver for a shared-bias / all-context z block.

Conditional global optimality only; not a global neural-network optimizer.
Mathematical algorithm is exact over reals; implementation uses float64 roots.
"""
import numpy as np


SLOPES=(0.,2.,-2.,0.)
INTERCEPTS=(-1.,1.,1.,-1.)
LO=(-np.inf,-1.,0.,1.)
HI=(-1.,0.,1.,np.inf)


def activation(z):return np.maximum(-1.,1.-2*np.abs(z))


def prox(center,target,old,trust):
    s=np.array(SLOPES)[:,None];k=np.array(INTERCEPTS)[:,None]
    z=np.clip((center[None]+s*(target[None]-k)+trust*old[None])/(1+s*s+trust),np.array(LO)[:,None],np.array(HI)[:,None])
    e=(z-center)**2+(activation(z)-target)**2+trust*(z-old)**2
    idx=np.argmin(e,axis=0)
    return z[idx,np.arange(len(center))]


def energy(b,z,c,target,old,trust):
    return float(np.sum((z-c-b)**2+(activation(z)-target)**2+trust*(z-old)**2))


def pieces(c,t,old,trust,bound):
    intervals=[];coefficients=[]
    for s,k,lo,hi in zip(SLOPES,INTERCEPTS,LO,HI):
        den=1+s*s+trust;num=c+s*(t-k)+trust*old
        modes=[(1/den,num/den,lo*den-num,hi*den-num)]
        if np.isfinite(lo):modes.append((0.,lo,-np.inf,lo*den-num))
        if np.isfinite(hi):modes.append((0.,hi,hi*den-num,np.inf))
        for alpha,beta,left,right in modes:
            left=max(-bound,left);right=min(bound,right)
            if right<=left:continue
            a=(alpha-1)**2+(s*alpha)**2+trust*alpha**2
            b=2*((alpha-1)*(beta-c)+s*alpha*(s*beta+k-t)+trust*alpha*(beta-old))
            cc=(beta-c)**2+(s*beta+k-t)**2+trust*(beta-old)**2
            intervals.append((left,right));coefficients.append((a,b,cc))
    return np.array(intervals),np.array(coefficients)


def envelope(c,t,old,trust,bound):
    intervals,coeff=pieces(c,t,old,trust,bound)
    breaks=list(intervals.ravel())
    for i in range(len(coeff)):
        for j in range(i):
            left=max(intervals[i,0],intervals[j,0]);right=min(intervals[i,1],intervals[j,1])
            if left>=right:continue
            a,b,cc=coeff[i]-coeff[j]
            scale=max(1.,abs(a),abs(b),abs(cc));tiny=2e-14*scale
            if abs(a)<=tiny:
                roots=[] if abs(b)<=tiny else [-cc/b]
            else:
                disc=b*b-4*a*cc
                if disc < -2e-14*max(1.,b*b,abs(4*a*cc)):continue
                root=np.sqrt(max(0.,disc));q=-.5*(b+np.copysign(root,b))
                roots=[q/a,cc/q] if q else [-b/(2*a)]
            breaks.extend(r for r in roots if left<r<right)
    breaks=np.unique(breaks);mid=(breaks[:-1]+breaks[1:])/2
    values=coeff[:,0,None]*mid**2+coeff[:,1,None]*mid+coeff[:,2,None]
    values=np.where((mid>=intervals[:,0,None])&(mid<=intervals[:,1,None]),values,np.inf)
    winner=np.argmin(values,axis=0);assert np.isfinite(values[winner,np.arange(len(mid))]).all()
    changes=np.r_[True,winner[1:]!=winner[:-1]]
    return breaks[:-1][changes],coeff[winner[changes]]


def solve(c,target,old,trust=.01,bound=.3):
    c=np.asarray(c,dtype=float);target=np.asarray(target,dtype=float);old=np.asarray(old,dtype=float)
    assert c.ndim==1 and c.shape==target.shape==old.shape and len(c)>0 and trust>0 and bound>0
    initial=np.zeros(3);events=[];deltas=[];pieces_total=0
    for ci,ti,oi in zip(c,target,old):
        breaks,coef=envelope(ci,ti,oi,trust,bound);initial+=coef[0];pieces_total+=len(coef)
        events.extend(breaks[1:]);deltas.extend(np.diff(coef,axis=0))
    if events:
        order=np.argsort(events,kind='stable');events=np.array(events)[order];deltas=np.array(deltas)[order]
        unique,first=np.unique(events,return_index=True);merged=np.add.reduceat(deltas,first,axis=0)
        boundaries=np.r_[-bound,unique,bound];coefs=np.vstack([initial,initial+np.cumsum(merged,axis=0)])
    else:boundaries=np.array([-bound,bound]);coefs=initial[None]
    stationary=-coefs[:,1]/(2*np.maximum(coefs[:,0],1e-30))
    candidates=np.r_[boundaries,np.clip(stationary,boundaries[:-1],boundaries[1:])]
    # Re-evaluate actual proximal energy instead of trusting accumulated polynomial constants.
    s=np.array(SLOPES)[:,None,None];k=np.array(INTERCEPTS)[:,None,None]
    z=np.clip((c[None,:,None]+candidates[None,None,:]+s*(target[None,:,None]-k)+trust*old[None,:,None])/(1+s*s+trust),
        np.array(LO)[:,None,None],np.array(HI)[:,None,None])
    val=(z-c[None,:,None]-candidates[None,None,:])**2+(activation(z)-target[None,:,None])**2+trust*(z-old[None,:,None])**2
    totals=np.min(val,axis=0).sum(axis=0);winner=int(np.argmin(totals));bias=float(candidates[winner])
    bestz=prox(c+bias,target,old,trust)
    return bias,bestz,{'energy':energy(bias,bestz,c,target,old,trust),'individual_envelope_pieces':pieces_total,
        'sum_intervals':len(coefs),'candidate_biases':len(candidates),'contexts':len(c)}

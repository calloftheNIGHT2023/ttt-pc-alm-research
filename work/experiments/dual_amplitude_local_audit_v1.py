"""Independent scalar local-block solver and within-cell amplitude derivative.

This differentiates a single local block family with respect to its input
amplitude, not the whole-network supervised loss; diagnostic only.
"""
import math
import numpy as np

TAU=.01
BOUND=.12
SLOPES=[0.,2.,-2.,0.]
INTERCEPTS=[0.,0.,2.,0.]
KNOTS=[0.,.5,1.]


def branch(z):
    return sum(z>=k for k in KNOTS)


def g(z):
    k=branch(z)
    return SLOPES[k]*z+INTERCEPTS[k]


def clip(value,lo,hi):
    if value<lo:
        return lo,'lo'
    if value>hi:
        return hi,'hi'
    return value,'free'


def scalar_step(b0,h0,direction,x,v,alpha):
    """Primal half of the frozen ALM step, independent direct-cost enumeration."""
    d,n=h0.shape
    h=h0.copy()
    dh=np.zeros_like(h)
    b=b0.copy()
    db=np.zeros_like(b)
    signatures=[]
    for j in reversed(range(d)):
        for i in range(n):
            prev=x[i] if j==0 else h0[j-1,i]
            a=g(float(prev+b0[j]))-alpha*direction[j,i]
            da=-direction[j,i]
            if j==d-1:
                raw=(a+TAU*h0[j,i])/(1+TAU)
                value,status=clip(raw,max(0.,v[i]-.001),min(1.,v[i]+.001))
                h[j,i]=value
                dh[j,i]=da/(1+TAU) if status=='free' else 0.
                signatures.append(('h',j,i,4,status))
                continue
            candidates=[]
            target=h[j+1,i]+alpha*direction[j+1,i]
            dt=dh[j+1,i]+direction[j+1,i]
            lows=[-math.inf,0.,.5,1.]
            highs=[0.,.5,1.,math.inf]
            for k,(s,c) in enumerate(zip(SLOPES,INTERCEPTS)):
                lo=max(0.,lows[k]-b0[j+1])
                hi=min(1.,highs[k]-b0[j+1])
                if lo>hi:
                    candidates.append((math.inf,0.,0.,'invalid'))
                    continue
                offset=s*b0[j+1]+c
                raw=(a+s*(target-offset)+TAU*h0[j,i])/(1+s*s+TAU)
                value,status=clip(raw,lo,hi)
                deriv=(da+s*dt)/(1+s*s+TAU) if status=='free' else 0.
                energy=(value-a)**2+(g(float(value+b0[j+1]))-target)**2+TAU*(value-h0[j,i])**2
                candidates.append((energy,value,deriv,status))
            k=min(range(4),key=lambda t:candidates[t][0])
            _,h[j,i],dh[j,i],status=candidates[k]
            signatures.append(('h',j,i,k,status))
    for j in range(d):
        prev=x.copy() if j==0 else h[j-1].copy()
        dp=np.zeros(n) if j==0 else dh[j-1].copy()
        target=h[j]+alpha*direction[j]
        dt=dh[j]+direction[j]
        # Enumerate all intervals directly; do not reuse the production cumsum.
        events=[]
        for ki,knot in enumerate(KNOTS):
            for i in range(n):
                value,status=clip(knot-prev[i],-BOUND,BOUND)
                deriv=-dp[i] if status=='free' else 0.
                events.append((value,deriv,ki*n+i))
        events.sort(key=lambda t:t[0])
        bounds=[(-BOUND,0.,-1)]+events+[(BOUND,0.,-2)]
        candidates=[]
        for low,high in zip(bounds[:-1],bounds[1:]):
            mid=(low[0]+high[0])/2
            ks=[branch(float(p+mid)) for p in prev]
            slopes=np.array([SLOPES[k] for k in ks])
            offsets=np.array([INTERCEPTS[k] for k in ks])
            den=math.fsum(float(s*s) for s in slopes)/n+TAU
            num=math.fsum(float(s*(t-s*p-c)) for s,t,p,c in zip(slopes,target,prev,offsets))/n+TAU*b0[j]
            raw=num/den
            value,status=clip(raw,low[0],high[0])
            deriv=math.fsum(float(s*(t-s*p)) for s,t,p in zip(slopes,dt,dp))/n/den
            if status=='lo': deriv=low[1]
            if status=='hi': deriv=high[1]
            cost=math.fsum((g(float(p+value))-float(t))**2 for p,t in zip(prev,target))/n+TAU*(value-b0[j])**2
            candidates.append((cost,value,deriv,tuple(ks),status,low[2],high[2]))
        winner=min(range(len(candidates)),key=lambda k:candidates[k][0])
        _,b[j],db[j],ks,status,lowid,hiid=candidates[winner]
        signatures.append(('b',j,ks,status,lowid,hiid,tuple(t[2] for t in events)))
    return dict(b=b,h=h,db=db,dh=dh,signature=tuple(signatures))


def audit_state(b,h,u,x,v,production):
    """Three pre-fixed local windows plus a wide cross-branch counterexample."""
    narrow=[]
    max_scalar=0.
    for alpha in [0.,.5,1.]:
        eps=1e-7
        triples=[scalar_step(b,h,u,x,v,a) for a in [alpha-eps,alpha,alpha+eps]]
        center=triples[1]
        pb,ph=production(alpha)
        max_scalar=max(max_scalar,float(np.max(abs(center['b']-pb))),float(np.max(abs(center['h']-ph))))
        same=triples[0]['signature']==center['signature']==triples[2]['signature']
        gaps=[]
        if same:
            for sign,r in [(-1,triples[0]),(1,triples[2])]:
                gaps.extend([float(np.max(abs(r['b']-center['b']-sign*eps*center['db']))),
                             float(np.max(abs(r['h']-center['h']-sign*eps*center['dh'])))])
            assert max(gaps)<1e-10
        narrow.append(dict(alpha=alpha,epsilon=eps,same_internal_branch_signature=same,
                           max_affine_gap=max(gaps) if gaps else None))
    wide=[scalar_step(b,h,u,x,v,a) for a in [-1.,0.,1.]]
    midpoint_gap=max(float(np.max(abs(wide[1]['b']-(wide[0]['b']+wide[2]['b'])/2))),
                     float(np.max(abs(wide[1]['h']-(wide[0]['h']+wide[2]['h'])/2))))
    assert max_scalar<1e-10,max_scalar
    return dict(narrow=narrow,max_scalar_gap=max_scalar,wide_midpoint_gap=midpoint_gap,
        wide_internal_signature_changes=not(wide[0]['signature']==wide[1]['signature']==wide[2]['signature']))

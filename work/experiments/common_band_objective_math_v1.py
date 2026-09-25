"""407 exact mathematical primitives; no BP-based multiplier initialization."""
from fractions import Fraction as F


def clip(value, low, high):
    value,low,high=map(F,(value,low,high))
    assert low<=high
    return min(high,max(low,value))


def tent(z):
    return max(F(0),1-abs(2*F(z)-1))


def slope(z):
    z=F(z)
    assert z not in [0,F(1,2),1], 'No classical derivative is asserted at a knot.'
    return F(0) if z<0 or z>1 else (F(2) if z<F(1,2) else F(-2))


def band(v,eps):
    lo,hi=max(F(0),F(v)-F(eps)),min(F(1),F(v)+F(eps))
    assert F(eps)>=0 and lo<=hi
    return lo,hi


def loss(y,low,high,tau=1):
    tau=F(tau);assert tau>0
    return (F(y)-clip(y,low,high))**2/(2*tau)


def loss_derivative(y,low,high,tau=1):
    tau=F(tau);assert tau>0
    return (F(y)-clip(y,low,high))/tau


def output_prox(a,previous,low,high,*,rho=1,trust=F(1,100),tau=1):
    a,previous,low,high,rho,trust,tau=map(F,(a,previous,low,high,rho,trust,tau))
    assert rho>0 and trust>=0 and tau>0 and 0<=low<=high<=1
    kappa=rho*(1+trust);center=(a+trust*previous)/(1+trust)
    point=(kappa*tau*center+clip(center,low,high))/(1+kappa*tau)
    return clip(point,0,1)


def output_energy(h,a,previous,low,high,*,rho=1,trust=F(1,100),tau=1):
    h,a,previous,rho,trust,tau=map(F,(h,a,previous,rho,trust,tau))
    assert rho>0 and trust>=0 and tau>0
    return loss(h,low,high,tau)+rho*(h-a)**2/2+rho*trust*(h-previous)**2/2


def forward_trace(x,bias):
    previous=list(map(F,x));trace=[]
    for b in bias:
        previous=[tent(t+F(b)) for t in previous];trace.append(previous)
    return trace


def objective(x,v,bias,eps,*,tau=1):
    output=forward_trace(x,bias)[-1]
    assert len(output)==len(v)>0
    return sum((loss(y,*band(t,eps),tau) for y,t in zip(output,v)),F(0))/len(v)


def augmented(x,v,bias,h,u,eps,*,rho=1,tau=1):
    rho=F(rho);assert rho>0 and F(tau)>0
    n=len(x);depth=len(bias)
    assert n==len(v)>0 and len(h)==len(u)==depth and depth>0
    assert all(len(row)==n for row in h+u)
    value=sum((loss(F(y),*band(t,eps),tau) for y,t in zip(h[-1],v)),F(0))
    previous=list(map(F,x))
    for b,activity,dual in zip(bias,h,u):
        for last,current,multiplier in zip(previous,activity,dual):
            residual=F(current)-tent(last+F(b))
            value+=rho*(F(multiplier)*residual+residual**2/2)
        previous=list(map(F,activity))
    return value/n


def local_partials(x,v,bias,h,u,eps,*,rho=1,tau=1):
    """Adjacent-layer derivatives only; no entire-network chain is formed."""
    rho=F(rho);n=len(x);depth=len(bias)
    assert rho>0 and n==len(v)>0 and len(h)==len(u)==depth
    credits=[];derivatives=[];previous=list(map(F,x))
    for b,activity,dual in zip(bias,h,u):
        z=[last+F(b) for last in previous]
        derivatives.append([slope(t) for t in z])
        credits.append([F(uu)+F(hh)-tent(t) for uu,hh,t in zip(dual,activity,z)])
        previous=list(map(F,activity))
    gb=[-rho*sum((c*d for c,d in zip(cc,dd)),F(0))/n for cc,dd in zip(credits,derivatives)]
    gh=[]
    for layer in range(depth):
        row=[]
        for i in range(n):
            value=rho*credits[layer][i]
            if layer+1<depth:value-=rho*derivatives[layer+1][i]*credits[layer+1][i]
            else:value+=loss_derivative(h[layer][i],*band(v[i],eps),tau)
            row.append(value/n)
        gh.append(row)
    return gb,gh

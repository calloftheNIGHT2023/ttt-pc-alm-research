"""258B exact piecewise-quadratic first crossing on the same selected block.

The direction is the old actual binary write divided by its grid tau. Thus
the old physical certificate supplies an exact feasible upper bound.
"""
from fractions import Fraction as F
from functools import cmp_to_key
from math import isqrt
import local_dual_jump as original

def pack(t):return original.exact.pack(t)
def encode(z):
    if isinstance(z,F):return pack(z)
    if isinstance(z,dict):return {k:encode(v) for k,v in z.items()}
    if isinstance(z,(list,tuple)):return [encode(v) for v in z]
    return z

def sign(z):return (z>0)-(z<0)
def surd_sign(r,s,d):
    """Exact sign(r+s*sqrt(d)); rational d>=0, no numerical radical."""
    assert d>=0
    if not d or not s:return sign(r)
    if not r:return sign(s)
    if sign(r)==sign(s):return sign(r)
    return sign(r)*sign(r*r-s*s*d)

def root_descriptor(poly):
    a,b,c=poly
    if not a:assert b>0;return dict(kind='rational',value=-c/b)
    disc=b*b-4*a*c;assert disc>=0
    nr,dr=isqrt(disc.numerator),isqrt(disc.denominator)
    if nr*nr==disc.numerator and dr*dr==disc.denominator:return dict(kind='rational',value=(-b+F(nr,dr))/(2*a))
    return dict(kind='quadratic',poly=list(poly),discriminant=disc)

def root_vs_rational(root,q):
    if root['kind']=='rational':return sign(root['value']-q)
    a,b,c=root['poly'];return sign(a)*surd_sign(-b-2*a*q,F(1),root['discriminant'])

def polynomial_at_root_sign(poly,root):
    if root['kind']=='rational':return sign(value(poly,root['value']))
    a,b,c=root['poly'];aa,bb,cc=poly;u=bb-aa*b/a;v=cc-aa*c/a
    return sign(a)*surd_sign(-u*b+2*a*v,u,root['discriminant'])

def compare_roots(left,right):
    lr,rr=left['root'],right['root']
    if lr['kind']=='rational':return -root_vs_rational(rr,lr['value'])
    if root_vs_rational(rr,left['segment'][0])<0:return 1
    if root_vs_rational(rr,left['positive_point'])>=0:return -1
    # On this left-to-positive bracket, its polynomial is <=0 before its
    # unique first crossing and >0 after. A later falling root is excluded.
    return -polynomial_at_root_sign(left['poly'],rr)

def value(poly,t):a,b,c=poly;return (a*t+b)*t+c
def maximum(poly,lo,hi):
    a,b,c=poly;points=[lo,hi]
    if a<0:
        vertex=-b/(2*a)
        if lo<vertex<hi:points.append(vertex)
    return max(((value(poly,t),t) for t in points),key=lambda p:(p[0],-p[1]))

def branches(s,d):
    prev,bias,nb,z0,y=s;incoming=original.exact.g(prev+bias);ans=[]
    for k,slope in enumerate(original.exact.S):
        lo=F(0) if k==0 else max(F(0),original.exact.K[k-1]-nb)
        hi=F(1) if k==3 else min(F(1),original.exact.K[k]-nb)
        if lo>hi:continue
        offset=slope*nb+original.exact.C[k];den=1+slope*slope+original.exact.TRUST
        alpha=(incoming+slope*(y-offset)+original.exact.TRUST*z0)/den;beta=(-d[0]+slope*d[1])/den
        ans.append(dict(k=k,lo=lo,hi=hi,slope=slope,offset=offset,alpha=alpha,beta=beta))
    return ans

def coefficients(s,d,branch,mid):
    prev,bias,nb,z0,y=s;a=original.exact.g(prev+bias);aa,bb=branch['alpha'],branch['beta'];z=aa+bb*mid
    if z<=branch['lo']:aa,bb=branch['lo'],F(0)
    elif z>=branch['hi']:aa,bb=branch['hi'],F(0)
    s0,c=branch['slope'],branch['offset'];r1=aa-a;r2=y-s0*aa-c;r3=aa-z0;t1=bb+d[0];t2=d[1]-s0*bb;t3=bb;gamma=original.exact.TRUST
    return [t1*t1+t2*t2+gamma*t3*t3,2*(r1*t1+r2*t2+gamma*r3*t3),r1*r1+r2*r2+gamma*r3*r3]

def threshold(s,d,cap):
    blocks=branches(s,d);k0=original.branch(s[3]+s[2]);cuts={F(0),cap}
    for b in blocks:
        if b['beta']:
            for endpoint in [b['lo'],b['hi']]:
                t=(endpoint-b['alpha'])/b['beta']
                if 0<t<cap:cuts.add(t)
    cuts=sorted(cuts);pieces=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        polynomials={b['k']:coefficients(s,d,b,(lo+hi)/2) for b in blocks};pieces.append(dict(lo=lo,hi=hi,polynomials=polynomials))
    competitors=[]
    for block in blocks:
        k=block['k']
        if k==k0:continue
        checked=[];found=None
        for piece in pieces:
            lo,hi=piece['lo'],piece['hi'];poly=[a-b for a,b in zip(piece['polynomials'][k0],piece['polynomials'][k])];maximum_value,point=maximum(poly,lo,hi)
            checked.append(dict(lo=lo,hi=hi,poly=poly,maximum=maximum_value,maximum_at=point))
            if maximum_value<=0:continue
            assert value(poly,lo)<=0 and point>lo
            low,high=lo,point
            for _ in range(32):
                mid=(low+high)/2
                if value(poly,mid)>0:high=mid
                else:low=mid
            found=dict(k=k,poly=poly,segment=[lo,hi],positive_point=point,root=root_descriptor(poly),lower=low,upper=high)
            assert root_vs_rational(found['root'],low)>=0 and root_vs_rational(found['root'],high)<0
            break
        competitors.append(dict(k=k,checked=checked,threshold=found))
    roots=[c['threshold'] for c in competitors if c['threshold'] is not None];assert roots,'Old physical endpoint must be strictly feasible.'
    def compare(a,b):return compare_roots(a,b) or sign(a['k']-b['k'])
    ordered=sorted(roots,key=cmp_to_key(compare));chosen=ordered[0]
    return dict(current_branch=k0,branches=blocks,cuts=cuts,pieces=pieces,competitors=competitors,chosen=chosen,
        root_order=[r['k'] for r in ordered],comparisons=[dict(k=r['k'],sign=compare_roots(chosen,r)) for r in roots])

def minimum_event(x,b,h,old):
    if old is None:return dict(selected=None,certificate=None,fallbacks=0)
    j,i=old['j'],old['i'];fs=original.scalars(x,b,h,j,i);rs=original.scalars(x,b,h,j,i,True);cap=F(old['tau']);old_u=[F(float(t)) for t in old['u']];direction=[t/cap for t in old_u]
    zero=original.values(rs,F(0),True);fzero=original.values(fs,0.);endpoint=original.values(rs,cap,True,old_u)
    assert original.classify(zero,endpoint)['accepted']
    proof=threshold(rs,direction,cap);proposed=proof['chosen']['upper'];tau=min(float(cap),float(proposed));attempts=[];chosen=None
    for attempt in range(17):
        if attempt==16:tau=float(cap)
        exact_tau=F(tau);u=[float(exact_tau*t) for t in direction];fv=original.values(fs,0.,u_override=u);fd=original.classify(fzero,fv)
        ev=original.values(rs,exact_tau,True,[exact_tau*t for t in direction]);ed=original.classify(zero,ev);rounded=original.check_rounded(rs,fv,fd) if fd['accepted'] else None
        accepted=bool(fd['accepted'] and ed['accepted'] and rounded['accepted']);attempts.append(dict(tau=exact_tau,accepted=accepted,float_positive=bool(fd['accepted']),ideal_positive=bool(ed['accepted']),rounded=rounded))
        if accepted:
            chosen=dict(tau=tau,j=j,i=i,k=fd['competing_branch'],z=float(fd['z']),u=u,exact_margin=pack(ed['margin']),rounded_margin=rounded['margin']);break
        tau=float((exact_tau+cap)/2)
    assert chosen is not None
    new_u=[F(float(t)) for t in chosen['u']];old_norm=sum(t*t for t in old_u);new_norm=sum(t*t for t in new_u)
    assert F(chosen['tau'])<=cap and new_norm<=old_norm and all(abs(a)<=abs(z) for a,z in zip(new_u,old_u))
    float_residual=[F(float(t)) for t in fzero['residual']]
    proof.update(direction=direction,ideal_residual=zero['residual'],float_residual=float_residual,direction_minus_float=[a-z for a,z in zip(direction,float_residual)],
        old_tau=cap,old_u=old_u,old_norm2=old_norm,new_norm2=new_norm,norm2_ratio=new_norm/old_norm,attempts=attempts,
        implementation_tau=F(chosen['tau']),implementation_above_threshold_upper=F(chosen['tau'])-proof['chosen']['upper'])
    return dict(selected=chosen,certificate=encode(proof),fallbacks=len(attempts)-1)

def verify():
    def make(poly,lo,hi):
        poly=list(map(F,poly));v,p=maximum(poly,F(lo),F(hi));assert v>0 and value(poly,F(lo))<=0
        return dict(root=root_descriptor(poly),poly=poly,segment=[F(lo),F(hi)],positive_point=p)
    a=make([1,0,-2],0,2);b=make([1,0,-3],0,2);c=make([-1,4,-2],0,1);same=make([2,0,-4],0,2)
    assert compare_roots(a,b)<0 and compare_roots(b,a)>0 and compare_roots(a,same)==0 and compare_roots(c,a)<0
    small=F(1,2**80);narrow=make([-1,2+small,-1-small],0,1+small)
    after=make([0,1,-1-2*small],0,2);before=make([0,1,-1+2*small],0,2)
    assert compare_roots(narrow,after)<0 and compare_roots(narrow,before)>0
    return dict(passed=True,exact_algebraic_order_cases=6,narrow_positive_interval_width=str(small))

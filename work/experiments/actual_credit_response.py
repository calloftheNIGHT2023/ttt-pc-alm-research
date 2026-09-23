"""279 exact quadratic envelopes and true-forward open-segment coverage.

This is a mathematical diagnostic, not a deployment implementation. Isolated
breakpoints are explicitly outside the open-segment completeness claim.
"""
from fractions import Fraction as F
from functools import cmp_to_key
from math import gcd,isqrt,lcm

K=(F(0),F(1,2),F(1));S=(0,2,-2,0);C=(0,0,2,0)
BOUND=F(float(.12));TRUST=F(float(.01))


def fraction(x):return x if isinstance(x,F) else F(float(x))
def branch(z):return sum(z>=k for k in K)
def fold(z):return max(F(0),1-abs(2*z-1))
def sign(z):return int(z>0)-int(z<0)
def poly(q,t):return q[0]+t*(q[1]+t*q[2])
def subtract(a,b):return tuple(x-y for x,y in zip(a,b))


class Root:
    def __init__(self,value=None,abc=None,side=None):
        self.value=value;self.abc=abc;self.side=side
        if value is not None:self.lo=self.hi=value;self.key=('q',value)
        else:
            a,b,c=abc;disc=b*b-4*a*c;assert a>0 and disc>0 and isqrt(disc)**2!=disc
            bound=F(1)+max(F(abs(b),a),F(abs(c),a));self.lo=-bound;self.hi=bound;self.key=('r',abc,side)

    def compare_rational(self,q):
        if self.value is not None:return sign(self.value-q)
        a,b,c=self.abc;disc=b*b-4*a*c;r=-F(b)-2*a*q
        if not r:return self.side
        if sign(r)==self.side:return self.side
        return sign(r)*sign(r*r-disc)

    def refine(self):
        if self.value is not None:return
        mid=(self.lo+self.hi)/2
        if self.compare_rational(mid)>0:self.lo=mid
        else:self.hi=mid

    def __repr__(self):return str(self.value) if self.value is not None else f'root({self.abc},{self.side})'


def rational(x):return Root(value=F(x))


def compare(a,b):
    if a.key==b.key:return 0
    if b.value is not None:return a.compare_rational(b.value)
    if a.value is not None:return -b.compare_rational(a.value)
    for _ in range(4096):
        if a.hi<=b.lo:return -1
        if b.hi<=a.lo:return 1
        a.refine();b.refine()
    raise ArithmeticError('Algebraic root isolation budget exhausted')


def ordered(values):return sorted({r.key:r for r in values}.values(),key=cmp_to_key(compare))


def between(a,b):
    assert compare(a,b)<0
    for _ in range(4096):
        if a.hi<b.lo:return (a.hi+b.lo)/2
        a.refine();b.refine()
    raise ArithmeticError('Rational representative isolation budget exhausted')


def roots(q):
    c,b,a=q
    if not a:return [] if not b else [rational(-c/b)]
    den=lcm(*(x.denominator for x in [a,b,c]));aa,bb,cc=[int(x*den) for x in [a,b,c]];d=gcd(gcd(abs(aa),abs(bb)),abs(cc));aa//=d;bb//=d;cc//=d
    if aa<0:aa,bb,cc=-aa,-bb,-cc
    disc=bb*bb-4*aa*cc
    if disc<0:return []
    sq=isqrt(disc)
    if sq*sq==disc:return ordered([rational(F(-bb-sq,2*aa)),rational(F(-bb+sq,2*aa))])
    return [Root(abc=(aa,bb,cc),side=-1),Root(abc=(aa,bb,cc),side=1)]


def value_sign(q,root):
    if root.value is not None:return sign(poly(q,root.value))
    a,b,c=root.abc;linear=q[1]-q[2]*F(b,a);constant=q[0]-q[2]*F(c,a)
    if not linear:return sign(constant)
    return sign(linear)*root.compare_rational(-constant/linear)


def nonnegative(q,lo,hi):
    if value_sign(q,lo)<0 or value_sign(q,hi)<0:return False
    if q[2]>0:
        vertex=-q[1]/(2*q[2]);vv=rational(vertex)
        if compare(lo,vv)<0 and compare(vv,hi)<0 and poly(q,vertex)<0:return False
    return True


def encode(obj):
    if isinstance(obj,F):return str(obj)
    if isinstance(obj,Root):
        if obj.value is not None:return dict(kind='rational',value=str(obj.value))
        return dict(kind='quadratic',abc=list(obj.abc),side=obj.side,isolating_interval=[str(obj.lo),str(obj.hi)])
    if isinstance(obj,dict):return {str(k):encode(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [encode(v) for v in obj]
    return obj


def decode_root(data):
    if data['kind']=='rational':return rational(F(data['value']))
    r=Root(abc=tuple(data['abc']),side=data['side']);r.lo,r.hi=map(F,data['isolating_interval']);assert r.compare_rational(r.lo)>0 and r.compare_rational(r.hi)<0
    return r


def candidates(previous,target,direction,old,bound=BOUND,trust=TRUST):
    previous=list(map(fraction,previous));target=list(map(fraction,target));direction=list(map(fraction,direction));old=fraction(old);n=len(previous)
    cuts=sorted({-bound,bound}|{k-p for p in previous for k in K if -bound<k-p<bound});ans=[]
    for index,(left,right) in enumerate(zip(cuts[:-1],cuts[1:])):
        mid=(left+right)/2;slopes=[S[branch(p+mid)] for p in previous];offsets=[s*p+C[branch(p+mid)] for p,s in zip(previous,slopes)]
        den=sum(s*s for s in slopes)/F(n)+trust
        alpha=(sum(s*(y-c) for s,y,c in zip(slopes,target,offsets))/n+trust*old)/den;beta=sum(s*u for s,u in zip(slopes,direction))/n/den
        tcuts={F(0),F(1)}
        if beta:
            for endpoint in [left,right]:
                t=(endpoint-alpha)/beta
                if 0<t<1:tcuts.add(t)
        tcuts=sorted(tcuts)
        for lo,hi in zip(tcuts[:-1],tcuts[1:]):
            z=alpha+beta*(lo+hi)/2
            affine=(left,F(0)) if z<=left else (right,F(0)) if z>=right else (alpha,beta)
            aa,bb=affine;e0=[s*aa+c-y for s,c,y in zip(slopes,offsets,target)];e1=[s*bb-u for s,u in zip(slopes,direction)]
            q=(sum(e*e for e in e0)/n+trust*(aa-old)**2,2*sum(e*f for e,f in zip(e0,e1))/n+2*trust*(aa-old)*bb,sum(e*e for e in e1)/n+trust*bb*bb)
            ans.append(dict(index=len(ans),bias_interval=[left,right],domain=[lo,hi],affine=affine,cost=q,source_interval=index))
    return ans


def envelope(previous,target,direction,old,max_intersections=10000):
    choices=candidates(previous,target,direction,old);cuts=[rational(F(0)),rational(F(1))];intersections=[];tested=0
    for c in choices:cuts.extend(rational(t) for t in c['domain'])
    for i,first in enumerate(choices):
        for second in choices[i+1:]:
            lo=max(first['domain'][0],second['domain'][0]);hi=min(first['domain'][1],second['domain'][1])
            if lo>=hi:continue
            tested+=1
            if tested>max_intersections:raise OverflowError('Layer candidate intersection budget exceeded')
            q=subtract(first['cost'],second['cost'])
            for r in roots(q):
                if r.compare_rational(lo)>0 and r.compare_rational(hi)<0:cuts.append(r);intersections.append(dict(first=first['index'],second=second['index'],root=r))
    cuts=ordered(cuts);cells=[];pieces=[];checks=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        t=between(lo,hi);active=[c for c in choices if c['domain'][0]<t<c['domain'][1]]
        assert active;best=min(active,key=lambda c:(poly(c['cost'],t),c['index']))
        for c in active:
            assert nonnegative(subtract(c['cost'],best['cost']),lo,hi);checks+=1
        cell=dict(lo=lo,hi=hi,representative=t,winner=best['index'],active=[c['index'] for c in active]);cells.append(cell)
        if pieces and pieces[-1]['affine']==best['affine']:pieces[-1]['hi']=hi
        else:pieces.append(dict(lo=lo,hi=hi,affine=best['affine']))
    return dict(candidates=choices,cuts=cuts,intersections=intersections,cells=cells,pieces=pieces,pair_tests=tested,inequality_checks=checks)


def at(layer,t):
    active=[c for c in layer['candidates'] if c['domain'][0]<=t<=c['domain'][1]];value=min(poly(c['cost'],t) for c in active)
    winners=[c for c in active if poly(c['cost'],t)==value];distinct=sorted({c['affine'][0]+t*c['affine'][1] for c in winners})
    chosen=min(winners,key=lambda c:c['index']);return chosen['affine'][0]+t*chosen['affine'][1],distinct


def true_forward_piece(x,bias,lo,hi,max_segments=10000):
    states=[dict(lo=lo,hi=hi,output=[(fraction(xx),F(0)) for xx in x],codes=[])]
    for layer in range(4):
        following=[]
        for state in states:
            pre=[(a+bias[layer][0],b+bias[layer][1]) for a,b in state['output']];cuts=[state['lo'],state['hi']]
            for a,b in pre:
                if b:
                    for k in K:
                        r=rational((k-a)/b)
                        if compare(state['lo'],r)<0 and compare(r,state['hi'])<0:cuts.append(r)
            cuts=ordered(cuts)
            for a,b in zip(cuts[:-1],cuts[1:]):
                t=between(a,b);codes=[branch(p+q*t) for p,q in pre];output=[(S[k]*p+C[k],S[k]*q) for (p,q),k in zip(pre,codes)]
                following.append(dict(lo=a,hi=b,output=output,codes=state['codes']+codes))
                if len(following)>max_segments:raise OverflowError('State forward segment budget exceeded')
        states=following
    return [dict(lo=s['lo'],hi=s['hi'],representative=between(s['lo'],s['hi']),bias=bias,mode=bytes(s['codes']).hex()) for s in states]


def response_path(x,layers,max_segments=10000):
    cuts=ordered([r for layer in layers for piece in layer['pieces'] for r in [piece['lo'],piece['hi']]]);out=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        t=between(lo,hi);bias=[]
        for layer in layers:
            pieces=[p for p in layer['pieces'] if p['lo'].compare_rational(t)<0 and p['hi'].compare_rational(t)>0];assert len(pieces)==1;bias.append(pieces[0]['affine'])
        out.extend(true_forward_piece(x,bias,lo,hi,max_segments))
        if len(out)>max_segments:raise OverflowError('State forward segment budget exceeded')
    return out


def exact_mode(x,b):
    y=list(map(fraction,x));codes=[]
    for bias in b:
        z=[v+bias for v in y];codes.extend(branch(t) for t in z);y=[fold(t) for t in z]
    return bytes(codes).hex()


def verify():
    sq2=roots((F(-2),F(0),F(1)));sq3=roots((F(-3),F(0),F(1)))
    assert compare(sq2[1],sq3[1])<0 and compare(sq2[0],sq3[0])>0
    assert all(value_sign((F(-2),F(0),F(1)),r)==0 for r in sq2)
    assert compare(roots((F(-4),F(0),F(2)))[1],sq2[1])==0
    line=envelope([F(1,4)],[F(1,2)],[F(1,10)],F(0));gamma=TRUST
    for t in [F(0),F(1,3),F(1)]:assert at(line,t)[0]==(F(1,5)*t)/(4+gamma)
    clipped=envelope([F(1,4)],[F(1,2)],[F(1)],F(0));assert at(clipped,F(1))[0]==BOUND
    # Symmetric tent maxima give two exact minimizers; deterministic lowest
    # bias interval is selected and the alternate is explicitly reported.
    tie=envelope([F(1,2)],[F(4,5)],[F(0)],F(0));assert len(at(tie,F(1,2))[1])==2 and at(tie,F(1,2))[0]<0
    jump=envelope([F(49,100),F(51,100)],[F(3,4),F(17,20)],[F(1,5),F(-1,5)],F(0))
    assert at(jump,F(0))[0]<0 and at(jump,F(1))[0]>0
    assert len(at(jump,F(1,4))[1])==2 and len(jump['pieces'])>=2
    affine=[(F(0),F(1,10))]+[(F(0),F(0))]*3
    paths=true_forward_piece([F(1,10),F(2,5)],affine,rational(0),rational(1))
    for p in paths:
        t=p['representative'];assert exact_mode([F(1,10),F(2,5)],[a+b*t for a,b in affine])==p['mode']
    return dict(passed=True,root_order_and_identity=True,affine_solution=True,clipping=True,symmetric_tie=True,exact_jump_at_one_quarter=True,true_forward_pieces=len(paths))

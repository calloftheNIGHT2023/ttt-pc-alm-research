"""315 exact affine normalization of the selected 313 floating execution formulas.

Equality certifies formula identity, NOT winner optimality or future guards.
Index -1 denotes the constant, followed by flattened b,h,u input variables.
"""
from fractions import Fraction as F
import hashlib
import json
import numpy as np

S=(0,2,-2,0);C=(0,0,2,0);K=(F(0),F(1,2),F(1))


class Affine:
    __slots__=('terms',)
    def __init__(self,value=0):
        if isinstance(value,dict):self.terms={int(k):F(v) for k,v in value.items() if v}
        else:self.terms={-1:F(value)} if value else {}
    def __add__(self,other):
        if not isinstance(other,Affine):other=Affine(other)
        d=self.terms.copy()
        for k,v in other.terms.items():
            d[k]=d.get(k,F(0))+v
            if not d[k]:del d[k]
        return Affine(d)
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,other):return self+-other if isinstance(other,Affine) else self+(-F(other))
    def __rsub__(self,other):return -self+other
    def __mul__(self,c):
        c=F(c)
        if c==1:return self
        return Affine({k:v*c for k,v in self.terms.items()}) if c else Affine()
    __rmul__=__mul__
    def __truediv__(self,c):return self*(1/F(c))
    def at(self,point):return sum((v*(F(1) if k==-1 else point[k]) for k,v in self.terms.items()),F(0))
    def canonical(self):return tuple(sorted(self.terms.items()))


def unpack(policies,schema):
    decoded={}
    for group,fields in schema.items():
        flat=np.asarray(policies[group]).reshape(-1);offset=0;record={}
        for field in fields:
            width=field['width'];record[field['name']]=flat[offset:offset+width].astype(int);offset+=width
        assert offset==len(flat);decoded[group]=record
    return decoded


def branch(value,code):return S[int(code)]*value+C[int(code)]


def choose_clip(raw,lo,hi,low_sign,high_sign):
    # The upper endpoint takes precedence at double equality / zero-width intervals.
    if high_sign>=0:return hi
    if low_sign<=0:return lo
    return raw


def build(policies,schema,x,v,method='alm',*,bound=.12,trust=.01,eps=.001):
    p=unpack(policies,schema);d=sum(g.startswith('bias_') for g in schema);n=len(x)
    x=list(map(F,x));v=list(map(F,v));B,T,E=F(bound),F(trust),F(eps)
    var=lambda i:Affine({i:F(1)})
    b=[var(j) for j in range(d)]
    h=[[var(d+j*n+i) for i in range(n)] for j in range(d)]
    u=[[var(d+d*n+j*n+i) for i in range(n)] for j in range(d)]
    before_h=[row[:] for row in h]
    for j in reversed(range(d)):
        q=p[f'activity_{j}']
        for i in range(n):
            prev=Affine(x[i]) if j==0 else before_h[j-1][i]
            a=branch(prev+b[j],q['read_regions'][i])-u[j][i]
            if j==d-1:
                raw=(a+T*before_h[j][i])/(1+T)
                lo,hi=Affine(max(F(0),v[i]-E)),Affine(min(F(1),v[i]+E))
                h[j][i]=choose_clip(raw,lo,hi,q['candidate_vs_low'][i],q['candidate_vs_high'][i])
            else:
                k=int(q['winner'][i]);s,c=S[k],C[k]
                assert q['interval_valid'][k]
                lo=Affine(0) if k==0 or q['low_source'][k]<=0 else Affine(K[k-1])-b[j+1]
                hi=Affine(1) if k==3 or q['high_source'][k]>=0 else Affine(K[k])-b[j+1]
                raw=(a+s*(h[j+1][i]+u[j+1][i]-s*b[j+1]-c)+T*before_h[j][i])/(1+s*s+T)
                h[j][i]=choose_clip(raw,lo,hi,q['candidate_vs_low'][k*n+i],q['candidate_vs_high'][k*n+i])
    newb=[]
    for j in range(d):
        q=p[f'bias_{j}'];winner=int(q['winner'][0]);order=q['stable_order'];valid=q['event_valid']
        prev=[Affine(a) for a in x] if j==0 else h[j-1]
        target=[h[j][i]+u[j][i] for i in range(n)]
        # Rebuild exact cumulative coefficients, also valid for finite-precision
        # branch records with an endpoint-rounding inconsistency. Never infer
        # a slope from a floating midpoint or silently repair recorded choices.
        pp=[S[int(k)] for k in q['initial_regions']]
        qq=[s*s for s in pp];rr=[S[int(k)]*C[int(k)] for k in q['initial_regions']]
        for e in order[:winner]:
            if valid[e]:
                knot,i=divmod(int(e),n)
                pp[i]+=S[knot+1]-S[knot]
                qq[i]+=S[knot+1]**2-S[knot]**2
                rr[i]+=S[knot+1]*C[knot+1]-S[knot]*C[knot]
        aa=sum(map(F,qq))/n
        aa=max(aa,F(0))
        bb=sum((pp[i]*target[i]-qq[i]*prev[i]-rr[i] for i in range(n)),Affine())/n
        raw=(bb+T*b[j])/(aa+T)
        def endpoint(index,outer):
            if index is None:return Affine(outer)
            e=int(order[index]);k,i=divmod(e,n)
            if q['event_vs_box_low'][e]<=0:return Affine(-B)
            if q['event_vs_box_high'][e]>=0:return Affine(B)
            return Affine(K[k])-prev[i]
        lo=endpoint(winner-1 if winner else None,-B)
        hi=endpoint(winner if winner<3*n else None,B)
        newb.append(choose_clip(raw,lo,hi,q['candidate_vs_low'][winner],q['candidate_vs_high'][winner]))
    newu=[]
    assert method in ['alm','nodual'];rate=F(1,2) if method=='alm' else F(0)
    for j in range(d):
        prev=[Affine(a) for a in x] if j==0 else h[j-1]
        newu.append([u[j][i]+rate*(h[j][i]-branch(prev[i]+newb[j],p[f'residual_{j}']['regions'][i])) for i in range(n)])
    return newb+[t for row in h for t in row]+[t for row in newu for t in row]


def pack(b,h,u):return [F(float(t)) for t in np.r_[np.asarray(b).ravel(),np.asarray(h).ravel(),np.asarray(u).ravel()]]


def canonical(rows):return tuple(row.canonical() for row in rows)


def dump(rows):return [[[k,str(v)] for k,v in row.canonical()] for row in rows]


def restore(encoded):return [Affine({k:F(v) for k,v in row}) for row in encoded]


def digest(rows):return hashlib.sha256(json.dumps(dump(rows),separators=(',',':')).encode()).hexdigest()


def evaluate(rows,point):return [row.at(point) for row in rows]

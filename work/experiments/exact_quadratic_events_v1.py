"""308 exact degree<=2 event arithmetic. No floating root ordering."""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key
import math


class WorkLimit(RuntimeError):pass


@dataclass
class Budget:
    limit:int=1000000
    comparisons:int=0
    def use(self):
        if self.comparisons>=self.limit:raise WorkLimit('polynomial/root comparison cap')
        self.comparisons+=1


def sign(x):return (x>0)-(x<0)


class Poly:
    """Rational coefficients in ascending order; affine values have c[2]==0."""
    __slots__=('c',)
    def __init__(self,value=0):
        if isinstance(value,Poly):self.c=value.c
        elif isinstance(value,(tuple,list)):
            assert len(value)<=3
            self.c=tuple(F(x) for x in value)+(F(0),)*(3-len(value))
        else:self.c=(F(value),F(0),F(0))
    def __add__(self,other):return Poly(tuple(a+b for a,b in zip(self.c,Poly(other).c)))
    __radd__=__add__
    def __neg__(self):return Poly(tuple(-x for x in self.c))
    def __sub__(self,other):return self+-Poly(other)
    def __rsub__(self,other):return Poly(other)+-self
    def __mul__(self,other):
        b=Poly(other).c;out=[F(0)]*5
        for i,a in enumerate(self.c):
            for j,z in enumerate(b):out[i+j]+=a*z
        assert not any(out[3:]),'An operation exceeded the proved quadratic degree'
        return Poly(out[:3])
    __rmul__=__mul__
    def __truediv__(self,other):
        other=F(other);assert other
        return Poly(tuple(x/other for x in self.c))
    def square(self):return self*self
    def at(self,q):
        q=F(q);return self.c[0]+q*(self.c[1]+q*self.c[2])
    def affine(self):assert self.c[2]==0;return self
    def __repr__(self):return f'Poly({self.c!r})'


class Root:
    """One irrational root of a canonical irreducible integer quadratic.

    Rational isolating bounds are refined exactly. Distinct canonical
    irreducible quadratics cannot have a common irrational root.
    """
    __slots__=('poly','index','lo','hi')
    def __init__(self,poly,index):
        a,b,c=poly;assert a>0 and index in [0,1]
        disc=b*b-4*a*c;assert disc>0 and math.isqrt(disc)**2!=disc
        self.poly=tuple(poly);self.index=index
        radius=1+max(abs(F(b,a)),abs(F(c,a)));vertex=-F(b,2*a)
        self.lo,self.hi=(-radius,vertex) if index==0 else (vertex,radius)
    def __hash__(self):return hash((self.poly,self.index))
    def __eq__(self,other):return isinstance(other,Root) and self.poly==other.poly and self.index==other.index
    def versus_rational(self,q,budget=None):
        if budget:budget.use()
        q=F(q);a,b,c=self.poly;vertex=-F(b,2*a)
        value=(a*q+b)*q+c
        assert value!=0,'An irrational root cannot equal a rational point'
        if self.index==0:
            return -1 if q>=vertex else sign(value)
        return 1 if q<=vertex else -sign(value)
    def refine(self,budget=None):
        mid=(self.lo+self.hi)/2
        if self.versus_rational(mid,budget)>0:self.lo=mid
        else:self.hi=mid
    def spec(self):return dict(polynomial=list(self.poly),root_index=self.index)
    def __repr__(self):return f'Root({self.poly!r}, {self.index})'


def compare(a,b,budget=None):
    if budget:budget.use()
    if not isinstance(a,Root) and not isinstance(b,Root):return sign(F(a)-F(b))
    if not isinstance(a,Root):return -b.versus_rational(a,budget)
    if not isinstance(b,Root):return a.versus_rational(b,budget)
    if a.poly==b.poly:return sign(a.index-b.index)
    while True:
        if a.hi<=b.lo:return -1
        if b.hi<=a.lo:return 1
        if a.hi-a.lo>=b.hi-b.lo:a.refine(budget)
        else:b.refine(budget)


def between(a,b,budget=None):
    assert compare(a,b,budget)<0
    while True:
        left=a.hi if isinstance(a,Root) else F(a)
        right=b.lo if isinstance(b,Root) else F(b)
        if left<right:
            q=(left+right)/2
            assert compare(a,q,budget)<0 and compare(q,b,budget)<0
            return q
        if isinstance(a,Root):a.refine(budget)
        if isinstance(b,Root):b.refine(budget)


def sign_at(poly,point,budget=None):
    if budget:budget.use()
    p=Poly(poly)
    if not isinstance(point,Root):return sign(p.at(point))
    a,b,c=point.poly
    constant=p.c[0]-p.c[2]*F(c,a);linear=p.c[1]-p.c[2]*F(b,a)
    if not linear:return sign(constant)
    return sign(linear)*compare(point,-constant/linear,budget)


def roots(poly,cache=None,budget=None):
    if budget:budget.use()
    c,b,a=Poly(poly).c
    if not a:return [] if not b else [-c/b]
    denominator=math.lcm(a.denominator,b.denominator,c.denominator)
    aa,bb,cc=[int(x*denominator) for x in [a,b,c]]
    divisor=math.gcd(math.gcd(abs(aa),abs(bb)),abs(cc))
    aa,bb,cc=[x//divisor for x in [aa,bb,cc]]
    if aa<0:aa,bb,cc=-aa,-bb,-cc
    key=(aa,bb,cc)
    if cache is not None and key in cache:return cache[key]
    disc=bb*bb-4*aa*cc
    if disc<0:result=[]
    elif not disc:result=[-F(bb,2*aa)]
    elif math.isqrt(disc)**2==disc:
        s=math.isqrt(disc);result=[F(-bb-s,2*aa),F(-bb+s,2*aa)]
    else:result=[Root(key,0),Root(key,1)]
    if cache is not None:cache[key]=result
    return result


def sorted_points(points,budget=None):
    return sorted(set(points),key=cmp_to_key(lambda a,b:compare(a,b,budget)))


def encode(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,Root):return value.spec()
    if isinstance(value,Poly):return [str(x) for x in value.c]
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value

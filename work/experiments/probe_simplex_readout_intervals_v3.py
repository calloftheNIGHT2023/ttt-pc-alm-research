"""Exact affine-envelope bounds with conservative 64-bit dyadic endpoints.

Scalar lines are certified at every endpoint and tent knot. Composition uses
the correct lower/upper affine form according to line slope. Each selected
line is saved for independent replay; choosing a line is not itself a proof.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import heapq
from itertools import combinations
import time
from probe_simplex_readout_intervals import enclosure as range_enclosure, tent, split_vertices, longest_edge


def average(form,centroid):
    return form[4]+sum((a*b for a,b in zip(form[:4],centroid)),F(0))


def forms_for_interval(low,high):
    assert low<=high
    knots=sorted({low,high}|{t for t in [F(0),F(1,2),F(1)] if low<=t<=high})
    lines={(F(0),F(0)),(F(0),F(1))}
    if low==high:lines.add((F(0),tent(low)))
    for x,y in combinations(knots,2):
        slope=(tent(y)-tent(x))/(y-x)
        lines.add((slope,tent(x)-slope*x))
    lower=[line for line in sorted(lines) if all(line[0]*t+line[1]<=tent(t) for t in knots)]
    upper=[line for line in sorted(lines) if all(line[0]*t+line[1]>=tent(t) for t in knots)]
    assert lower and upper
    return lower,upper


def compose(line,low_form,high_form,lower):
    slope,offset=line
    form=low_form if (slope>=0)==lower else high_form
    ans=[slope*t for t in form];ans[4]+=offset
    return tuple(ans)


def enclosure(q,vertices,layers=4):
    old_low,old_high,affine=range_enclosure(q,vertices,layers)
    if affine:return old_low,old_high,True,[]
    centroid=tuple(sum((v[j] for v in vertices),F(0))/5 for j in range(4))
    low_form=high_form=(F(0),F(0),F(0),F(0),q)
    certificate=[]
    for layer in range(layers):
        left=list(low_form);right=list(high_form);left[layer]+=1;right[layer]+=1
        left,right=tuple(left),tuple(right)
        z_low=min(left[4]+sum((a*b for a,b in zip(left[:4],v)),F(0)) for v in vertices)
        z_high=max(right[4]+sum((a*b for a,b in zip(right[:4],v)),F(0)) for v in vertices)
        lower_lines,upper_lines=forms_for_interval(z_low,z_high)
        low_line=max(lower_lines,key=lambda line:average(compose(line,left,right,True),centroid))
        high_line=min(upper_lines,key=lambda line:average(compose(line,left,right,False),centroid))
        low_form=compose(low_line,left,right,True)
        high_form=compose(high_line,left,right,False)
        certificate.append(dict(exact_z_lower=str(z_low),exact_z_upper=str(z_high),
            lower_line=[str(t) for t in low_line],upper_line=[str(t) for t in high_line]))
    lower=max(F(0),old_low,average(low_form,centroid))
    upper=min(F(1),old_high,average(high_form,centroid))
    assert lower<=upper
    # Outward rounding keeps every enclosure valid and prevents denominator
    # growth across different supporting-line slopes. Affine exact means above
    # are preserved. This is not a tolerance change or an inward rounding.
    denominator=1<<64
    lower=F((lower.numerator*denominator)//lower.denominator,denominator)
    upper=F((upper.numerator*denominator+upper.denominator-1)//upper.denominator,denominator)
    assert lower<=upper
    return lower,upper,False,certificate


@dataclass(frozen=True)
class Node:
    root:int
    path:str
    vertices:tuple
    mass:F
    low:F
    high:F
    affine:bool
    envelope:list

    @property
    def contribution(self):
        a,b=self.mass*self.low,self.mass*self.high
        return min(a,b),max(a,b)

    @property
    def uncertainty(self):
        return abs(self.mass)*(self.high-self.low)


def integrate(q,roots,coefficients,*,layers=4,target=None,max_width=None,max_splits=4096):
    q=F(q);roots=[tuple(tuple(F(x) for x in v) for v in root) for root in roots]
    coefficients=[F(c) for c in coefficients]
    assert len(roots)==len(coefficients) and roots
    assert all(len(root)==5 and all(len(v)==4 for v in root) for root in roots)
    assert 1<=layers<=4 and type(max_splits) is int and max_splits>=0
    assert target is not None or max_width is not None
    target=None if target is None else F(target);max_width=None if max_width is None else F(max_width)
    assert target is None or target>=0
    assert max_width is None or max_width>=0
    started=time.perf_counter();leaves={};heap=[];operations=[];low=F(0);high=F(0)

    def add(index,path,vertices,mass):
        nonlocal low,high
        a,b,affine,certificate=enclosure(q,vertices,layers)
        node=Node(index,path,vertices,mass,a,b,affine,certificate)
        leaves[(index,path)]=node;left,right=node.contribution;low+=left;high+=right
        if node.uncertainty:heapq.heappush(heap,(-node.uncertainty,index,path))

    for index,(vertices,mass) in enumerate(zip(roots,coefficients)):
        if mass:add(index,'',vertices,mass)
    while True:
        assert low<=high
        if target is not None and max(abs(low),abs(high))<=target:status='upper_bound_met';break
        if target is not None and (low>target or high<-target):status='lower_bound_exceeds_target';break
        if max_width is not None and high-low<=max_width:status='width_met';break
        if not heap:status='exact_integral';break
        if len(operations)>=max_splits:status='work_limit_unresolved';break
        _,index,path=heapq.heappop(heap);node=leaves.pop((index,path))
        a,b=node.contribution;low-=a;high-=b
        edge=longest_edge(node.vertices,layers);left,right=split_vertices(node.vertices,edge)
        operations.append(dict(root=index,path=path,edge=list(edge)))
        add(index,path+'0',left,node.mass/2);add(index,path+'1',right,node.mass/2)
    ledger=[dict(root=index,path=path,mass=str(node.mass),low=str(node.low),high=str(node.high),
        affine=node.affine,envelope=node.envelope) for (index,path),node in sorted(leaves.items())]
    assert sum((node.mass for node in leaves.values()),F(0))==sum(coefficients,F(0))
    return dict(enclosure_version=3,rounding_bits=64,query_coordinate=str(q),layers=layers,coefficients=[str(c) for c in coefficients],
        target=None if target is None else str(target),max_width=None if max_width is None else str(max_width),
        max_splits=max_splits,status=status,exact_lower=str(low),exact_upper=str(high),lower=float(low),upper=float(high),
        absolute_upper_bound=float(max(abs(low),abs(high))),exact_absolute_upper_bound=str(max(abs(low),abs(high))),
        split_operations=operations,leaves=ledger,splits=len(operations),leaf_count=len(ledger),
        affine_leaves=sum(node.affine for node in leaves.values()),seconds=time.perf_counter()-started,
        scope='Rational affine-envelope nominal signed integral enclosure; every scalar supporting line is explicit')

"""Rational enclosures of signed uniform-simplex tent-readout integrals.

Midpoint subdivision is performed in the abstract four-dimensional simplex;
each child has exactly half its parent's probability, even when its physical
bias image is degenerate. All arithmetic establishing bounds is rational.
No teacher/query labels, approximation of a bound by floats, or gate waiver.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import heapq
from itertools import combinations
import time


def tent(z):
    if z <= 0 or z >= 1:
        return F(0)
    return 2*z if z <= F(1,2) else 2-2*z


def range_image(lo, hi):
    assert lo <= hi
    a, b = tent(lo), tent(hi)
    return min(a,b), (F(1) if lo <= F(1,2) <= hi else max(a,b))


def enclosure(q, vertices, layers=4):
    """Exact affine mean, otherwise a conservative range enclosure."""
    assert 1 <= layers <= 4 and len(vertices) == 5
    assert all(len(v) == 4 for v in vertices)
    values = [q]*5
    for layer in range(layers):
        z = [h+v[layer] for h,v in zip(values,vertices)]
        lo, hi = min(z), max(z)
        if hi <= 0 or lo >= 1:
            values = [F(0)]*5
        elif lo >= 0 and hi <= F(1,2):
            values = [2*t for t in z]
        elif lo >= F(1,2) and hi <= 1:
            values = [2-2*t for t in z]
        else:
            low, high = range_image(lo,hi)
            for later in range(layer+1,layers):
                low, high = range_image(low+min(v[later] for v in vertices),
                                        high+max(v[later] for v in vertices))
            return low, high, False
    value = sum(values,F(0))/5
    assert 0 <= value <= 1
    return value,value,True


def split_vertices(vertices, edge):
    i,j = edge
    assert 0 <= i < j < 5
    midpoint = tuple((a+b)/2 for a,b in zip(vertices[i],vertices[j]))
    left, right = list(vertices),list(vertices)
    left[i] = midpoint
    right[j] = midpoint
    return tuple(left),tuple(right)


def longest_edge(vertices,layers=4):
    edges = list(combinations(range(5),2))
    def length(edge):
        i,j = edge
        return sum((F(2**(layers-k))*abs(vertices[i][k]-vertices[j][k])
                    for k in range(layers)),F(0))
    edge = max(edges,key=length)
    assert length(edge)>0, 'An unresolved readout must vary on at least one active coordinate'
    return edge


@dataclass(frozen=True)
class Node:
    root: int
    path: str
    vertices: tuple
    mass: F
    low: F
    high: F
    affine: bool

    @property
    def contribution(self):
        a,b = self.mass*self.low,self.mass*self.high
        return min(a,b),max(a,b)

    @property
    def uncertainty(self):
        return abs(self.mass)*(self.high-self.low)


def integrate(q, roots, coefficients, *, layers=4, target=None, max_width=None, max_splits=4096):
    q = F(q)
    roots = [tuple(tuple(F(x) for x in v) for v in root) for root in roots]
    coefficients = [F(c) for c in coefficients]
    assert len(roots) == len(coefficients) and roots
    assert all(len(root)==5 and all(len(v)==4 for v in root) for root in roots)
    assert 1 <= layers <= 4 and type(max_splits) is int and max_splits>=0
    assert target is not None or max_width is not None
    target = None if target is None else F(target)
    max_width = None if max_width is None else F(max_width)
    assert target is None or target>=0
    assert max_width is None or max_width>=0
    started = time.perf_counter()
    leaves,heap,operations = {},[],[]
    low,high = F(0),F(0)

    def add(root,path,vertices,mass):
        nonlocal low,high
        a,b,affine = enclosure(q,vertices,layers)
        node = Node(root,path,vertices,mass,a,b,affine)
        leaves[(root,path)] = node
        lo,hi = node.contribution
        low += lo
        high += hi
        if node.uncertainty:
            heapq.heappush(heap,(-node.uncertainty,root,path))

    for index,(vertices,mass) in enumerate(zip(roots,coefficients)):
        if mass:
            add(index,'',vertices,mass)
    status = None
    while True:
        assert low<=high
        if target is not None and max(abs(low),abs(high))<=target:
            status='upper_bound_met'
            break
        if target is not None and (low>target or high<-target):
            status='lower_bound_exceeds_target'
            break
        if max_width is not None and high-low<=max_width:
            status='width_met'
            break
        if not heap:
            status='exact_integral'
            break
        if len(operations)>=max_splits:
            status='work_limit_unresolved'
            break
        _,index,path = heapq.heappop(heap)
        node = leaves.pop((index,path))
        a,b = node.contribution
        low -= a
        high -= b
        edge = longest_edge(node.vertices,layers)
        left,right = split_vertices(node.vertices,edge)
        operations.append(dict(root=index,path=path,edge=list(edge)))
        add(index,path+'0',left,node.mass/2)
        add(index,path+'1',right,node.mass/2)
    ledger=[]
    for (index,path),node in sorted(leaves.items()):
        ledger.append(dict(root=index,path=path,mass=str(node.mass),low=str(node.low),
                           high=str(node.high),affine=node.affine))
    assert sum((node.mass for node in leaves.values()),F(0)) == sum(coefficients,F(0))
    return dict(query_coordinate=str(q),layers=layers,
        coefficients=[str(c) for c in coefficients],
        target=None if target is None else str(target),
        max_width=None if max_width is None else str(max_width),max_splits=max_splits,
        status=status,exact_lower=str(low),exact_upper=str(high),
        lower=float(low),upper=float(high),absolute_upper_bound=float(max(abs(low),abs(high))),
        exact_absolute_upper_bound=str(max(abs(low),abs(high))),
        split_operations=operations,leaves=ledger,splits=len(operations),
        leaf_count=len(ledger),affine_leaves=sum(node.affine for node in leaves.values()),
        seconds=time.perf_counter()-started,
        scope='Exact nominal uniform-barycentric signed integral enclosure, not finite-RNG quality')

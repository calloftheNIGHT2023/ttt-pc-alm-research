"""Exact-in-real-arithmetic one-line conditional moments for tent memories.

Uniform bounded convex parameter cell; direction is fixed per cell, not chosen
from the sampled point or query answer. This is a shared geometric primitive,
not a PC-specific method. Floating arithmetic is not an interval certificate.
"""
import numpy as np

KNOTS = (0., .5, 1.)
SLOPES = (0., 2., -2., 0.)
OFFSETS = (0., 0., 2., 0.)


def fiber_interval(matrix, rhs, point, direction):
    """Intersect point+t*direction with every cell inequality A b <= rhs."""
    matrix=np.asarray(matrix,dtype=float);rhs=np.asarray(rhs,dtype=float)
    point=np.asarray(point,dtype=float);direction=np.asarray(direction,dtype=float)
    if matrix.ndim!=2 or matrix.shape[1]!=len(point) or direction.shape!=point.shape:
        raise ValueError('Invalid matrix, point or direction shape')
    if not np.all(np.isfinite(direction)) or not np.any(direction):
        raise ValueError('A finite nonzero direction is required')
    slack=rhs-matrix@point;rate=matrix@direction
    if np.min(slack)<-1e-10:
        raise ValueError('The point is outside its cell')
    if np.any((rate==0)&(slack<0)):
        raise ValueError('Floating zero-rate constraint is violated')
    positive=rate>0;negative=rate<0
    lo=float(np.max(slack[negative]/rate[negative])) if np.any(negative) else -np.inf
    hi=float(np.min(slack[positive]/rate[positive])) if np.any(positive) else np.inf
    if not np.isfinite(lo+hi) or not lo<hi:
        raise ValueError('A finite positive-length conditional fiber is required')
    return lo,hi


def line_moments(query,point,direction,lo,hi):
    """First/second moments over a uniform t in [lo,hi], one query input.

    Propagates every breakpoint through the complete network. The number of
    pieces can grow with depth: no claim of local or depth-independent cost.
    Endpoint formulas avoid subtracting nearly equal polynomial primitives.
    """
    if not np.isfinite(lo+hi) or not lo<hi:
        raise ValueError('Positive finite interval required')
    pieces=[(float(lo),float(hi),0.,float(query))]
    for bias,rate in zip(point,direction):
        following=[]
        for left,right,slope,offset in pieces:
            s=slope+rate;c=offset+bias
            edges=[left,right]
            if s:
                edges.extend(root for knot in KNOTS if left<(root:=(knot-c)/s)<right)
            edges=sorted(set(edges))
            for start,end in zip(edges[:-1],edges[1:]):
                mid=(start+end)/2
                code=sum(s*mid+c>=k for k in KNOTS)
                following.append((start,end,SLOPES[code]*s,SLOPES[code]*c+OFFSETS[code]))
        pieces=following
    total=0.;second=0.
    for left,right,slope,offset in pieces:
        y0=slope*left+offset;y1=slope*right+offset;length=right-left
        total+=length*(y0+y1)/2
        second+=length*(y0*y0+y0*y1+y1*y1)/3
    return total/(hi-lo),second/(hi-lo),len(pieces)


def conditional_moments(queries,point,direction,matrix,rhs):
    lo,hi=fiber_interval(matrix,rhs,point,direction)
    first=[];second=[];pieces=0
    for query in queries:
        a,b,n=line_moments(query,point,direction,lo,hi)
        first.append(a);second.append(b);pieces+=n
    return np.array(first),np.array(second),dict(lo=lo,hi=hi,total_pieces=pieces)

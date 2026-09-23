"""LP-free enclosing proposal boxes for affine support cells.

Component selection proportional to proposal volume, followed by full cell
rejection, samples the uniform union of interior-disjoint cells. Unknown cell
volumes are never required. Numerical linear algebra is not interval proof.
"""
import math
from fractions import Fraction as F
import numpy as np


def pivot_basis(matrix):
    """Exact complete-pivot elimination of the given binary64 coefficients."""
    n,d=matrix.shape;a=[[F(float(v)) for v in row] for row in matrix]
    rows=list(range(n));cols=list(range(d));rank=0
    for k in range(min(n,d)):
        value,i,j=max((abs(a[i][j]),i,j) for i in range(k,n) for j in range(k,d))
        if not value:break
        a[k],a[i]=a[i],a[k];rows[k],rows[i]=rows[i],rows[k]
        for row in a:row[k],row[j]=row[j],row[k]
        cols[k],cols[j]=cols[j],cols[k]
        for i in range(k+1,n):
            ratio=a[i][k]/a[k][k]
            for j in range(k,d):a[i][j]-=ratio*a[k][j]
        rank+=1
    return rows[:rank],sorted(cols[rank:])


def determinant_exact(matrix):
    a=[[F(float(v)) for v in row] for row in matrix];d=len(a);answer=F(1)
    for j in range(d):
        k=next((k for k in range(j,d) if a[k][j]),None)
        if k is None:return F(0)
        if k!=j:a[j],a[k]=a[k],a[j];answer=-answer
        pivot=a[j][j];answer*=pivot
        for i in range(j+1,d):
            ratio=a[i][j]/pivot
            for k in range(j+1,d):a[i][k]-=ratio*a[j][k]
    return answer


def make_box(output,offset,observations,eps=.001,bound=.12):
    """Use independent observed-output rows and free original coordinates."""
    output=np.asarray(output,dtype=float);offset=np.asarray(offset,dtype=float);observations=np.asarray(observations,dtype=float)
    n,d=output.shape
    if eps<=0 or bound<=0 or offset.shape!=(n,) or observations.shape!=(n,):raise ValueError('Invalid observation box')
    selected,free=pivot_basis(output);rank=len(selected)
    transform=np.vstack([output[selected],np.eye(d)[free]])
    determinant=determinant_exact(transform)
    if not determinant:raise ValueError('Singular proposal transform')
    center=np.r_[observations[selected]-offset[selected],np.zeros(d-rank)]
    radius=np.r_[np.full(rank,eps),np.full(d-rank,bound)]
    inverse=np.linalg.inv(transform)
    volume=float(np.prod(2*radius))/abs(float(determinant))
    if not math.isfinite(volume) or volume<=0:raise ValueError('Nonpositive or underflowed proposal volume')
    return dict(transform=transform,inverse=inverse,center=center,radius=radius,volume=volume,
                determinant=str(determinant),selected_rows=selected,free_coordinates=free,rank=rank)


def draw(box,count,rng):
    transformed=box['center']+box['radius']*rng.uniform(-1,1,(count,len(box['center'])))
    return transformed@box['inverse'].T


def contained_in_proposal(box,points,tolerance=0.):
    return np.all(np.abs(points@box['transform'].T-box['center'])<=box['radius']+tolerance,axis=1)


def attempts(boxes,matrices,right_sides,count,rng,bound=.12):
    """Return all accepted draws in attempt order; no fixed-count truncation."""
    if not boxes:raise ValueError('At least one proposal is needed')
    volumes=np.array([box['volume'] for box in boxes]);probability=volumes/volumes.sum()
    choices=rng.choice(len(boxes),size=count,p=probability);d=len(boxes[0]['center'])
    points=np.empty((count,d));accepted=np.zeros(count,bool)
    for k,box in enumerate(boxes):
        indices=np.flatnonzero(choices==k)
        if not len(indices):continue
        bank=draw(box,len(indices),rng);points[indices]=bank
        accepted[indices]=np.all(np.abs(bank)<=bound,axis=1)&np.all(bank@matrices[k].T<=right_sides[k],axis=1)
    return points[accepted],choices[accepted],dict(proposals=count,accepted=int(accepted.sum()),total_proposal_volume=float(volumes.sum()))


def verify():
    rng=np.random.default_rng(172914);basis_cases=0;enclosure_checks=0
    for d in [2,3,4,6]:
        for r in range(d+1):
            rows=rng.integers(-4,5,(r,d)).astype(float)
            while r and np.linalg.matrix_rank(rows)!=r:rows=rng.integers(-4,5,(r,d)).astype(float)
            matrix=np.vstack([rows,rows[:1]]) if r else np.empty((0,d))
            b=rng.uniform(-.1,.1,d);offset=np.zeros(len(matrix));observations=matrix@b
            box=make_box(matrix,offset,observations)
            assert box['rank']==r and len(box['free_coordinates'])==d-r
            assert np.max(np.abs(box['transform']@box['inverse']-np.eye(d)))<1e-10
            assert contained_in_proposal(box,b[None,:],1e-13)[0];enclosure_checks+=1
            sample=draw(box,128,rng);assert np.all(contained_in_proposal(box,sample,1e-10));enclosure_checks+=len(sample);basis_cases+=1
    # Two disjoint rectangles with proposal volumes 1 and 1/2, but valid
    # volumes 1/2 and 1/8. No valid-volume weights enter the sampler.
    one=make_box(np.array([[1.,0.]]),np.zeros(1),np.array([-.5]),eps=.25,bound=1.)
    two=make_box(np.array([[2.,0.]]),np.zeros(1),np.array([1.]),eps=.25,bound=1.)
    aa=[np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])]
    rr=[np.array([-.25,.75,.5,.5]),np.array([.625,-.375,.25,.25])]
    points,labels,meta=attempts([one,two],aa,rr,40000,rng,bound=1.)
    frequency=float(np.mean(labels==0));acceptance=meta['accepted']/meta['proposals']
    assert abs(frequency-.8)<.02 and abs(acceptance-5/12)<.015
    assert abs(one['volume']-1)<1e-14 and abs(two['volume']-.5)<1e-14
    # Adding a certified empty full-prior box changes only rejection rate.
    empty=make_box(np.empty((0,2)),np.empty(0),np.empty(0),eps=.25,bound=1.)
    pp,ll,mm=attempts([one,two,empty],aa+[np.array([[-1.,0.]])],rr+[np.array([-2.])],40000,rng,bound=1.)
    frequency_with_empty=float(np.mean(ll==0));assert abs(frequency_with_empty-.8)<.03
    assert abs(mm['accepted']/40000-5/44)<.01
    assert F(11,2)/F(3,2)==F(11,3)
    return dict(passed=True,exact_basis_cases=basis_cases,proposal_enclosure_checks=enclosure_checks,
                empirical_region_one_fraction=frequency,expected_region_one_fraction=.8,
                empirical_acceptance=acceptance,expected_acceptance=5/12,
                empirical_region_fraction_with_empty=frequency_with_empty,
                empirical_acceptance_with_empty=mm['accepted']/40000,expected_acceptance_with_empty=5/44,
                exact_expected_proposal_speed_ratio_after_empty_removal='11/3',
                scope='generic sampling primitive and exact volume identities, not PC-specific task or wall-time advantage')

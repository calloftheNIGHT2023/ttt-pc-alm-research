"""Shared vectorized exact piecewise line integration, not local learning.

Each (particle, query) pair has a separate affine line in unit coordinate t.
Splitting is exact in real arithmetic; floating output is not a certificate.
"""
import numpy as np
import conditional_fiber_readout as reference


def intervals(matrix,rhs,points,direction):
    rate=matrix@direction;slack=rhs[None,:]-points@matrix.T
    if np.min(slack)<-1e-10:raise ValueError('Sample outside cell')
    zero=rate==0
    if np.any(slack[:,zero]<0):raise ValueError('Violated zero-rate constraint')
    lo=np.max(slack[:,rate<0]/rate[rate<0],axis=1)
    hi=np.min(slack[:,rate>0]/rate[rate>0],axis=1)
    if not np.all(np.isfinite(lo+hi)) or np.any(hi<=lo):raise ValueError('Invalid conditional line')
    return lo,hi


def moments(queries,points,direction,lo,hi):
    """One bounded particle/query block; caller controls the block size."""
    nq=len(queries);nb=len(points);size=nq*nb
    ids=np.arange(size);owners=np.repeat(np.arange(nb),nq)
    left=np.zeros(size);right=np.ones(size);slope=np.zeros(size);offset=np.tile(queries,nb)
    origin=points+lo[:,None]*direction;delta=(hi-lo)[:,None]*direction
    slopes=np.array(reference.SLOPES);offsets=np.array(reference.OFFSETS)
    max_pieces=size
    for layer in range(points.shape[1]):
        zs=slope+delta[owners,layer];zc=offset+origin[owners,layer]
        roots=np.broadcast_to(right[:,None],(len(right),3)).copy()
        nz=zs!=0
        roots[nz]=(np.array(reference.KNOTS)[None,:]-zc[nz,None])/zs[nz,None]
        roots=np.sort(np.clip(roots,left[:,None],right[:,None]),axis=1)
        edges=np.c_[left,roots,right]
        ll=edges[:,:-1].ravel();rr=edges[:,1:].ravel();mask=rr>ll
        parent=np.repeat(np.arange(len(left)),4)[mask];left=ll[mask];right=rr[mask]
        mids=(left+right)/2;values=zs[parent]*mids+zc[parent]
        codes=np.searchsorted(reference.KNOTS,values,side='right')
        slope=slopes[codes]*zs[parent];offset=slopes[codes]*zc[parent]+offsets[codes]
        ids=ids[parent];owners=owners[parent];max_pieces=max(max_pieces,len(ids))
    y0=slope*left+offset;y1=slope*right+offset;length=right-left
    mean=np.bincount(ids,weights=length*(y0+y1)/2,minlength=size)
    second=np.bincount(ids,weights=length*(y0*y0+y0*y1+y1*y1)/3,minlength=size)
    return mean.reshape(nb,nq),second.reshape(nb,nq),dict(final_pieces=len(ids),max_pieces=max_pieces)


def verify():
    rng=np.random.default_rng(921376);checks=0;max_error=0.;max_pieces=0
    for depth in [1,2,4,6]:
        points=rng.uniform(-.08,.08,(7,depth));direction=rng.normal(size=depth);direction/=np.linalg.norm(direction)
        a=np.r_[np.eye(depth),-np.eye(depth)];rhs=np.full(2*depth,.12);q=np.linspace(0,1,13)
        lo,hi=intervals(a,rhs,points,direction);mu,second,meta=moments(q,points,direction,lo,hi)
        max_pieces=max(max_pieces,meta['max_pieces'])
        for i,point in enumerate(points):
            rlo,rhi=reference.fiber_interval(a,rhs,point,direction)
            assert abs(lo[i]-rlo)<1e-14 and abs(hi[i]-rhi)<1e-14
            for j,query in enumerate(q):
                first2,second2,_=reference.line_moments(query,point,direction,rlo,rhi)
                error=max(abs(mu[i,j]-first2),abs(second[i,j]-second2))
                assert error<1e-11;checks+=1;max_error=max(max_error,error)
    return dict(passed=True,scalar_reference_pairs=checks,maximum_moment_error=max_error,maximum_pieces_in_test_block=max_pieces)

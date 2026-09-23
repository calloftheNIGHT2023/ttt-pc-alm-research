"""Support-only proposal channels. These points NEVER pass a state cut gate.

Free tied pieces use adjacent-layer data and return floating-point proposals.
Coordinate scans are a stronger simple control using the full forward map.
Neither channel certifies feasibility: downstream mode geometry must do that.
"""
from fractions import Fraction as F
import numpy as np
import conflict_local_repair as common

base=common.base
B=float(common.BOUND)
TAU=float(common.TRUST)


def forward_many(x,b):
    prev=np.broadcast_to(x,(len(b),len(x)));codes=[];activities=[]
    for j in range(b.shape[1]):
        z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8))
        prev=base.g(z);activities.append(prev)
    return np.array(codes,dtype=np.uint8).transpose(1,0,2),np.array(activities).transpose(1,0,2)


def finite_values(optimum,lo,hi,prior):
    t=float(np.clip(optimum,lo,hi));values=[t]
    for direction in [-np.inf,np.inf]:
        nxt=float(np.nextafter(t,direction))
        if lo<=nxt<=hi:values.append(nxt)
    values.extend([float((lo+hi)/2),float(lo),float(hi)])
    if lo<=prior<=hi:values.append(float(prior))
    return list(dict.fromkeys(values))


def tied_free(x,b,h,u):
    """Floating free-piece minima, adjacent doubles, midpoint and endpoints."""
    d,n=h.shape;bs=[];hs=[];labels=[];seen=set();segments=0
    for j in range(d-1):
        prev=x if j==0 else h[j-1];neighbor=b[j+1]
        events=[-B,B]+(base.KNOTS[:,None]-prev).ravel().tolist()
        for knot in base.KNOTS:
            y=knot-neighbor
            if 0<=y<=1:events.extend((y/2-prev).tolist());events.extend((1-y/2-prev).tolist())
        events=np.unique([v for v in events if -B<=v<=B]);assert len(events)-1<=9*n+1
        for k,(lo,hi) in enumerate(zip(events[:-1],events[1:])):
            mid=(lo+hi)/2;first=np.searchsorted(base.KNOTS,prev+mid,side='right');s=base.SLOPES[first];q=s*prev+base.INTERCEPTS[first]
            second=np.searchsorted(base.KNOTS,s*mid+q+neighbor,side='right');m=base.SLOPES[second]*s;offset=base.SLOPES[second]*(q+neighbor)+base.INTERCEPTS[second]
            quad=np.sum(m*m)+TAU*(np.sum(s*s)+n)
            linear=np.sum(m*(h[j+1]+u[j+1]-offset))+TAU*np.sum(s*(h[j]-q))+TAU*n*b[j]
            segments+=1
            for t in finite_values(linear/quad,lo,hi,b[j]):
                token=j,t
                if token in seen:continue
                seen.add(token);bb=b.copy();hh=h.copy();bb[j]=t;hh[j]=base.g(prev+t)
                bs.append(bb);hs.append(hh);labels.append((j,k,t))
    return np.array(bs).reshape(-1,d),np.array(hs).reshape(-1,d,n),dict(segments=segments,labels=labels,candidates=len(bs),
        scope='proposal only; floating free-piece calculation; no state certification')


def coordinate_intervals(x,b,coordinate):
    """Propagate a piecewise-affine one-dimensional full-forward map."""
    states=[(-B,B,np.zeros(len(x)),np.array(x),[])]
    for j in range(len(b)):
        after=[]
        for lo,hi,slope,offset,history in states:
            zs=slope+(1 if j==coordinate else 0);zc=offset+(0 if j==coordinate else b[j])
            nonzero=zs!=0;events=[lo,hi]
            if np.any(nonzero):
                roots=((base.KNOTS[:,None]-zc[nonzero])/zs[nonzero]).ravel()
                events.extend(roots[(roots>lo)&(roots<hi)].tolist())
            events=np.unique(events)
            for left,right in zip(events[:-1],events[1:]):
                mid=(left+right)/2;reg=np.searchsorted(base.KNOTS,zs*mid+zc,side='right').astype(np.uint8)
                after.append((left,right,base.SLOPES[reg]*zs,base.SLOPES[reg]*zc+base.INTERCEPTS[reg],history+[reg]))
        states=after
    return states


def coordinate_scan(x,b):
    """Shared simple baseline: every bias, every propagated interval."""
    bs=[];labels=[];seen=set();segments=0
    for j in range(len(b)):
        intervals=coordinate_intervals(x,b,j);segments+=len(intervals)
        for k,(lo,hi,_,_,_) in enumerate(intervals):
            for t in finite_values((lo+hi)/2,lo,hi,b[j]):
                token=j,t
                if token in seen:continue
                seen.add(token);bb=b.copy();bb[j]=t;bs.append(bb);labels.append((j,k,t))
    return np.array(bs).reshape(-1,len(b)),dict(segments=segments,labels=labels,candidates=len(bs),
        scope='floating interval propagation; dense probes verify coverage, not a floating completeness proof')


def coordinate_intervals_exact(x,b,coordinate):
    """Independent rational reference, for primitive checks only."""
    bound=F(B);xx=[F(float(t)) for t in x];bb=[F(float(t)) for t in b];knots=[F(0),F(1,2),F(1)]
    states=[(-bound,bound,[F(0)]*len(xx),xx,[])]
    for j in range(len(bb)):
        after=[]
        for lo,hi,slope,offset,history in states:
            zs=[s+int(j==coordinate) for s in slope];zc=[c+(F(0) if j==coordinate else bb[j]) for c in offset];events=[lo,hi]
            for s,c in zip(zs,zc):
                if s:events.extend((k-c)/s for k in knots if lo<(k-c)/s<hi)
            events=sorted(set(events))
            for left,right in zip(events[:-1],events[1:]):
                mid=(left+right)/2;reg=[sum(s*mid+c>=k for k in knots) for s,c in zip(zs,zc)]
                after.append((left,right,[F(int(base.SLOPES[r]))*s for r,s in zip(reg,zs)],
                    [F(int(base.SLOPES[r]))*c+F(int(base.INTERCEPTS[r])) for r,c in zip(reg,zc)],history+[reg]))
        states=after
    return states


def verify():
    import tied_local_block_repair as reference
    rng=np.random.default_rng(632917);formula_checks=0;fiber_points=0;probe_checks=0;near_boundaries=0;cases=0;max_objective_error=0.
    for d,n in [(2,2),(4,4),(4,8)]:
        for _ in range(3):
            x=rng.uniform(0,1,n);b=rng.uniform(-B,B,d);h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n))
            bb,hh,meta=tied_free(x,b,h,u);fp,full=forward_many(x,bb)
            assert np.all((hh>=0)&(hh<=1)) and np.all(np.abs(bb)<=B) and np.all(hh[:,-1]==h[-1])
            for i in range(min(8,len(bb))):assert np.array_equal(fp[i],base.pattern(x,bb[i]).astype(np.uint8))
            _,(_,segments),_=reference.candidate_bank(x,b,h,u,[])
            for segment in segments:
                j=segment['j'];lo=float(segment['lo']);hi=float(segment['hi']);t=(2*lo+hi)/3;other=(lo+2*hi)/3;prior=b[j]
                # Independent exact quadratic coefficient identity. The
                # constant is eliminated by comparing two actual objectives.
                left=b.copy();right=b.copy();hl=h.copy();hr=h.copy();left[j]=t;right[j]=other
                prev=x if j==0 else h[j-1];hl[j]=base.g(prev+t);hr[j]=base.g(prev+other)
                actual=common.energy_many(x,left[None],hl[None],u)[0]-common.energy_many(x,right[None],hr[None],u)[0]
                actual+=TAU*(np.sum((hl-h)**2)-np.sum((hr-h)**2)+n*((t-prior)**2-(other-prior)**2))
                expected=float(segment['quad'])*(t*t-other*other)-2*float(segment['linear'])*(t-other)
                err=abs(actual-expected);assert err<1e-10;max_objective_error=max(max_objective_error,err);formula_checks+=1
            cb,cm=coordinate_scan(x,b);patterns={r.tobytes() for r in forward_many(x,cb)[0]}
            for j in range(d):
                exact=coordinate_intervals_exact(x,b,j);assert exact[0][0]==-F(B) and exact[-1][1]==F(B)
                assert all(a[1]==z[0] for a,z in zip(exact[:-1],exact[1:]))
                for lo,hi,s,c,history in exact:
                    t=(lo+hi)/2;prev=[F(float(q)) for q in x];codes=[]
                    for layer in range(d):
                        z=[q+(t if layer==j else F(float(b[layer]))) for q in prev];codes.append([sum(v>=k for k in [F(0),F(1,2),F(1)]) for v in z]);prev=[reference.scalar.g(v) for v in z]
                    assert prev==[a*t+z for a,z in zip(s,c)] and codes==history;fiber_points+=1
                edges=np.array([float(t[0]) for t in exact]+[float(exact[-1][1])])
                for t in np.linspace(-B,B,257):
                    if np.min(np.abs(edges-t))<1e-12:near_boundaries+=1;continue
                    point=b.copy();point[j]=t;key=base.pattern(x,point).astype(np.uint8).tobytes();assert key in patterns;probe_checks+=1
            cases+=1
    return dict(passed=True,cases=cases,tied_quadratic_difference_checks=formula_checks,max_difference_error=max_objective_error,
        rational_fiber_midpoint_checks=fiber_points,dense_nonboundary_probes=probe_checks,excluded_near_boundary_probes=near_boundaries,
        scope='proposal primitive tests; not task benefit, floating completeness, or solver certification')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

"""Exact residual cuts and one-coordinate piecewise-quadratic repair.

Each stored direction a defines Phi_a = sum a_l (h_l - g(hprev+b_l)).
Every actual network has Phi_a=0. Unlike canonical branch labels, these
continuous cuts respect both sides of a shared activation endpoint.
Fractions are used for cut intervals, candidate certification and ordering.
This is a correctness prototype, not an accelerated implementation.
"""
from fractions import Fraction as F
import math
import numpy as np
import conflict_local_repair as old

base=old.base
B=F(float(old.BOUND)); TAU=F(float(old.TRUST))
S=[0,2,-2,0]; C=[0,0,2,0]


def g(z):
    return max(F(0),F(1)-abs(2*z-1))


def frac_array(a):
    return [[F(float(v)) for v in row] for row in np.asarray(a)]


def phi_exact(x,b,h,a):
    xx=[F(float(t)) for t in x];bb=[F(float(t)) for t in b]
    hh=frac_array(h);aa=frac_array(a);prev=xx;value=F(0)
    for j in range(len(bb)):
        value+=sum(aa[j][i]*(hh[j][i]-g(prev[i]+bb[j])) for i in range(len(xx)))
        prev=hh[j]
    return value


def representable_interval(lo,hi):
    """Smallest/largest binary64 contained in an exact closed interval."""
    left=float(lo);right=float(hi)
    if F(left)<lo:left=float(np.nextafter(left,np.inf))
    if F(right)>hi:right=float(np.nextafter(right,-np.inf))
    if left>right:return None
    assert lo<=F(left)<=F(right)<=hi
    return left,right


def solve_interval(lo,hi,alpha,beta):
    """Intersect affine inequalities alpha*t+beta <= 0, exactly."""
    for a,c in zip(alpha,beta):
        if a>0:hi=min(hi,-c/a)
        elif a<0:lo=max(lo,-c/a)
        elif c>0:return None
        if lo>hi:return None
    return lo,hi


def nearest_representable(t,lo,hi):
    interval=representable_interval(lo,hi)
    if interval is None:return None
    value=min(max(float(t),interval[0]),interval[1])
    assert lo<=F(value)<=hi
    return F(value)


def candidate_bank(x,b,h,u,clauses):
    """Return both unconstrained and all-cut constrained segment minima.

    Output h[-1] is fixed. Every segment gives its exact real optimum and
    nearest feasible representable point. Global objectives use rational
    arithmetic, including the original point's branch, not an extrapolation.
    """
    d,n=h.shape;xx=[F(float(t)) for t in x];bb=[F(float(t)) for t in b]
    hh=frac_array(h);uu=frac_array(u);aa=[frac_array(c['a']) for c in clauses]
    prevs=[xx]+hh[:-1];pred=[[g(t+bb[j]) for t in prevs[j]] for j in range(d)]
    residual=[[hh[j][i]-pred[j][i] for i in range(n)] for j in range(d)]
    phi=[sum(a[j][i]*residual[j][i] for j in range(d) for i in range(n)) for a in aa]
    energy=sum((residual[j][i]+uu[j][i])**2 for j in range(d) for i in range(n))
    original_allowed=all(v<=0 for v in phi)
    candidates=[dict(kind='unchanged',j=-1,i=-1,segment=-1,value=None,objective=energy,
                     allowed=original_allowed,real_objective=energy)]
    segments=[];empty=0;unrepresentable=0

    def append(kind,j,i,k,lo,hi,alpha,beta,quad,linear,old_value,new_cost,old_cost,weight):
        nonlocal empty,unrepresentable
        optimum=linear/quad
        free=min(max(optimum,lo),hi)
        v=nearest_representable(free,lo,hi)
        if v is not None:
            obj=energy+new_cost(v)-old_cost+weight*(v-old_value)**2
            candidates.append(dict(kind=kind,j=j,i=i,segment=k,value=v,objective=obj,
                                   allowed=all(a*v+c<=0 for a,c in zip(alpha,beta)),
                                   real_objective=energy+new_cost(free)-old_cost+weight*(free-old_value)**2))
        domain=solve_interval(lo,hi,alpha,beta)
        segment=dict(kind=kind,j=j,i=i,segment=k,lo=lo,hi=hi,alpha=alpha,beta=beta,
                     feasible=domain is not None)
        segments.append(segment)
        if domain is None:empty+=1;return
        cutlo,cuthi=domain;exact_value=min(max(optimum,cutlo),cuthi)
        value=nearest_representable(exact_value,cutlo,cuthi)
        segment.update(cutlo=cutlo,cuthi=cuthi,exact_value=exact_value)
        if value is None:unrepresentable+=1;return
        obj=energy+new_cost(value)-old_cost+weight*(value-old_value)**2
        assert all(a*value+c<=0 for a,c in zip(alpha,beta))
        candidates.append(dict(kind=kind,j=j,i=i,segment=k,value=value,objective=obj,allowed=True,
                               real_objective=energy+new_cost(exact_value)-old_cost+weight*(exact_value-old_value)**2))

    for j in range(d-1):
        nb=bb[j+1]
        for i in range(n):
            target=hh[j+1][i]+uu[j+1][i];incoming=pred[j][i]-uu[j][i];prior=hh[j][i]
            old_cost=(prior-incoming)**2+(target-g(prior+nb))**2
            for k in range(4):
                lo=max(F(0),[-B,F(0),F(1,2),F(1)][k]-nb)
                hi=min(F(1),[F(0),F(1,2),F(1),F(1)+B][k]-nb)
                if lo>hi:continue
                s,c=S[k],C[k];offset=s*nb+c
                alpha=[a[j][i]-s*a[j+1][i] for a in aa]
                beta=[p-a[j][i]*prior+a[j+1][i]*g(prior+nb)-a[j+1][i]*offset for p,a in zip(phi,aa)]
                append('activity',j,i,k,lo,hi,alpha,beta,F(1+s*s)+TAU,
                       incoming+s*(target-offset)+TAU*prior,prior,
                       lambda t,inc=incoming,tar=target,sl=s,off=offset:(t-inc)**2+(tar-sl*t-off)**2,
                       old_cost,TAU)
    for j in range(d):
        prev=prevs[j];target=[hh[j][i]+uu[j][i] for i in range(n)];prior=bb[j]
        events=sorted(set([-B,B]+[min(max(k-t,-B),B) for k in [F(0),F(1,2),F(1)] for t in prev]))
        old_cost=sum((target[i]-pred[j][i])**2 for i in range(n))
        for k,(lo,hi) in enumerate(zip(events[:-1],events[1:])):
            mid=(lo+hi)/2;reg=[sum(z>=t for t in [F(0),F(1,2),F(1)]) for z in [v+mid for v in prev]]
            ss=[S[r] for r in reg];off=[ss[i]*prev[i]+C[reg[i]] for i in range(n)]
            alpha=[-sum(a[j][i]*ss[i] for i in range(n)) for a in aa]
            beta=[p+sum(a[j][i]*(pred[j][i]-off[i]) for i in range(n)) for p,a in zip(phi,aa)]
            append('bias',j,-1,k,lo,hi,alpha,beta,F(sum(s*s for s in ss))+n*TAU,
                   sum(ss[i]*(target[i]-off[i]) for i in range(n))+n*TAU*prior,prior,
                   lambda t,tar=target,sl=ss,ofs=off:sum((tar[i]-sl[i]*t-ofs[i])**2 for i in range(n)),
                   old_cost,n*TAU)
    return candidates,segments,dict(phi=phi,energy=energy,segments=len(segments),empty_segments=empty,
                                   unrepresentable_segments=unrepresentable)


def repair_one(x,b,h,u,clauses,enforce=True,details=False):
    reg=old.split_many(x,b[None],h[None])[0]
    if not clauses or not old.conflict.clause_mask(reg[None],clauses)[0]:
        return b,h,dict(triggered=False,accepted=False,candidates=0)
    bank,segments,stats=candidate_bank(x,b,h,u,clauses)
    allowed=[i for i,c in enumerate(bank) if c['allowed'] or not enforce]
    meta=dict(triggered=True,accepted=False,candidates=len(bank),segments=stats['segments'],
              empty_segments=stats['empty_segments'],unrepresentable_segments=stats['unrepresentable_segments'],
              allowed_candidates=len(allowed),positive_cuts_before=sum(v>0 for v in stats['phi']))
    if not allowed:
        meta['reason']='no feasible representable one-coordinate point'
        if details:meta['details']=(bank,segments,stats)
        return b,h,meta
    index=min(allowed,key=lambda i:(bank[i]['objective'],i));chosen=bank[index]
    if chosen['kind']=='unchanged':meta['reason']='minimum is unchanged';return b,h,meta
    nb=b.copy();nh=h.copy();value=float(chosen['value'])
    if chosen['kind']=='activity':nh[chosen['j'],chosen['i']]=value
    else:nb[chosen['j']]=value
    after=[phi_exact(x,nb,nh,c['a']) for c in clauses]
    assert not enforce or all(v<=0 for v in after)
    meta.update(accepted=True,selected_index=index,selected_kind=chosen['kind'],
                selected_coordinate=[chosen['j'],chosen['i'],chosen['segment']],
                exact_cut_checks=len(after),positive_cuts_after=sum(v>0 for v in after),
                objective_before=float(stats['energy']),proximal_objective=float(chosen['objective']),
                real_proximal_objective=float(chosen['real_objective']),
                state_change_squared=float(np.sum((nb-b)**2)+np.sum((nh-h)**2)),
                post_repair_conflict=bool(old.conflict.clause_mask(old.split_many(x,nb[None],nh[None]),clauses)[0]))
    if details:meta['details']=(bank,segments,stats)
    return nb,nh,meta


def verify():
    rng=np.random.default_rng(921720);cases=0;segments_checked=0;max_affine_error=0.;accepted=0
    for d,n in [(2,2),(4,4)]:
        for _ in range(6):
            x=rng.uniform(0,1,n);b=rng.uniform(-float(B),float(B),d);h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n))
            reg=old.split_many(x,b[None],h[None])[0];clauses=[dict(positions=list(range(d*n)),codes=reg.ravel().tolist(),a=rng.normal(size=(d,n)).tolist()) for _ in range(3)]
            bank,segs,stats=candidate_bank(x,b,h,u,clauses)
            for seg in segs:
                val=(seg['lo']+seg['hi'])/2;bb=b.copy();hh=h.copy()
                if seg['kind']=='activity':hh[seg['j'],seg['i']]=float(val)
                else:bb[seg['j']]=float(val)
                t=F(float(val))
                for a,c,clause in zip(seg['alpha'],seg['beta'],clauses):
                    expected=phi_exact(x,bb,hh,clause['a']);assert expected==a*t+c
                    max_affine_error=max(max_affine_error,abs(float(expected-a*t-c)))
                segments_checked+=1
            nb,nh,meta=repair_one(x,b,h,u,clauses,True)
            if meta['accepted']:
                accepted+=1;assert all(phi_exact(x,nb,nh,c['a'])<=0 for c in clauses)
                assert meta['proximal_objective']==float(min(c['objective'] for c in bank if c['allowed']))
                assert np.count_nonzero(nb!=b)+np.count_nonzero(nh!=h)<=1 and np.array_equal(nh[-1],h[-1])
            else:assert np.array_equal(nb,b) and np.array_equal(nh,h)
            assert sum(seg['feasible'] for seg in segs)+stats['empty_segments']==len(segs);cases+=1
    # An interval can be mathematically nonempty but have no binary64 point.
    t=F(1,3);assert representable_interval(t,t) is None
    assert solve_interval(F(0),F(1),[F(1),F(-1)],[F(-1,4),F(3,4)]) is None
    assert solve_interval(F(0),F(1),[F(1),F(-1)],[F(-3,4),F(1,4)])==(F(1,4),F(3,4))
    return dict(passed=True,random_cases=cases,exact_affine_segments=segments_checked,accepted_synthetic_repairs=accepted,
                max_affine_error=max_affine_error,nonrepresentable_boundary_test=True,
                scope='synthetic arbitrary cuts verify primitive algebra; not valid learned conflicts or task superiority')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

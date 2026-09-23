"""Support-only exhaustive cell reference using observation-wise intersections.

This is a global reference solver, not the proposed local algorithm. Every LP
rejection is checked with an exact binary-rational box-separation certificate.
All unresolved LP branches are retained. Final volumes are still numerical.
"""
from fractions import Fraction as F
from itertools import product
import time
import numpy as np
from scipy.optimize import linprog
import stateful_posterior_memory as memory
base=memory.base
B=F(.12)


def constraints(x,v,regs,exact=False):
    d,n=regs.shape;rows=[];rhs=[]
    # Assemble per observation; this is only a row permutation of the old map.
    for i in range(n):
        p=[F(0)]*d if exact else [0.]*d;c=F(float(x[i])) if exact else float(x[i])
        for j in range(d):
            z=p.copy();z[j]+=1;reg=int(regs[j,i])
            lo=[None,0,F(1,2) if exact else .5,1][reg];hi=[0,F(1,2) if exact else .5,1,None][reg]
            if hi is not None:rows.append(z.copy());rhs.append(hi-c)
            if lo is not None:rows.append([-a for a in z]);rhs.append(c-lo)
            s=[0,2,-2,0][reg];intercept=[0,0,2,0][reg]
            p=[s*a for a in z];c=s*c+intercept
        vi=F(float(v[i])) if exact else float(v[i]);eps=F(base.EPS) if exact else base.EPS
        rows.extend([p.copy(),[-a for a in p]]);rhs.extend([vi+eps-c,-vi+eps+c])
    return rows,rhs


def exact_separation(a,rhs,multipliers):
    """d(lambda)=-lambda*r-B*||lambda*A||_1 > 0 excludes the box."""
    ids=np.flatnonzero(multipliers>0);lam=[F(float(multipliers[i])) for i in ids]
    bc=[sum((ll*a[int(i)][j] for i,ll in zip(ids,lam)),F(0)) for j in range(len(a[0]))]
    lower=-sum((ll*rhs[int(i)] for i,ll in zip(ids,lam)),F(0))-B*sum(map(abs,bc),F(0))
    return lower,[[int(i),float(multipliers[i])] for i in ids]


class BudgetReached(Exception):pass


class Reference:
    def __init__(self,x,v,depth=4,max_lp=50000,max_seconds=180):
        self.x=x;self.v=v;self.depth=depth;self.max_lp=max_lp;self.max_seconds=max_seconds;self.begin=time.perf_counter()
        self.lp_calls=0;self.certificates=[];self.unresolved_calls=0;self.radius_positive_calls=0

    def classify(self,ids,regs):
        if self.lp_calls>=self.max_lp or time.perf_counter()-self.begin>self.max_seconds:raise BudgetReached()
        aa,rr=constraints(self.x[ids],self.v[ids],regs,exact=True);d=self.depth
        for j in range(d):
            row=[F(0)]*d;row[j]=F(1);aa.append(row);rr.append(B)
            aa.append([-q for q in row]);rr.append(B)
        a=np.array(aa,dtype=float);rhs=np.array(rr,dtype=float);norm=np.abs(a).sum(1)
        result=linprog(np.r_[np.zeros(d),-1.],A_ub=np.c_[a,norm],b_ub=rhs,bounds=[(None,None)]*(d+1),
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        self.lp_calls+=1
        if result.success and result.x[-1]>1e-10:
            self.radius_positive_calls+=1;return True
        if result.success:
            lam=np.maximum(-result.ineqlin.marginals,0.);lower,sparse=exact_separation(aa,rr,lam)
            if lower>0:
                self.certificates.append(dict(observation_ids=list(map(int,ids)),pattern=regs.tolist(),multipliers=sparse,
                    lower_numerator=str(lower.numerator),lower_denominator=str(lower.denominator)))
                return False
        self.unresolved_calls+=1;return True

    def enumerate(self):
        n=len(self.x);d=self.depth;allpaths=np.array(list(product(range(4),repeat=d)),dtype=np.uint8)
        singles=[];screen_counts=[];counts=[];complete=False;polys=[];geometry_unresolved=[]
        try:
            for i in range(n):
                regs=allpaths[:,:,None];skip=memory.screen.contract(self.x[i:i+1],self.v[i:i+1],regs,5)
                surviving=[]
                for path in allpaths[~skip]:
                    if self.classify(np.array([i]),path[:,None]):surviving.append(path.copy())
                singles.append(surviving);screen_counts.append(int(skip.sum()))
            order=sorted(range(n),key=lambda i:(len(singles[i]),i));partial=[np.empty((d,0),dtype=np.uint8)];ids=[]
            for i in order:
                ids=ids+[i];new=[]
                for previous in partial:
                    for path in singles[i]:
                        reg=np.column_stack([previous,path])
                        if len(ids)==1 or self.classify(np.array(ids),reg):new.append(reg)
                partial=new;counts.append(dict(observations=ids.copy(),retained_patterns=len(partial),lp_calls=self.lp_calls))
            for reg in partial:
                reg=reg[:,np.argsort(order)]
                # Match original geometry's exact row order, not the reference LP.
                p=np.zeros((n,d));c=self.x.copy();ar=[];br=[]
                for j in range(d):
                    z=p.copy();z[:,j]+=1;r=reg[j]
                    low=np.array([-np.inf,0,.5,1.]);high=np.array([0,.5,1.,np.inf])
                    for i in range(n):
                        if np.isfinite(high[r[i]]):ar.append(z[i]);br.append(high[r[i]]-c[i])
                        if np.isfinite(low[r[i]]):ar.append(-z[i]);br.append(c[i]-low[r[i]])
                    p=base.SLOPES[r,None]*z;c=base.SLOPES[r]*c+base.INTERCEPTS[r]
                for i in range(n):ar.extend([p[i],-p[i]]);br.extend([self.v[i]+base.EPS-c[i],-self.v[i]+base.EPS+c[i]])
                poly,note=memory.posterior.polytope(np.array(ar),np.array(br))
                if poly is None:geometry_unresolved.append(dict(pattern=reg.tobytes().hex(),**note));continue
                representative=poly['center']+poly['scale']*poly['interior']
                assert np.array_equal(base.pattern(self.x,representative),reg)
                polys.append(dict(pattern=reg.tobytes().hex(),representative=representative.tolist(),**note))
            complete=True
        except BudgetReached:pass
        return dict(enumeration_completed=complete,final_geometry_unresolved=geometry_unresolved,
            numerical_volume_reference_complete=complete and not geometry_unresolved,lp_calls=self.lp_calls,
            positive_radius_calls=self.radius_positive_calls,unresolved_lp_calls_retained=self.unresolved_calls,
            certified_lp_rejections=len(self.certificates),singleton_interval_rejections=screen_counts,
            singleton_retained_counts=list(map(len,singles)),prefix_counts=counts,positive_regions=polys,
            total_prior_mass=sum(p['volume'] for p in polys)/float((2*B)**d),
            elapsed_seconds=time.perf_counter()-self.begin,certificates=self.certificates,
            scope='complete pattern enumeration when flagged; rational LP rejection proofs, outward interval screen; numerical final volumes, not exact-real integration')


def verify():
    rng=np.random.default_rng(90184);checks=0
    for d in [2,4]:
        x=rng.uniform(0,1,4);b=rng.uniform(-.12,.12,d);v=base.forward(x,b)
        a,r=constraints(x,v,base.pattern(x,b),True);aa=np.array(a,float);rr=np.array(r,float)
        # Every exact affine constraint is obeyed by an interior true parameter.
        assert np.max(aa@b-rr)<1e-12;checks+=len(r)
    x=np.array([.41,.63]);b=np.array([.017,-.031]);v=base.forward(x,b)
    reference=Reference(x,v,depth=2,max_lp=2000,max_seconds=20);out=reference.enumerate();assert out['enumeration_completed']
    key=base.pattern(x,b).astype(np.uint8).tobytes().hex();assert key in {p['pattern'] for p in out['positive_regions']}
    for c in out['certificates']:
        ids=np.array(c['observation_ids']);aa,rr=constraints(x[ids],v[ids],np.array(c['pattern'],np.uint8),True)
        for j in range(2):
            row=[F(0)]*2;row[j]=F(1);aa.extend([row,[-v for v in row]]);rr.extend([B,B])
        lam=np.zeros(len(aa))
        for i,val in c['multipliers']:lam[i]=val
        val,_=exact_separation(aa,rr,lam);assert val==F(int(c['lower_numerator']),int(c['lower_denominator'])) and val>0
    return dict(passed=True,affine_constraint_checks=checks,toy_positive_regions=len(out['positive_regions']),
        toy_exact_rejection_proofs=len(out['certificates']),toy_all_patterns_processed=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

"""372 offline diagnosis of C20 prefix survivors, not a candidate solver."""
from fractions import Fraction as F
import time
import numpy as np
from scipy.optimize import linprog
import local_region_screen as screen
import complete_credit_mode_geometry_v1 as proof
from test_region_conditioned_credit_v1 import guarded

B=F(.12)


def matrices(x,v,reg):
    # Layer-major exact construction, independent of old observation-major builder.
    d,n=reg.shape;assert d==4 and x.shape==v.shape==(n,)
    _,_,hl,hh=screen.boxes(v,reg[None]);hl,hh=hl[0,-1],hh[0,-1]
    p=[[F(0)]*d for _ in range(n)];c=[F(float(t)) for t in x];a=[];rhs=[]
    for j in range(d):
        for i in range(n):
            z=p[i].copy();z[j]+=1;code=int(reg[j,i]);lo=[None,F(0),F(1,2),F(1)][code];hi=[F(0),F(1,2),F(1),None][code]
            if hi is not None:a.append(z.copy());rhs.append(hi-c[i])
            if lo is not None:a.append([-t for t in z]);rhs.append(c[i]-lo)
            slope=[0,2,-2,0][code];offset=[0,0,2,0][code]
            p[i]=[slope*t for t in z];c[i]=slope*c[i]+offset
    for i in range(n):
        a.extend([p[i],[-t for t in p[i]]]);rhs.extend([F(float(hh[i]))-c[i],c[i]-F(float(hl[i]))])
    ba,br=proof.box_rows(d);return a+ba,rhs+br


def classify(a,rhs):
    begin=time.perf_counter();d=len(a[0])
    answer=dict(classification='unresolved',certificates=[],lp_calls=0,closed_region_feasible=None)
    bad=next((i for i,(row,r) in enumerate(zip(a,rhs)) if not any(row) and r<0),None)
    if bad is not None:
        answer.update(classification='infeasible',closed_region_feasible=False,
            certificates=[dict(type='negative_constant_row',row_index=bad,rhs=rhs[bad])],seconds=time.perf_counter()-begin)
        return answer
    aa=np.array(a,float);rr=np.array(rhs,float);norm=abs(aa).sum(1)
    result=linprog(np.r_[np.zeros(d),-1.],A_ub=np.c_[aa,norm],b_ub=rr,bounds=[(None,None)]*(d+1),
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    answer.update(lp_calls=1,lp_status=int(result.status))
    if result.success:
        point=proof.point_certificate(a,rhs,result.x[:-1]);answer['certificates'].append(point)
        answer['closed_region_feasible']=point['closed_region_feasible']
        if point['strict_interior']:answer['classification']='positive_volume'
        else:
            weights=[F(float(t)) for t in np.maximum(-result.ineqlin.marginals,0.)]
            candidates=[proof.weighted_certificate(a,rhs,weights)]
            for den in [2**12,2**20]:candidates.append(proof.weighted_certificate(a,rhs,[w.limit_denominator(den) for w in weights],source=f'rationalized_{den}'))
            answer['certificates']+=candidates
            conclusion=next((q['conclusion'] for q in candidates if q['conclusion']=='infeasible'),None)
            if conclusion is None:conclusion=next((q['conclusion'] for q in candidates if q['conclusion']=='zero_volume_or_empty'),None)
            if conclusion is not None:answer['classification']=conclusion
            if conclusion=='infeasible':assert not point['closed_region_feasible'];answer['closed_region_feasible']=False
    answer['seconds']=time.perf_counter()-begin;return answer


def exact_box_bound(x,v,reg,p,a):
    # Evaluate the actual outward-rounded box, not the narrower EPS domain.
    zl,zh,hl,hh=[t[0] for t in screen.boxes(v,reg[None])];d,n=reg.shape
    pp=[[F(float(t)) for t in row] for row in p];aa=[[F(float(t)) for t in row] for row in a]
    total=F(0)
    def minimum(c,l,h):return c*F(float(l if c>=0 else h))
    for j in range(d):
        total+=minimum(-sum(pp[j],F(0)),-.12,.12)
        for i in range(n):
            s=[0,2,-2,0][int(reg[j,i])];offset=[0,0,2,0][int(reg[j,i])]
            total+=minimum(pp[j][i]-s*aa[j][i],zl[j,i],zh[j,i])
            total+=minimum(aa[j][i]-(pp[j+1][i] if j<d-1 else 0),hl[j,i],hh[j,i])-aa[j][i]*offset
            if j==0:total-=pp[j][i]*F(float(x[i]))
    return total


def probes(x,v,regs):
    result={};arrays={}
    with guarded():
        tick=time.perf_counter();arrays['c100_reject']=screen.contract(x,v,regs,100);result['c100_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();reject,meta=screen.pdhg(x,v,np.zeros((len(regs),4)),regs,steps=128,check_every=32)
        result['pdhg128_seconds']=time.perf_counter()-tick
    arrays['pdhg128_reject']=reject;result['pdhg']=meta
    result.update(query_targets_accessed=False,lp_accessed=False,generation_is_cold=True)
    return arrays,result

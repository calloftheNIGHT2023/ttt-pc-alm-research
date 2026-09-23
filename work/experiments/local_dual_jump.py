"""251: exact adjacent-multiplier branch comparison at arbitrary primal state.

No fixed-point assumption, query answer, teacher, or global chain derivative.
Only the two residuals incident on the selected scalar activity are used.
"""
from fractions import Fraction as F
import numpy as np
import multiplier_fixed_point_exact as exact

TAUS=[.5,1.,2.,4.,8.,16.,32.]

def scalars(x,b,h,j,i,rational=False):
    values=[x[i] if j==0 else h[j-1,i],b[j],b[j+1],h[j,i],h[j+1,i]]
    return [F(float(t)) for t in values] if rational else [float(t) for t in values]

def branch(z):return sum(z>=k for k in [0,.5,1.])

def values(s,tau,rational=False,u_override=None):
    prev,bias,nb,before,nexth=s
    g=exact.g if rational else lambda z:np.maximum(0.,1.-np.abs(2.*z-1.))
    zero=F(0) if rational else 0.;one=F(1) if rational else 1.;trust=exact.TRUST if rational else .01
    knots=exact.K if rational else [0.,.5,1.]
    incoming=g(prev+bias);residual=[before-incoming,nexth-g(before+nb)]
    u=[tau*t for t in residual] if u_override is None else list(u_override)
    a=incoming-u[0];target=nexth+u[1];ans=[]
    for k,slope in enumerate(exact.S):
        lo=zero if k==0 else max(zero,knots[k-1]-nb)
        hi=one if k==3 else min(one,knots[k]-nb)
        if lo>hi:continue
        offset=slope*nb+exact.C[k]
        z=min(hi,max(lo,(a+slope*(target-offset)+trust*before)/(1+slope*slope+trust)))
        energy=(z-a)**2+(g(z+nb)-target)**2+trust*(z-before)**2
        ans.append(dict(k=k,lo=lo,hi=hi,z=z,energy=energy,actual_branch=branch(z+nb)))
    return dict(current_branch=branch(before+nb),residual=residual,u=u,branches=ans)

def classify(zero,test):
    k0=zero['current_branch'];own=next((r for r in zero['branches'] if r['k']==k0),None);others=[r for r in zero['branches'] if r['k']!=k0]
    if own is None:return dict(accepted=False,reason='current branch outside activity domain')
    if not others:return dict(accepted=False,reason='no competing branch')
    if not all(own['energy']<r['energy'] for r in others):return dict(accepted=False,reason='zero dual not strict same-branch optimum')
    same=next(r for r in test['branches'] if r['k']==k0);outside=[r for r in test['branches'] if r['k']!=k0]
    best=min(outside,key=lambda r:(r['energy'],r['k']));gap=same['energy']-best['energy']
    return dict(accepted=bool(gap>0),reason='strict different-branch minimum' if gap>0 else 'no strict branch separation',
        original_branch=k0,competing_branch=best['k'],actual_branch=best['actual_branch'],margin=gap,z=best['z'])

def encode(values,rational):
    conv=exact.pack if rational else float
    return dict(current_branch=values['current_branch'],residual=[conv(t) for t in values['residual']],u=[conv(t) for t in values['u']],
        branches=[{k:(v if k in ['k','actual_branch'] else conv(v)) for k,v in r.items()} for r in values['branches']])

def decision_encode(decision,rational):
    return {k:(exact.pack(v) if rational else float(v)) if k in ['margin','z'] else v for k,v in decision.items()}

def check_rounded(s,float_values,float_decision):
    """Exact check of the actual binary h/u to be written, not just ideal tau."""
    u=[F(float(t)) for t in float_values['u']];z=F(float_decision['z']);exact_values=values(s,F(0),True,u)
    own=next(r for r in exact_values['branches'] if r['k']==exact_values['current_branch'])
    prev,bias,nb,before,nexth=s
    energy=(z-exact.g(prev+bias)+u[0])**2+(nexth-exact.g(z+nb)+u[1])**2+exact.TRUST*(z-before)**2
    margin=own['energy']-energy
    return dict(accepted=bool(F(0)<=z<=F(1) and branch(z+nb)!=exact_values['current_branch'] and margin>0),
        margin=exact.pack(margin),actual_branch=branch(z+nb),z=exact.pack(z),u=[exact.pack(t) for t in u])

def scan(x,b,h):
    blocks=[];candidates=[]
    for j in range(len(b)-1):
        for i in range(len(x)):
            fs=scalars(x,b,h,j,i);rs=scalars(x,b,h,j,i,True);f0=values(fs,0.);e0=values(rs,F(0),True);trials=[]
            for tau in TAUS:
                fv=values(fs,tau);ev=values(rs,F(tau),True);fd=classify(f0,fv);ed=classify(e0,ev)
                rounded=check_rounded(rs,fv,fd) if fd['accepted'] else None
                accepted=bool(fd['accepted'] and ed['accepted'] and rounded['accepted'])
                trial=dict(tau=tau,float_values=encode(fv,False),exact_values=encode(ev,True),float_decision=decision_encode(fd,False),
                    exact_decision=decision_encode(ed,True),rounded_write=rounded,accepted=accepted)
                trials.append(trial)
                if accepted:candidates.append(dict(tau=tau,j=j,i=i,k=fd['competing_branch'],z=float(fd['z']),u=list(map(float,fv['u'])),
                    exact_margin=exact.pack(ed['margin']),rounded_margin=rounded['margin']))
            blocks.append(dict(j=j,i=i,float_zero=encode(f0,False),exact_zero=encode(e0,True),trials=trials))
    candidates.sort(key=lambda r:(r['tau'],r['j'],r['i'],r['k']))
    return dict(blocks=blocks,candidates=candidates,selected=candidates[0] if candidates else None)

def verify():
    rng=np.random.default_rng(251731);checks=0
    for _ in range(12):
        x=rng.uniform(0,1,4);b=rng.uniform(-.12,.12,4);h=rng.uniform(0,1,(4,4));xx=[F(float(t)) for t in x];bb=[F(float(t)) for t in b];hh=[[F(float(t)) for t in row] for row in h]
        for j in range(3):
            for i in range(4):
                s=scalars(x,b,h,j,i,True)
                for tau in [F(0),F(1,2),F(4)]:
                    actual=values(s,tau,True);u=[[F(0)]*4 for _ in range(4)];u[j][i],u[j+1][i]=actual['u']
                    expected=exact.activity_candidates(xx,[F(0)]*4,bb,hh,j,i,u)
                    assert [(r['z'],r['energy']) for r in actual['branches']]==expected;checks+=1
    return dict(passed=True,exact_frozen_block_formula_cases=checks,no_global_bp_or_query=True)

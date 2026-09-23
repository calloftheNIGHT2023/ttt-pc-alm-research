"""Exact separable credit clauses; no query information or global derivatives."""
from fractions import Fraction as F
from itertools import product,combinations
import numpy as np
import local_region_screen as screen
import branch_normal_credit as normal
base=screen.base


def rational_table(x,v,p,a):
    d,n=a.shape;dummy=np.zeros((1,d,n),np.uint8)
    _,_,hl,hh=screen.boxes(v,dummy)
    zlo=[F(-screen.B),F(0),F(1,2),F(1)]
    zhi=[F(0),F(1,2),F(1),F(float(screen.up(1+screen.B)))]
    pp=[[F(float(t)) for t in row] for row in p]
    aa=[[F(float(t)) for t in row] for row in a]
    const=-F(screen.B)*sum(abs(sum(row)) for row in pp)
    const-=sum(pp[0][i]*F(float(x[i])) for i in range(n));table=[]
    for j in range(d):
        for i in range(n):
            hc=aa[j][i]-(pp[j+1][i] if j+1<d else 0)
            const+=min(hc*F(float(hl[0,j,i])),hc*F(float(hh[0,j,i])))
            row=[]
            for k in range(4):
                zc=pp[j][i]-int(base.SLOPES[k])*aa[j][i]
                row.append(min(zc*zlo[k],zc*zhi[k])-int(base.INTERCEPTS[k])*aa[j][i])
            table.append(row)
    return const,table


def value(const,table,reg):
    return const+sum(row[int(k)] for row,k in zip(table,reg.ravel()))


def extract(x,v,reg,p,a):
    const,table=rational_table(x,v,p,a);m=[min(row) for row in table]
    floor=const+sum(m);gain=[row[int(k)]-low for row,k,low in zip(table,reg.ravel(),m)]
    original=floor+sum(gain)
    if original<=0:return None
    order=sorted(range(len(gain)),key=lambda t:(-gain[t],t));bound=floor;chosen=[]
    for t in order:
        if bound>0:break
        chosen.append(t);bound+=gain[t]
    assert bound>0
    previous=floor+sum(gain[t] for t in order[:max(0,len(chosen)-1)])
    assert not chosen or previous<=0
    cumulative=F(0);required=len(gain)+1
    for k,t in enumerate(order,1):
        cumulative+=gain[t]
        if cumulative>=original:required=k;break
    return dict(positions=chosen,codes=[int(reg.ravel()[t]) for t in chosen],
        positive_numerator=str(bound.numerator),positive_denominator=str(bound.denominator),
        previous_cardinality_best=float(previous),original_value=float(original),global_minimum=float(floor),
        minimum_required_edits=required,cardinality=len(chosen),p=p.tolist(),a=a.tolist())


def clause_mask(regs,clauses):
    flat=regs.reshape(len(regs),-1);reject=np.zeros(len(regs),bool)
    for c in clauses:
        reject|=np.all(flat[:,c['positions']]==np.array(c['codes'])[None],axis=1)
    return reject


def full_mask(x,v,regs,clauses):
    """Float proposals, exact table recomputation for every accepted deletion."""
    reject=np.zeros(len(regs),bool);checks=0
    for c in clauses:
        p=np.array(c['p']);a=np.array(c['a']);ids=np.flatnonzero(~reject)
        if not len(ids):break
        rough=screen.float_bound(x,v,regs[ids],np.broadcast_to(p,(len(ids),*p.shape)),np.broadcast_to(a,(len(ids),*a.shape)))
        const,table=rational_table(x,v,p,a)
        for index in ids[rough>1e-10*(1+np.abs(p).sum()+np.abs(a).sum())]:
            checks+=1
            if value(const,table,regs[index])>0:reject[index]=True
    # Clauses can prove a very small positive bound below the rough threshold.
    reject|=clause_mask(regs,clauses)
    return reject,checks


def radius_two(reg):
    flat=reg.ravel();out={reg.tobytes():0}
    for k in [1,2]:
        for positions in combinations(range(len(flat)),k):
            choices=[[i for i in range(4) if i!=flat[t]] for t in positions]
            for codes in product(*choices):
                new=flat.copy();new[list(positions)]=codes;out[new.tobytes()]=k
    return out


def verify():
    rng=np.random.default_rng(553481);cases=0;certificates=0;max_error=0.
    for d,n in [(1,2),(2,2)]:
        regs=np.array(list(product(range(4),repeat=d*n)),np.uint8).reshape(-1,d,n)
        for _ in range(4):
            x=rng.uniform(0,1,n);v=base.forward(x,rng.uniform(-.12,.12,d));p=rng.normal(size=(d,n));a=rng.normal(size=(d,n))
            const,table=rational_table(x,v,p,a);ev=np.array([float(value(const,table,r)) for r in regs])
            fv=screen.float_bound(x,v,regs,np.broadcast_to(p,regs.shape),np.broadcast_to(a,regs.shape))
            error=float(np.max(np.abs(ev-fv)));assert error<1e-10;max_error=max(max_error,error);cases+=len(regs)
            for reg in regs[ev>1e-10][:4]:
                c=extract(x,v,reg,p,a);assert c is not None
                matched=clause_mask(regs,[c]);exact=[value(const,table,r) for r in regs[matched]]
                assert min(exact)==F(int(c['positive_numerator']),int(c['positive_denominator']))>0
                loss=[row[int(k)]-min(row) for row,k in zip(table,reg.ravel())];floor=const+sum(min(row) for row in table)
                for subset in combinations(range(d*n),max(0,c['cardinality']-1)):
                    assert floor+sum(loss[t] for t in subset)<=0
                distance=(regs!=reg).sum((1,2));assert all(value(const,table,r)>0 for r in regs[distance<c['minimum_required_edits']])
                certificates+=1
    assert len(radius_two(np.ones((4,4),np.uint8)))==1129
    return dict(passed=True,exhaustive_pattern_values=cases,exhaustive_clause_and_cardinality_checks=certificates,
        max_exact_vs_original_float_error=max_error,radius_two_count=1129,claim='minimum cardinality only within fixed-credit separable certificate family')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))

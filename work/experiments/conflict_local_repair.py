"""Minimum-energy allowed finite one-coordinate proposals under old clauses."""
import numpy as np
import credit_conflict as conflict
base=conflict.base;BOUND=conflict.screen.B;TRUST=.01


def split_many(x,b,h):
    prev=np.broadcast_to(x,(len(b),len(x)));codes=[]
    for j in range(b.shape[1]):
        codes.append(np.searchsorted(base.KNOTS,prev+b[:,j,None],side='right').astype(np.uint8));prev=h[:,j]
    return np.array(codes).transpose(1,0,2)


def energy_many(x,b,h,u):
    prev=np.broadcast_to(x,(len(b),len(x)));total=np.zeros(len(b))
    for j in range(b.shape[1]):
        total+=np.sum((h[:,j]-base.g(prev+b[:,j,None])+u[j])**2,axis=1);prev=h[:,j]
    return total


def candidate_bank(x,b,h,u):
    d,n=h.shape;bb=[b.copy()];hh=[h.copy()];labels=[('unchanged',-1,-1,-1)];loz=[-np.inf,0.,.5,1.];hiz=[0.,.5,1.,np.inf]
    for j in range(d-1):
        prev=x if j==0 else h[j-1];a=base.g(prev+b[j])-u[j];target=h[j+1]+u[j+1];nb=b[j+1]
        for i in range(n):
            for branch in range(4):
                lo=max(0.,loz[branch]-nb);hi=min(1.,hiz[branch]-nb)
                if lo>hi:continue
                s=base.SLOPES[branch];offset=s*nb+base.INTERCEPTS[branch]
                value=float(np.clip((a[i]+s*(target[i]-offset)+TRUST*h[j,i])/(1+s*s+TRUST),lo,hi))
                hb=h.copy();hb[j,i]=value;bb.append(b.copy());hh.append(hb);labels.append(('activity',j,i,branch))
    for j in range(d):
        prev=x if j==0 else h[j-1];target=h[j]+u[j];events=np.unique(np.r_[-BOUND,np.clip((base.KNOTS[:,None]-prev).ravel(),-BOUND,BOUND),BOUND])
        for k,(lo,hi) in enumerate(zip(events[:-1],events[1:])):
            mid=(lo+hi)/2;reg=np.searchsorted(base.KNOTS,prev+mid,side='right');s=base.SLOPES[reg];offset=s*prev+base.INTERCEPTS[reg]
            value=float(np.clip((np.sum(s*(target-offset))+n*TRUST*b[j])/(np.sum(s*s)+n*TRUST),lo,hi))
            bc=b.copy();bc[j]=value;bb.append(bc);hh.append(h.copy());labels.append(('bias',j,-1,k))
    bnew=np.array(bb);hnew=np.array(hh);objective=energy_many(x,bnew,hnew,u)
    objective+=TRUST*(np.sum((hnew-h)**2,axis=(1,2))+n*np.sum((bnew-b)**2,axis=1))
    return bnew,hnew,objective,labels


def repair_one(x,b,h,u,clauses,enforce=True):
    reg=split_many(x,b[None],h[None])[0]
    if not clauses or not conflict.clause_mask(reg[None],clauses)[0]:return b,h,dict(triggered=False,accepted=False,candidates=0)
    bb,hh,objective,labels=candidate_bank(x,b,h,u);regs=split_many(x,bb,hh);violates=conflict.clause_mask(regs,clauses);allowed=~violates if enforce else np.ones(len(bb),bool)
    if not np.any(allowed):return b,h,dict(triggered=True,accepted=False,candidates=len(bb),reason='no allowed finite one-coordinate candidate')
    values=np.where(allowed,objective,np.inf);index=int(np.argmin(values));before=float(energy_many(x,b[None],h[None],u)[0]);after=float(energy_many(x,bb[index:index+1],hh[index:index+1],u)[0])
    if labels[index][0]=='unchanged':return b,h,dict(triggered=True,accepted=False,candidates=len(bb),reason='unconstrained minimum is unchanged')
    assert not enforce or not violates[index]
    return bb[index],hh[index],dict(triggered=True,accepted=True,candidates=len(bb),allowed_candidates=int(allowed.sum()),selected_index=index,selected_kind=labels[index][0],
        selected_coordinate=[int(v) for v in labels[index][1:]],objective_before=before,objective_after=after,proximal_objective=float(objective[index]),
        state_change_squared=float(np.sum((bb[index]-b)**2)+np.sum((hh[index]-h)**2)),post_repair_conflict=bool(violates[index]))


def verify():
    rng=np.random.default_rng(238491);cases=0;accepted=0;max_bias_gap=0.;max_local_gap=0.;oldbound=base.BOUND
    try:
        base.BOUND=BOUND
        for d,n in [(2,3),(4,4),(4,8)]:
            for _ in range(8):
                x=rng.uniform(0,1,n);b=rng.uniform(-BOUND,BOUND,d);h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n));bb,hh,obj,labels=candidate_bank(x,b,h,u)
                cb,ch,meta=repair_one(x,b,h,u,[]);assert np.array_equal(cb,b) and np.array_equal(ch,h) and not meta['triggered']
                for j in range(d):
                    ids=[k for k,label in enumerate(labels) if label[0]=='bias' and label[1]==j];prev=x if j==0 else h[j-1];target=h[j]+u[j]
                    expected=base.bias_solve(prev[None],target[None],b[j:j+1],0.,float('inf'),TRUST)[0];cost=lambda z:float(np.sum((base.g(prev+z)-target)**2)+n*TRUST*(z-b[j])**2)
                    got=bb[ids[int(np.argmin(obj[ids]))],j];gap=abs(cost(got)-cost(expected));max_bias_gap=max(max_bias_gap,gap);assert gap<1e-10
                reg=split_many(x,b[None],h[None])[0];clause=dict(positions=list(range(d*n)),codes=reg.ravel().tolist());cb,ch,meta=repair_one(x,b,h,u,[clause]);cases+=1
                if meta['accepted']:
                    accepted+=1;allow=~conflict.clause_mask(split_many(x,bb,hh),[clause]);assert meta['proximal_objective']==float(obj[allow].min());assert np.max(np.abs(cb))<=BOUND and np.all((ch>=0)&(ch<=1)) and np.array_equal(ch[-1],h[-1])
                    assert np.count_nonzero(cb!=b)+np.count_nonzero(ch!=h)==1
                impossible=[dict(positions=[0],codes=[k]) for k in range(4)];cb,ch,meta=repair_one(x,b,h,u,impossible);assert not meta['accepted'] and np.array_equal(cb,b) and np.array_equal(ch,h)
                initial=float(energy_many(x,b[None],h[None],u)[0])
                for k,label in enumerate(labels):
                    if label[0]!='activity':continue
                    _,j,i,_=label;prev=x[i] if j==0 else h[j-1,i];v0=h[j,i];v1=hh[k,j,i]
                    local=lambda z:(z-base.g(prev+b[j])+u[j,i])**2+(h[j+1,i]-base.g(z+b[j+1])+u[j+1,i])**2
                    difference=float(energy_many(x,bb[k:k+1],hh[k:k+1],u)[0]-initial);gap=abs(difference-(local(v1)-local(v0)));max_local_gap=max(max_local_gap,float(gap));assert gap<1e-10
    finally:base.BOUND=oldbound
    return dict(passed=True,cases=cases,accepted_synthetic_clause_repairs=accepted,max_bias_candidate_vs_original_energy_gap=max_bias_gap,max_local_vs_global_energy_delta_gap=max_local_gap,
        scope='synthetic clauses test finite repair mechanics only; validity of learned clauses is audited separately')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))

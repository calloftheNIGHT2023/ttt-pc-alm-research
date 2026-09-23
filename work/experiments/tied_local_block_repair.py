"""Finite, exactly certified tied-layer proposals, unioned with scalar repair.

Ideal tied blocks set b[j]=t and h[j]=g(hprev+t). We round the actual written
activities, then recalculate the affected residuals exactly. An ideal affine
cut value is never used as acceptance of a rounded state.
"""
from fractions import Fraction as F
import numpy as np
import credit_residual_cut as scalar

base=scalar.base;B=scalar.B;TAU=scalar.TAU;KNOTS=[F(0),F(1,2),F(1)]


def proposals(optimum,lo,hi,prior):
    """Frozen finite set: optimum, adjacent doubles, midpoint and old bias."""
    value=scalar.nearest_representable(min(max(optimum,lo),hi),lo,hi)
    if value is None:return []
    values=[value]
    for direction in [-np.inf,np.inf]:
        neighbor=F(float(np.nextafter(float(value),direction)))
        if lo<=neighbor<=hi:values.append(neighbor)
    middle=scalar.nearest_representable((lo+hi)/2,lo,hi)
    if middle is not None:values.append(middle)
    if lo<=prior<=hi:values.append(prior)
    return list(dict.fromkeys(values))


def candidate_bank(x,b,h,u,clauses):
    bank,scalar_segments,original=scalar.candidate_bank(x,b,h,u,clauses)
    old_count=len(bank);d,n=h.shape;xx=[F(float(v)) for v in x];bb=[F(float(v)) for v in b]
    hh=scalar.frac_array(h);uu=scalar.frac_array(u);aa=[scalar.frac_array(c['a']) for c in clauses]
    prevs=[xx]+hh[:-1];r0=[[hh[j][i]-scalar.g(prevs[j][i]+bb[j]) for i in range(n)] for j in range(d)]
    phi=original['phi'];energy=original['energy'];seen=set();segment_records=[]
    empty=0;unrepresentable=0;rounding_rejections=0;certified=0;max_rounding=F(0)
    for j in range(d-1):
        prev=prevs[j];neighbor=bb[j+1];events=[-B,B]
        for p in prev:
            events.extend(k-p for k in KNOTS)
            for k in KNOTS:
                y=k-neighbor
                if 0<=y<=1:events.extend([y/2-p,1-y/2-p])
        events=sorted(set(t for t in events if -B<=t<=B));assert len(events)-1<=9*n+1
        old_cost=sum((r0[j][i]+uu[j][i])**2+(r0[j+1][i]+uu[j+1][i])**2 for i in range(n))
        for k,(lo,hi) in enumerate(zip(events[:-1],events[1:])):
            mid=(lo+hi)/2;first=[sum(p+mid>=t for t in KNOTS) for p in prev]
            s=[F(scalar.S[z]) for z in first];q=[s[i]*prev[i]+scalar.C[first[i]] for i in range(n)]
            second=[sum(s[i]*mid+q[i]+neighbor>=t for t in KNOTS) for i in range(n)]
            m=[scalar.S[second[i]]*s[i] for i in range(n)];offset=[scalar.S[second[i]]*(q[i]+neighbor)+scalar.C[second[i]] for i in range(n)]
            target=[hh[j+1][i]+uu[j+1][i] for i in range(n)]
            quad=sum(t*t for t in m)+TAU*(sum(t*t for t in s)+n)
            linear=sum(m[i]*(target[i]-offset[i]) for i in range(n))+TAU*sum(s[i]*(hh[j][i]-q[i]) for i in range(n))+TAU*n*bb[j]
            alpha=[-sum(a[j+1][i]*m[i] for i in range(n)) for a in aa]
            beta=[p-sum(a[j][i]*r0[j][i]+a[j+1][i]*r0[j+1][i] for i in range(n))+sum(a[j+1][i]*(hh[j+1][i]-offset[i]) for i in range(n)) for p,a in zip(phi,aa)]
            optimum=linear/quad;domain=scalar.solve_interval(lo,hi,alpha,beta)
            record=dict(j=j,segment=k,lo=lo,hi=hi,quad=quad,linear=linear,alpha=alpha,beta=beta,feasible=domain is not None)
            values=proposals(optimum,lo,hi,bb[j])
            if domain is None:empty+=1
            else:
                record.update(cutlo=domain[0],cuthi=domain[1]);more=proposals(optimum,*domain,bb[j]);values.extend(more)
                if not more:unrepresentable+=1
            segment_records.append(record)
            for t in dict.fromkeys(values):
                token=j,t
                if token in seen:continue
                seen.add(token);ideal=[scalar.g(p+t) for p in prev];written=[F(float(v)) for v in ideal]
                rj=[written[i]-ideal[i] for i in range(n)];rnext=[hh[j+1][i]-scalar.g(written[i]+neighbor) for i in range(n)]
                exact_phi=[p+sum(a[j][i]*(rj[i]-r0[j][i])+a[j+1][i]*(rnext[i]-r0[j+1][i]) for i in range(n)) for p,a in zip(phi,aa)]
                ideal_phi=[a*t+c for a,c in zip(alpha,beta)];allowed=all(v<=0 for v in exact_phi)
                if all(v<=0 for v in ideal_phi) and not allowed:rounding_rejections+=1
                if allowed:certified+=1
                for actual,ideal_value in zip(exact_phi,ideal_phi):max_rounding=max(max_rounding,abs(actual-ideal_value))
                objective=energy-old_cost+sum((rj[i]+uu[j][i])**2+(rnext[i]+uu[j+1][i])**2 for i in range(n))
                objective+=TAU*(sum((written[i]-hh[j][i])**2 for i in range(n))+n*(t-bb[j])**2)
                bank.append(dict(kind='tied',j=j,i=-1,segment=k,value=t,activities=written,objective=objective,allowed=allowed,
                                 exact_phi=exact_phi,ideal_phi=ideal_phi,max_layer_residual=max(abs(v) for v in rj)))
    stats=dict(scalar=original,scalar_candidates=old_count,tied_candidates=len(bank)-old_count,tied_segments=len(segment_records),
               tied_empty_real_segments=empty,tied_unrepresentable_real_segments=unrepresentable,tied_certified_candidates=certified,
               tied_rounding_rejected_candidates=rounding_rejections,max_cut_rounding_difference=float(max_rounding),energy=energy)
    return bank,(scalar_segments,segment_records),stats


def materialize(b,h,chosen):
    nb=b.copy();nh=h.copy()
    if chosen['kind']=='activity':nh[chosen['j'],chosen['i']]=float(chosen['value'])
    elif chosen['kind']=='bias':nb[chosen['j']]=float(chosen['value'])
    elif chosen['kind']=='tied':
        nb[chosen['j']]=float(chosen['value']);nh[chosen['j']]=[float(v) for v in chosen['activities']]
    return nb,nh


def select(x,b,h,clauses,bank,stats,enforce=True,include_tied=True):
    limit=len(bank) if include_tied else stats['scalar_candidates'];ids=[i for i in range(limit) if bank[i]['allowed'] or not enforce]
    meta=dict(triggered=True,accepted=False,candidates=limit,allowed_candidates=len(ids),scalar_candidates=stats['scalar_candidates'],
              tied_candidates=stats['tied_candidates'] if include_tied else 0,tied_segments=stats['tied_segments'] if include_tied else 0,
              tied_empty_real_segments=stats['tied_empty_real_segments'] if include_tied else 0,
              tied_rounding_rejected_candidates=stats['tied_rounding_rejected_candidates'] if include_tied else 0,
              max_cut_rounding_difference=stats['max_cut_rounding_difference'] if include_tied else 0.)
    if not ids:meta['reason']='no certified finite candidate';return b,h,meta
    index=min(ids,key=lambda i:(bank[i]['objective'],i));chosen=bank[index]
    if chosen['kind']=='unchanged':meta['reason']='minimum is unchanged';return b,h,meta
    nb,nh=materialize(b,h,chosen);values=[scalar.phi_exact(x,nb,nh,c['a']) for c in clauses]
    assert not enforce or all(v<=0 for v in values)
    if chosen['kind']=='tied':assert values==chosen['exact_phi']
    meta.update(accepted=True,selected_index=index,selected_kind=chosen['kind'],selected_coordinate=[chosen['j'],chosen['i'],chosen['segment']],
                objective_before=float(stats['energy']),proximal_objective=float(chosen['objective']),exact_cut_checks=len(values),
                positive_cuts_after=sum(v>0 for v in values),state_change_squared=float(np.sum((nb-b)**2)+np.sum((nh-h)**2)),
                selected_layer_rounding_residual=float(chosen.get('max_layer_residual',0)),
                post_repair_conflict=bool(scalar.old.conflict.clause_mask(scalar.old.split_many(x,nb[None],nh[None]),clauses)[0]))
    if enforce and include_tied:
        old_ids=[i for i in range(stats['scalar_candidates']) if bank[i]['allowed']]
        if old_ids:assert chosen['objective']<=min(bank[i]['objective'] for i in old_ids)
    return nb,nh,meta


def repair_one(x,b,h,u,clauses,enforce=True):
    reg=scalar.old.split_many(x,b[None],h[None])[0]
    if not clauses or not scalar.old.conflict.clause_mask(reg[None],clauses)[0]:return b,h,dict(triggered=False,accepted=False,candidates=0)
    bank,_,stats=candidate_bank(x,b,h,u,clauses);return select(x,b,h,clauses,bank,stats,enforce)


def verify():
    # This independent checker recalculates the entire rational network.
    from verify_physical_cut_quadratics import full_objective
    rng=np.random.default_rng(583917);cases=0;objectives=0;cut_checks=0;scalar_matches=0;extended=0;rounding_rejections=0
    for d,n in [(2,2),(4,4),(4,8)]:
        for _ in range(4):
            x=rng.uniform(0,1,n);b=rng.uniform(-float(B),float(B),d);h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n))
            reg=scalar.old.split_many(x,b[None],h[None])[0]
            clauses=[dict(positions=list(range(d*n)),codes=reg.ravel().tolist(),a=rng.normal(size=(d,n)).tolist()) for _ in range(3)]
            bank,_,stats=candidate_bank(x,b,h,u,clauses);rounding_rejections+=stats['tied_rounding_rejected_candidates']
            ob,oh,om=scalar.repair_one(x,b,h,u,clauses);sb,sh,sm=select(x,b,h,clauses,bank,stats,True,False)
            assert np.array_equal(ob,sb) and np.array_equal(oh,sh) and om['accepted']==sm['accepted'];scalar_matches+=1
            for row in bank:
                nb,nh=materialize(b,h,row);assert full_objective(x,nb,nh,u,b,h)==row['objective'];objectives+=1
                values=[scalar.phi_exact(x,nb,nh,c['a']) for c in clauses];assert row['allowed']==all(v<=0 for v in values);cut_checks+=len(values)
                if row['kind']=='tied':assert values==row['exact_phi']
            nb,nh,meta=select(x,b,h,clauses,bank,stats)
            if meta['accepted']:
                assert np.array_equal(nh[-1],h[-1]) and np.max(np.abs(nb))<=float(B) and np.all((nh>=0)&(nh<=1))
                extended+=meta['selected_kind']=='tied'
                if om['accepted']:assert meta['proximal_objective']<=om['proximal_objective']
            else:assert not om['accepted']
            cases+=1
    assert proposals(F(1,3),F(1,3),F(1,3),F(0))==[]
    return dict(passed=True,random_cases=cases,unchanged_scalar_selection_cases=scalar_matches,independent_full_objective_checks=objectives,
                exact_cut_checks=cut_checks,tied_selected_cases=int(extended),rounding_rejections=int(rounding_rejections),
                scope='finite certified union primitive; arbitrary synthetic directions, not learned-clause validity or query gain')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

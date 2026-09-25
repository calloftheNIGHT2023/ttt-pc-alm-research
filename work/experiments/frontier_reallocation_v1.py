"""362 independent credit trajectories under a common finite response budget."""
import time
import numpy as np
import region_conditioned_credit_v1 as local
import budget_frontier_v1 as lazy
from branch_image_chain_dyadic_v1 import IntegerProblem


def solve(x,v,regs,credit,*,strategy='frontier',steps=128,geometry_budget=8):
    started=time.perf_counter();regs=np.asarray(regs,np.uint8);m,d,n=regs.shape
    if strategy not in ['frontier','uniform'] or steps<1 or geometry_budget<1:
        raise ValueError('Invalid strategy or budgets')
    if credit.shape!=(d,n) or not np.isfinite(credit).all():raise ValueError('Invalid initial credit')
    limit=m*steps;scale=np.linalg.norm(credit);initial=credit/scale if scale>0 else credit.copy()
    a=np.broadcast_to(initial,regs.shape).copy();first=np.zeros(m,np.int32);disabled=np.zeros(m,bool)
    proof_credit=np.zeros_like(a);visits=np.zeros(m,np.int64);proofs=[];trace=[]
    lookup={r.tobytes():i for i,r in enumerate(regs)}
    if len(lookup)!=m:raise ValueError('Ordered candidate pool must contain unique modes')
    original_response=local.response
    def counted_response(xx,vv,rr,aa):
        for reg in rr:visits[lookup[reg.tobytes()]]+=1
        return original_response(xx,vv,rr,aa)
    try:
        local.response=counted_response
        if strategy=='frontier':
            base,bmeta=lazy.solve(x,v,regs,credit,steps=steps,budget=geometry_budget)
            processed=bmeta['processed'];base_calls=bmeta['calls'];base_work=bmeta['response_pairs']
        else:
            base,bmeta=local.solve(x,v,regs,credit,steps);processed=m;base_work=bmeta['response_pairs']
            base_calls=[dict(begin=0,end=m,metadata=bmeta)] if m else []
    finally:local.response=original_response
    a[:processed]=base['final_credit'];first[:processed]=base['first_step']
    disabled[:processed]=base['disabled'];proof_credit[:processed]=base['proof_credit']
    assert int(visits.sum())==base_work<=limit
    for call in base_calls:
        for proof in call['metadata']['proofs']:
            proofs.append(dict(proof,index=int(call['begin']+proof['index']),stage='base',kind='local',source_index=None))
    base_selected=np.flatnonzero(first==0)[:geometry_budget]
    base_first=first.copy();base_disabled=disabled.copy();base_a=a.copy();base_visits=visits.copy();used=base_work
    exact_calls=sum(c['metadata']['exact_calls'] for c in base_calls);extra_exact=0;peak=0;rounds=0
    while used<limit:
        ids=np.flatnonzero(first==0)
        if strategy=='frontier':ids=ids[:geometry_budget]
        ids=ids[~disabled[ids]][:limit-used]
        if not len(ids):break
        aa=a[ids];out=local.response(x,v,regs[ids],aa);visits[ids]+=1;used+=len(ids);rounds+=1
        threshold=1e-10*(1+abs(aa).sum((1,2)))
        check=(out['objective']>threshold)|~out['valid'];new=[]
        for j in np.flatnonzero(check):
            i=int(ids[j]);problem=IntegerProblem(x,v,aa[j]);value=problem.fixed(regs[i]);extra_exact+=1
            if value is not None and value<=0:continue
            first[i]=int(visits[i]);proof_credit[i]=aa[j];new.append(i)
            proofs.append(dict(index=i,step=int(visits[i]),mode=regs[i].tobytes().hex(),
                lower=None if value is None else problem.string(value),structural_empty=value is None,
                stage='continued',round=rounds,kind='local',source_index=None))
        norm2=(out['residual']**2).sum((1,2));valid=out['valid']&np.isfinite(out['objective'])&np.isfinite(norm2)&(norm2>0)
        unresolved=first[ids]==0;new_disabled=ids[unresolved&~valid].tolist();disabled[new_disabled]=True
        update=valid&unresolved;selected=ids[update]
        a[selected]+=((np.maximum(local.DELTA-out['objective'][update],0)/norm2[update])[:,None,None]*out['residual'][update])
        bad=selected[~np.isfinite(a[selected]).all((1,2))].tolist();disabled[bad]=True;new_disabled+=bad
        trace.append(dict(round=rounds,indices=ids.tolist(),certified=new,disabled=new_disabled,response_pairs=used))
        owned=[a,first,disabled,proof_credit,visits,base_first,base_disabled,base_a,base_visits,base_selected,
               ids,aa,threshold,check,norm2,valid,unresolved,update,selected]
        peak=max(peak,sum(t.nbytes for t in owned)+regs.nbytes+credit.nbytes+x.nbytes+v.nbytes+out['named_array_bytes_subtotal'])
    assert used==int(visits.sum())<=limit
    arrays=dict(first_step=first,proof_credit=proof_credit,final_credit=a,disabled=disabled,response_counts=visits,
        base_first_step=base_first,base_disabled=base_disabled,base_final_credit=base_a,base_selected=base_selected,
        base_response_counts=base_visits,
        selected_indices=np.flatnonzero(first==0)[:geometry_budget])
    return arrays,dict(strategy=strategy,steps=steps,geometry_budget=geometry_budget,budget_limit=limit,
        base_processed=processed,base_response_pairs=base_work,base_calls=base_calls,continuation_rounds=rounds,
        trace=trace,proofs=proofs,total_response_pairs=used,local_response_pairs=used,transfer_response_pairs=0,
        exact_calls=exact_calls+extra_exact,continued_exact_calls=extra_exact,directions=[],
        total_seconds=time.perf_counter()-started,query_targets_accessed=False,geometry_accessed=False,
        named_continuation_array_bytes_subtotal_max=peak,output_array_bytes=sum(t.nbytes for t in arrays.values()),
        input_generation_excluded=True,state_scope='Named continuation arrays and outputs separately; Python trace/objects and allocator peak not measured')

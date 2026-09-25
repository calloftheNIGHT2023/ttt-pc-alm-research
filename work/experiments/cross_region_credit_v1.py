"""357 causal cross-region exact-certificate propagation, not BP/LP."""
import time
import numpy as np
import region_conditioned_credit_v1 as local
from branch_image_chain_dyadic_v1 import IntegerProblem


def solve(x,v,regs,credit,*,propagate=True,steps=128):
    started=time.perf_counter();regs=np.asarray(regs,dtype=np.uint8);count,d,n=regs.shape
    if credit.shape!=(d,n) or not np.isfinite(credit).all() or steps<1:raise ValueError('Invalid finite credit/steps')
    scale=np.linalg.norm(credit);initial=credit/scale if scale>0 else credit.copy()
    a=np.broadcast_to(initial,regs.shape).copy();first=np.zeros(count,np.int32);disabled=np.zeros(count,bool)
    proof_credit=np.zeros_like(a);proofs=[];traces=[];directions=[];pending=[];seen=set()
    exact_calls=local_pairs=transfer_pairs=0;oracle_seconds=exact_seconds=update_seconds=transfer_seconds=0.;named_peak=0
    def certify(index,aa,step,source):
        nonlocal exact_calls,exact_seconds
        tick=time.perf_counter();problem=IntegerProblem(x,v,aa);value=problem.fixed(regs[index])
        exact_seconds+=time.perf_counter()-tick;exact_calls+=1
        if value is not None and value<=0:return False
        first[index]=step;proof_credit[index]=aa
        proof=dict(index=int(index),step=step,mode=regs[index].tobytes().hex(),
                   lower=None if value is None else problem.string(value),structural_empty=value is None,
                   source_index=source,kind='local' if source is None else 'transfer')
        proofs.append(proof)
        if propagate and source is None and value is not None and len(directions)<16 and aa.tobytes() not in seen:
            seen.add(aa.tobytes());item=dict(source_index=int(index),learned_step=step,credit=aa.copy())
            directions.append(item);pending.append(len(directions)-1)
        return True
    for step in range(1,steps+1):
        ids=np.flatnonzero((first==0)&~disabled)
        if len(ids):
            tick=time.perf_counter();aa=a[ids];out=local.response(x,v,regs[ids],aa)
            oracle_seconds+=time.perf_counter()-tick;local_pairs+=len(ids)
            values=out['objective'];residual=out['residual'];threshold=1e-10*(1+abs(aa).sum((1,2)))
            check=(values>threshold)|~out['valid']
            for index in np.flatnonzero(check):certify(int(ids[index]),aa[index],step,None)
            tick=time.perf_counter();norm2=(residual*residual).sum((1,2))
            usable=out['valid']&np.isfinite(values)&np.isfinite(norm2)&(norm2>0);unresolved=first[ids]==0
            disabled[ids[unresolved&~usable]]=True;update=usable&unresolved;selected=ids[update]
            a[selected]+=((np.maximum(local.DELTA-values[update],0.)/norm2[update])[:,None,None]*residual[update])
            disabled[selected[~np.isfinite(a[selected]).all((1,2))]]=True
            update_seconds+=time.perf_counter()-tick
            owned=[a,first,disabled,proof_credit,ids,aa,threshold,check,norm2,usable,unresolved,update]
            named_peak=max(named_peak,sum(t.nbytes for t in owned)+regs.nbytes+credit.nbytes+x.nbytes+v.nbytes+out['named_array_bytes_subtotal'])
        if propagate and (step%8==0 or step==steps):
            tick=time.perf_counter()
            for bi in pending:
                direction=directions[bi];ids=np.flatnonzero(first==0)
                direction['used_step']=step;direction['targets_tested']=len(ids)
                if not len(ids):continue
                aa=np.broadcast_to(direction['credit'],(len(ids),d,n));out=local.response(x,v,regs[ids],aa)
                transfer_pairs+=len(ids)
                rough=(out['objective']>1e-10*(1+abs(aa).sum((1,2))))|~out['valid']
                for index in np.flatnonzero(rough):certify(int(ids[index]),aa[index],step,int(direction['source_index']))
                named_peak=max(named_peak,a.nbytes+first.nbytes+disabled.nbytes+proof_credit.nbytes+regs.nbytes
                    +x.nbytes+v.nbytes+credit.nbytes+ids.nbytes+rough.nbytes+aa.nbytes+out['named_array_bytes_subtotal']
                    +sum(q['credit'].nbytes for q in directions))
            pending.clear();transfer_seconds+=time.perf_counter()-tick
        if step in local.PREFIXES or step==steps:
            traces.append(dict(step=step,rejected=int((first>0).sum()),disabled=int(disabled.sum()),
                               local_response_pairs=local_pairs,transfer_response_pairs=transfer_pairs,
                               elapsed_seconds=time.perf_counter()-started))
    arrays=dict(first_step=first,proof_credit=proof_credit,final_credit=a,disabled=disabled)
    return arrays,dict(proofs=proofs,traces=traces,directions=[dict(q,credit=q['credit'].tolist()) for q in directions],
        exact_calls=exact_calls,local_response_pairs=local_pairs,transfer_response_pairs=transfer_pairs,
        total_response_pairs=local_pairs+transfer_pairs,oracle_seconds=oracle_seconds,exact_seconds=exact_seconds,
        update_seconds=update_seconds,transfer_seconds_including_transfer_exact=transfer_seconds,
        total_seconds=time.perf_counter()-started,named_array_bytes_subtotal_max=named_peak,
        output_array_bytes=sum(q.nbytes for q in arrays.values()),propagate=propagate,steps=steps,
        query_targets_accessed=False,geometry_accessed=False,
        state_scope='Named NumPy subtotal; Python proof objects and allocator peak excluded')

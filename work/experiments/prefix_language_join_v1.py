"""368 support-only incremental observation join with explicit finite budgets."""
from math import prod
import time
import numpy as np
from branch_image_chain_dyadic_v1 import IntegerProblem
from support_language_chain_v1 import accepted_paths
from local_region_screen import contract


def join(x,v,*,ordering,depth=4,max_expanded=65536,max_states=20000,max_seconds=8.):
    begin=time.perf_counter();n=len(x)
    assert ordering in ['observed','small_first'] and x.shape==v.shape and n>0
    problem=IntegerProblem(x,v,np.zeros((depth,n)))
    languages=[np.array(accepted_paths(problem,i),np.uint8).reshape(-1,depth) for i in range(n)]
    language_seconds=time.perf_counter()-begin;counts=list(map(len,languages))
    order=list(range(n)) if ordering=='observed' else sorted(range(n),key=lambda i:(counts[i],i))
    partial=np.empty((1,depth,0),np.uint8);arrays={'language_'+str(i):p.copy() for i,p in enumerate(languages)}
    steps=[];expanded=0;reason=None;peak=1;generated_peak=1
    for k,obs in enumerate(order,1):
        possible=len(partial)*counts[obs]
        if expanded+possible>max_expanded:reason='expanded_limit_before_stage';break
        if time.perf_counter()-begin>=max_seconds:reason='seconds_before_stage';break
        blocks=[];removed5=removed20=done=0;interrupted=False
        for start in range(0,possible,1024):
            if time.perf_counter()-begin>=max_seconds:reason='seconds_at_batch_boundary';interrupted=True;break
            ids=np.arange(start,min(start+1024,possible));parent=ids//counts[obs];path=ids%counts[obs]
            regs=np.concatenate([partial[parent],languages[obs][path,:,None]],axis=2)
            xx=x[order[:k]];vv=v[order[:k]]
            skip=contract(xx,vv,regs,5);removed5+=int(skip.sum());regs=regs[~skip]
            if len(regs):
                skip=contract(xx,vv,regs,20);removed20+=int(skip.sum());regs=regs[~skip]
            if len(regs):blocks.append(regs)
            done+=len(ids);expanded+=len(ids)
            generated_peak=max(generated_peak,sum(len(b) for b in blocks))
            if sum(len(b) for b in blocks)>max_states:reason='state_limit_at_batch_boundary';interrupted=True;break
        row=dict(k=k,observation=obs,parent_count=len(partial),language_count=counts[obs],
            possible_expansions=possible,processed_expansions=done,rejected_c5=removed5,rejected_c20=removed20,
            retained=sum(len(b) for b in blocks),completed=not interrupted)
        steps.append(row)
        if interrupted:break
        partial=np.concatenate(blocks) if blocks else np.empty((0,depth,k),np.uint8)
        arrays['prefix_'+str(k)]=partial.copy();peak=max(peak,len(partial))
    complete=reason is None and len(steps)==n and all(s['completed'] for s in steps)
    if complete:
        regions=partial[:,:,np.argsort(order)].copy()
        regions=np.array(sorted(regions,key=lambda r:r.tobytes()),np.uint8).reshape(-1,depth,n)
    else:regions=np.empty((0,depth,n),np.uint8)
    arrays['regions']=regions
    meta=dict(completed=complete,stop_reason=reason,ordering=ordering,order=order,single_counts=counts,
        full_product=prod(counts),expanded=expanded,peak_completed_prefix_states=peak,steps=steps,
        peak_generated_prefix_states=generated_peak,
        max_expanded=max_expanded,max_states=max_states,max_seconds=max_seconds,
        language_seconds=language_seconds,total_seconds=time.perf_counter()-begin,
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        complete_stages=sum(s['completed'] for s in steps),query_targets_accessed=False,
        geometry_accessed=False,mother_or_credit_used=False,
        resource_scope='Component only; no final geometry or prediction; named return arrays, not process peak')
    return arrays,meta

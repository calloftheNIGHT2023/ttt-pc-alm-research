"""One analytic generalized-self-concordant step per credit response.

This is an application of a known damped Newton step, not a new generic
optimizer. No line search, LP, global BP, or query enters this module.
"""
import time
import numpy as np
from finite_credit_game import response,original
from credit_halfspace_projection import probabilities

STEPS=128
PREFIXES=(1,4,16,64,128)
DELTA=original.base.EPS+original.base.TOL


def update(logits,gains,delta=DELTA):
    """Status 0=updated, 1=target-limited, 2=zero variance, 3=nonfinite,
    4=already-at-target. All exceptional statuses remain unproved.
    """
    count,k=gains.shape;result=logits.copy();status=np.zeros(count,np.int8)
    beta=np.zeros(count);mean_count=np.zeros(count,np.int32);variance_count=np.zeros(count,np.int32)
    oldmean=np.full(count,np.nan);newmean=np.full(count,np.nan)
    finite=np.isfinite(gains).all(1)&np.isfinite(logits).all(1);status[~finite]=3
    low=gains.min(1);high=gains.max(1);span=high-low
    limited=finite&((high<=delta)|(span<=0));status[limited]=1
    ids=np.flatnonzero(finite&~limited)
    named=[result,status,beta,mean_count,variance_count,oldmean,newmean,finite,low,high,span,limited,ids]
    if len(ids):
        gg=(gains[ids]-low[ids,None])/span[ids,None];target=(delta-low[ids])/span[ids]
        weights=probabilities(logits[ids]);mu=(weights*gg).sum(1)
        var=(weights*(gg-mu[:,None])**2).sum(1);gap=target-mu
        mean_count[ids]+=1;variance_count[ids]+=1;oldmean[ids]=mu*span[ids]+low[ids]
        numerical=(var<=0)|~np.isfinite(var);status[ids[numerical]]=2
        satisfied=(gap<=0)&~numerical;status[ids[satisfied]]=4
        good=np.flatnonzero(~numerical&~satisfied)
        if len(good):
            # logaddexp avoids overflow in gap/variance, exactly the prescribed
            # log1p expression in real arithmetic, with no tunable coefficient.
            b=np.logaddexp(0.,np.log(gap[good])-np.log(var[good]))
            which=ids[good];beta[which]=b
            result[which]=logits[which]+b[:,None]*gg[good]
            result[which]-=result[which].max(1,keepdims=True)
            q=probabilities(result[which]);post=(q*gg[good]).sum(1)
            assert np.all(post>=mu[good]-1e-12) and np.all(post<=target[good]+1e-12)
            newmean[which]=post*span[which]+low[which];mean_count[which]+=1
            named.extend([b,which,q,post])
        named.extend([gg,target,weights,mu,var,gap,numerical,satisfied,good])
    return result,dict(status=status,beta=beta,mean_count=mean_count,variance_count=variance_count,
        oldmean=oldmean,newmean=newmean,named_bytes=sum(q.nbytes for q in named))


def solve(x,v,regs,bank):
    started=time.perf_counter();count,d,n=regs.shape;k=len(bank)
    assert k>0 and np.isfinite(bank).all() and np.max(abs(bank))<=1
    flat=bank.reshape(k,d*n);logits=np.zeros((count,k));best_weights=np.zeros_like(logits)
    best_credit=np.zeros((count,d,n));best=np.full(count,-np.inf)
    first=np.zeros(count,np.int32);best_step=np.zeros(count,np.int32)
    terminal=np.zeros(count,np.int8);stop_step=np.zeros(count,np.int32)
    oracle_count=np.zeros(count,np.int32);mean_count=np.zeros(count,np.int64);variance_count=np.zeros(count,np.int64)
    max_beta=np.zeros(count);prefix_values=np.full((count,len(PREFIXES)),-np.inf)
    proofs=[];traces=[];exact_calls=0;oracle_seconds=0.;update_seconds=0.;exact_seconds=0.;peak_named=0
    for step in range(1,STEPS+1):
        ids=np.flatnonzero(terminal==0)
        if len(ids):
            clock=time.perf_counter();weights=probabilities(logits[ids]);a=(weights@flat).reshape(len(ids),d,n)
            update_seconds+=time.perf_counter()-clock
            clock=time.perf_counter();out=response(x,v,regs[ids],a)
            oracle_seconds+=time.perf_counter()-clock;oracle_count[ids]+=1
            assert out['valid'].all() and np.max(abs(out['residual']))<=1+1e-12
            values=out['objective'];improved=values>best[ids];ii=ids[improved]
            best[ii]=values[improved];best_weights[ii]=weights[improved];best_credit[ii]=a[improved];best_step[ii]=step
            for local in np.flatnonzero(values>0):
                index=int(ids[local]);clock=time.perf_counter();exact=original.exact_optimum(x,v,regs[index],a[local])
                exact_seconds+=time.perf_counter()-clock;exact_calls+=1
                if exact['positive']:
                    assert 'empty_layer' not in exact
                    terminal[index]=1;stop_step[index]=step;first[index]=step
                    best_weights[index]=weights[local];best_credit[index]=a[local];best[index]=values[local];best_step[index]=step
                    proofs.append(dict(index=index,step=step,pattern=regs[index].tobytes().hex(),exact=exact))
            clock=time.perf_counter();todo=np.flatnonzero(terminal[ids]==0);moment_bytes=0
            gains=out['residual'][todo].reshape(len(todo),d*n)@flat.T
            if len(todo):
                target_ids=ids[todo];new,meta=update(logits[target_ids],gains)
                logits[target_ids]=new;mean_count[target_ids]+=meta['mean_count'];variance_count[target_ids]+=meta['variance_count']
                max_beta[target_ids]=np.maximum(max_beta[target_ids],meta['beta'])
                for status,code in [(1,2),(2,3),(3,4),(4,5)]:
                    retired=target_ids[meta['status']==status];terminal[retired]=code;stop_step[retired]=step
                moment_bytes=meta['named_bytes']+new.nbytes
            update_seconds+=time.perf_counter()-clock
            owned=[logits,best_weights,best_credit,best,first,best_step,terminal,stop_step,oracle_count,mean_count,variance_count,
                max_beta,prefix_values,ids,weights,a,improved,ii,todo,gains]
            peak_named=max(peak_named,bank.nbytes+regs.nbytes+x.nbytes+v.nbytes+sum(q.nbytes for q in owned)
                +out['named_array_bytes_subtotal']+moment_bytes)
        if step in PREFIXES:
            prefix_values[:,PREFIXES.index(step)]=best
            traces.append(dict(step=step,positive=int(np.sum(first>0)),oracle_pairs=int(oracle_count.sum()),
                mean_evaluations=int(mean_count.sum()),variance_evaluations=int(variance_count.sum()),elapsed_seconds=time.perf_counter()-started))
    arrays=dict(credit=best_credit,weights=best_weights,first_step=first,best_step=best_step,best_value=best,prefix_values=prefix_values,
        terminal=terminal,stop_step=stop_step,oracle_count=oracle_count,mean_count=mean_count,variance_count=variance_count,max_beta=max_beta)
    return arrays,dict(regions=count,directions=k,delta=DELTA,steps=STEPS,proofs=proofs,traces=traces,positive=len(proofs),exact_calls=exact_calls,
        oracle_pairs=int(oracle_count.sum()),mean_evaluations=int(mean_count.sum()),variance_evaluations=int(variance_count.sum()),
        target_limited=int(np.sum(terminal==2)),zero_variance=int(np.sum(terminal==3)),nonfinite_failures=int(np.sum(terminal==4)),
        already_at_target=int(np.sum(terminal==5)),oracle_seconds=oracle_seconds,update_seconds=update_seconds,exact_seconds=exact_seconds,
        total_seconds=time.perf_counter()-started,named_array_bytes_subtotal_max=peak_named,output_array_bytes=sum(q.nbytes for q in arrays.values()),
        memory_scope='Named-array subtotal, not allocator peak; all two mean plus one variance evaluations charged',
        terminal_codes={'0':'budget','1':'exact_positive','2':'target_limited_not_a_certificate','3':'zero_variance','4':'nonfinite','5':'already_at_target_unproved'})

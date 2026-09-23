"""KL halfspace projections on a fixed credit simplex; no LP or global BP."""
import time
import numpy as np
from finite_credit_game import response, original

STEPS = 128
PREFIXES = (1, 4, 16, 64, 128)
DELTA = original.base.EPS + original.base.TOL
DOUBLINGS = 64
BISECTIONS = 48


def probabilities(logits):
    shifted = logits - logits.max(1, keepdims=True)
    out = np.exp(shifted)
    return out / out.sum(1, keepdims=True)


def project(logits, gains, delta=DELTA):
    """Batch I-projections. Status 0=ok, 1=limited, 2=bracket, 3=nonfinite.

    Mean-evaluation counts include the final projection check. Normalized beta
    is stored; the exponent uses (g-min(g))/range(g), an exact affine invariance
    in ideal arithmetic, not an assumption about floating convergence.
    """
    count, k = gains.shape
    result = logits.copy(); status = np.zeros(count, np.int8)
    evaluations = np.zeros(count, np.int32); beta = np.zeros(count)
    residual = np.full(count, np.nan)
    finite = np.isfinite(gains).all(1) & np.isfinite(logits).all(1)
    status[~finite] = 3
    low = gains.min(1); high = gains.max(1); span = high-low
    limited = finite & ((high <= delta) | (span <= 0))
    status[limited] = 1
    ids = np.flatnonzero(finite & ~limited)
    named_peak = sum(q.nbytes for q in [result,status,evaluations,beta,residual,finite,low,high,span,limited,ids])
    if not len(ids):
        return result, dict(status=status,evaluations=evaluations,beta=beta,residual=residual,named_bytes=named_peak)
    gg = (gains[ids]-low[ids,None])/span[ids,None]
    target = (delta-low[ids])/span[ids]
    logs = logits[ids]; lo = np.zeros(len(ids)); hi = np.ones(len(ids))
    def means(which, value):
        pp = probabilities(logs[which] + value[:,None]*gg[which])
        evaluations[ids[which]] += 1
        return (pp*gg[which]).sum(1)
    current = (probabilities(logs)*gg).sum(1); evaluations[ids] += 1
    needs = current < target
    hi[~needs] = 0
    which = np.flatnonzero(needs); val = np.full(len(ids),np.inf)
    val[which] = means(which,hi[which])
    for _ in range(DOUBLINGS):
        pending = np.flatnonzero(needs & (val < target))
        if not len(pending): break
        lo[pending] = hi[pending]; hi[pending] *= 2
        val[pending] = means(pending,hi[pending])
    failed = needs & ((val < target) | ~np.isfinite(val))
    status[ids[failed]] = 2
    work = np.flatnonzero(needs & ~failed)
    for _ in range(BISECTIONS):
        if not len(work): break
        mid = (lo[work]+hi[work])/2; value = means(work,mid)
        below = value < target[work]
        lo[work[below]] = mid[below]; hi[work[~below]] = mid[~below]
    good = np.flatnonzero(~failed)
    result[ids[good]] = logs[good] + hi[good,None]*gg[good]
    result[ids[good]] -= result[ids[good]].max(1,keepdims=True)
    beta[ids[good]] = hi[good]
    ww = probabilities(result[ids[good]])
    residual[ids[good]] = (ww*gains[ids[good]]).sum(1)-delta
    evaluations[ids[good]] += 1
    named_peak += sum(q.nbytes for q in [gg,target,logs,lo,hi,current,needs,which,val,failed,work,good,ww])
    return result,dict(status=status,evaluations=evaluations,beta=beta,residual=residual,named_bytes=named_peak)


def solve(x,v,regs,bank):
    started=time.perf_counter();count,d,n=regs.shape;k=len(bank)
    assert k>0 and np.isfinite(bank).all() and np.max(abs(bank))<=1
    flat=bank.reshape(k,d*n);logits=np.zeros((count,k));best_weights=np.zeros_like(logits)
    best_credit=np.zeros((count,d,n));best=np.full(count,-np.inf)
    first=np.zeros(count,np.int32);best_step=np.zeros(count,np.int32)
    terminal=np.zeros(count,np.int8);stop_step=np.zeros(count,np.int32)
    oracle_count=np.zeros(count,np.int32);mean_count=np.zeros(count,np.int64)
    max_beta=np.zeros(count);max_root_error=np.zeros(count)
    prefix_values=np.full((count,len(PREFIXES)),-np.inf)
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
            clock=time.perf_counter();todo=np.flatnonzero(terminal[ids]==0);root_bytes=0
            gains=out['residual'][todo].reshape(len(todo),d*n)@flat.T
            if len(todo):
                target_ids=ids[todo]
                new,meta=project(logits[target_ids],gains)
                logits[target_ids]=new;mean_count[target_ids]+=meta['evaluations']
                max_beta[target_ids]=np.maximum(max_beta[target_ids],meta['beta'])
                max_root_error[target_ids]=np.maximum(max_root_error[target_ids],np.nan_to_num(abs(meta['residual']),nan=0.))
                for status,code in [(1,2),(2,3),(3,4)]:
                    retired=target_ids[meta['status']==status];terminal[retired]=code;stop_step[retired]=step
                root_bytes=meta['named_bytes']+new.nbytes
            update_seconds+=time.perf_counter()-clock
            owned=[logits,best_weights,best_credit,best,first,best_step,terminal,stop_step,oracle_count,mean_count,
                   max_beta,max_root_error,prefix_values,ids,weights,a,improved,ii,todo,gains]
            peak_named=max(peak_named,bank.nbytes+regs.nbytes+x.nbytes+v.nbytes+sum(q.nbytes for q in owned)
                           +out['named_array_bytes_subtotal']+root_bytes)
        if step in PREFIXES:
            prefix_values[:,PREFIXES.index(step)]=best
            traces.append(dict(step=step,positive=int(np.sum(first>0)),target_limited=int(np.sum(terminal==2)),
                oracle_pairs=int(oracle_count.sum()),mean_evaluations=int(mean_count.sum()),elapsed_seconds=time.perf_counter()-started))
    arrays=dict(credit=best_credit,weights=best_weights,first_step=first,best_step=best_step,best_value=best,
        prefix_values=prefix_values,terminal=terminal,stop_step=stop_step,oracle_count=oracle_count,mean_count=mean_count,
        max_beta=max_beta,max_root_error=max_root_error)
    return arrays,dict(regions=count,directions=k,delta=DELTA,steps=STEPS,doublings=DOUBLINGS,bisections=BISECTIONS,
        proofs=proofs,traces=traces,positive=len(proofs),exact_calls=exact_calls,oracle_pairs=int(oracle_count.sum()),
        mean_evaluations=int(mean_count.sum()),target_limited=int(np.sum(terminal==2)),bracket_failures=int(np.sum(terminal==3)),
        nonfinite_failures=int(np.sum(terminal==4)),max_root_error=float(max_root_error.max(initial=0)),
        oracle_seconds=oracle_seconds,update_seconds=update_seconds,exact_seconds=exact_seconds,total_seconds=time.perf_counter()-started,
        named_array_bytes_subtotal_max=peak_named,output_array_bytes=sum(q.nbytes for q in arrays.values()),
        memory_scope='Named NumPy arrays subtotal, not allocator peak; root mean and exponentiation work charged',
        terminal_codes={'0':'budget','1':'exact_positive','2':'target_limited_not_a_certificate','3':'bracket_failed','4':'nonfinite'})

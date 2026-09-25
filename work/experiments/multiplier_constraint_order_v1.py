"""370 constraint ordering; multiplier sensitivity is a heuristic, never a cut."""
import time
from math import log
import numpy as np
import cold_stagnation_switch as cold
import prefix_language_join_v1 as prefix
from branch_image_chain_dyadic_v1 import IntegerProblem
from support_language_chain_v1 import accepted_paths
from local_region_screen import contract
from test_region_conditioned_credit_v1 import guarded

NAMES=['observed','small_first','farthest_x','random','pair_compatibility','alm_dual','alm_residual','alm_bp_score','pc_residual']


def pair_order(x,v):
    n=len(x);p=IntegerProblem(x,v,np.zeros((4,n)))
    languages=[np.array(accepted_paths(p,i),np.uint8).reshape(-1,4) for i in range(n)]
    counts=np.array(list(map(len,languages)));assert np.all(counts>0)
    paircounts=np.zeros((n,n),int);expanded=0
    for i in range(n):
        for j in range(i+1,n):
            ii,jj=np.meshgrid(np.arange(counts[i]),np.arange(counts[j]),indexing='ij')
            regs=np.stack([languages[i][ii.ravel()],languages[j][jj.ravel()]],axis=2)
            expanded+=len(regs);regs=regs[~contract(x[[i,j]],v[[i,j]],regs,5)]
            if len(regs):regs=regs[~contract(x[[i,j]],v[[i,j]],regs,20)]
            paircounts[i,j]=paircounts[j,i]=len(regs)
    order=[min(range(n),key=lambda i:(counts[i],i))]
    while len(order)<n:
        remaining=set(range(n))-set(order)
        def score(i):return log(int(counts[i]))+sum(log(max(1,int(paircounts[i,j]))/int(counts[i]*counts[j])) for j in order)
        order.append(min(remaining,key=lambda i:(score(i),i)))
    return np.array(order),dict(pair_counts=paircounts,language_counts=counts),dict(pair_candidate_rows=expanded)


def order(x,v,seed,name):
    begin=time.perf_counter();n=len(x);arrays={};meta=dict(pair_candidate_rows=0,local_sweeps=0,solver_state_bytes=0)
    if name=='observed':ranking=np.arange(n)
    elif name=='random':ranking=np.random.default_rng(np.random.SeedSequence([370111,seed,n])).permutation(n)
    elif name=='farthest_x':
        selected=[int(np.argmin(x))]
        while len(selected)<n:
            remaining=set(range(n))-set(selected)
            selected.append(min(remaining,key=lambda i:(-min(abs(x[i]-x[j]) for j in selected),i)))
        ranking=np.array(selected)
    elif name in ['small_first','pair_compatibility']:
        if name=='pair_compatibility':ranking,arrays,extra=pair_order(x,v);meta.update(extra)
        else:
            p=IntegerProblem(x,v,np.zeros((4,n)));scores=np.array([len(accepted_paths(p,i)) for i in range(n)])
            arrays['language_counts']=scores;ranking=np.array(sorted(range(n),key=lambda i:(scores[i],i)))
    else:
        method='pc' if name=='pc_residual' else 'alm'
        state=cold.Local(cold.starts(),x,v,method)
        for _ in range(64):state.step()
        residual=[];previous=x
        for j in range(4):residual.append(state.h[j]-cold.base.g(previous+state.b[:,j,None]));previous=state.h[j]
        residual=np.array(residual)
        if name=='alm_dual':scores=abs(state.u).sum(axis=0).mean(axis=0)
        elif name in ['alm_residual','pc_residual']:scores=abs(residual).sum(axis=0).mean(axis=0)
        else:
            assert name=='alm_bp_score'
            _,r,jac,_=cold.bp.evaluate(state.b,x,v)
            scores=abs(jac*r[:,:,None]).sum(axis=2).mean(axis=0)
        ranking=np.array(sorted(range(n),key=lambda i:(-scores[i],i)))
        arrays.update(order_scores=scores,order_b=state.b.copy(),order_h=state.h.copy(),order_u=state.u.copy(),order_residual=residual)
        meta.update(local_sweeps=64,restarts=17,solver_state_bytes=state.numeric_state_bytes(),solver=method)
    arrays['support_order']=ranking
    meta.update(ordering=name,ordering_seconds=time.perf_counter()-begin,global_bp_score=name=='alm_bp_score')
    return ranking,arrays,meta


def fit(x,v,seed,name):
    begin=time.perf_counter()
    if name=='alm_bp_score':ranking,arrays,meta=order(x,v,seed,name)
    else:
        with guarded():ranking,arrays,meta=order(x,v,seed,name)
    remaining=8.-(time.perf_counter()-begin)
    with guarded():a,m=prefix.join(x[ranking],v[ranking],ordering='observed',max_seconds=max(0.,remaining))
    a.update(arrays);a.update(x_observed=x[ranking].copy(),v_observed=v[ranking].copy(),original_x=x.copy(),original_v=v.copy())
    m.update(meta,total_component_seconds=time.perf_counter()-begin,query_targets_accessed=False,
        returned_array_bytes=sum(t.nbytes for t in a.values()),generation_charged=True,
        resources_matched=False,geometry_and_query_readout_excluded=True)
    return a,m

"""319 frozen entry policies and grouped real-Local continuations."""
from fractions import Fraction as F
import time
import numpy as np
import cold_stagnation_switch as cold

VARIANTS=['original64','corrected64','escaped64','corrected_virtual','escaped_matched']


def entries(line,escape):
    original=np.array([float(F(z)) for z in line['point']]);result=escape['result']
    corrected=np.array([float(F(z)) for z in line['proposal']['q']]) if result['applicable'] else original.copy()
    found=result['status']=='first_primal_exit_found'
    k=result['first_exit_input_phase']+1 if found else 0
    assert 0<=k<=27
    escaped=np.array(escape['floating_exit_output'],dtype=float) if found else corrected.copy()
    starts=dict(original64=original,corrected64=corrected,escaped64=escaped,
                corrected_virtual=corrected.copy(),escaped_matched=escaped.copy())
    horizons=dict(original64=64,corrected64=64,escaped64=64,corrected_virtual=64+k,escaped_matched=64-k)
    return starts,horizons,k


def forward(b,x):
    b=np.asarray(b);values=np.broadcast_to(x,b.shape[:-1]+(len(x),)).copy();codes=[]
    for j in range(b.shape[-1]):
        z=values+b[...,j,None];codes.append(np.searchsorted(cold.base.KNOTS,z,side='right').astype(np.uint8))
        values=cold.base.g(z)
    return np.concatenate(codes,axis=-1),values


def rollout(points,horizons,x,v,method):
    points=np.asarray(points,dtype=float);horizons=np.asarray(horizons,dtype=int)
    r,width=points.shape;n=len(x);d=width//(1+2*n);dim=d+d*n
    assert width==d+2*d*n and len(horizons)==r and np.all(horizons>=0)
    maximum=int(horizons.max());b=np.empty((maximum+1,r,d));h=np.empty((maximum+1,d,r,n));u=np.empty_like(h)
    seconds=dict(initialization=0.,local_steps=0.,trace_assignment=0.,forward_readout=0.)
    batches=0;states=0;max_named_bytes=0
    for horizon in np.unique(horizons):
        ids=np.flatnonzero(horizons==horizon);initial=points[ids]
        tick=time.perf_counter();solver=cold.Local(initial[:,:d],x,v,method)
        solver.h=initial[:,d:dim].reshape(len(ids),d,n).transpose(1,0,2).copy()
        solver.u=initial[:,dim:].reshape(len(ids),d,n).transpose(1,0,2).copy()
        seconds['initialization']+=time.perf_counter()-tick;max_named_bytes=max(max_named_bytes,solver.numeric_state_bytes())
        for t in range(int(horizon)+1):
            if t:
                tick=time.perf_counter();solver.step();seconds['local_steps']+=time.perf_counter()-tick
                batches+=1;states+=len(ids)
            tick=time.perf_counter();b[t][ids]=solver.b;h[t][:,ids]=solver.h;u[t][:,ids]=solver.u
            seconds['trace_assignment']+=time.perf_counter()-tick
        # Padded storage repeats the final state but is never counted as a step.
        if horizon<maximum:
            b[horizon+1:,ids]=solver.b[None]
            h[horizon+1:][:,:,ids]=solver.h[None];u[horizon+1:][:,:,ids]=solver.u[None]
    tick=time.perf_counter();codes,values=forward(b,x);seconds['forward_readout']=time.perf_counter()-tick
    arrays=dict(b=b,h=h,u=u,forward_codes=codes,forward_values=values,horizons=horizons,initial_points=points)
    meta=dict(valid_local_updates=states,batched_local_calls=batches,seconds=seconds,
              maximum_live_named_solver_bytes=max_named_bytes,
              retained_diagnostic_array_bytes=sum(a.nbytes for a in arrays.values()),
              memory_scope='Named Local arrays and separate archive arrays; not process peak/workspaces.')
    assert states==int(horizons.sum())
    return arrays,meta

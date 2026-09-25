"""304: real support-observable stasis detection and batched retained-dual continuation.

Single-thread only. No archive, reference, certificate or target file access in fit.
The shared original trajectory, support-best point and geometry/readout are retained.
"""
import time
import numpy as np
import certificate_activity_attribution as original


def prediction(b,x):
    y=np.broadcast_to(x,(len(b),len(x)))
    for j in range(4):y=original.cold.base.g(y+b[:,j,None])
    return y


def fit(x,v,q,seed,rule='forward_stasis',trace=False):
    assert rule in ['none','forward_stasis'] and original.cold.base.BOUND==.12
    cfg=next(dict(c) for c in original.CONFIGS if c['name']=='credit_control_probe33')
    local_class,original_modes=original.cold.Local,original.modes
    created=0;previous={};buffers=[];locations=[];ready=False;flushed=False
    observer_seconds=shadow_seconds=mode_seconds=0.
    max_buffer_bytes=max_extra_numeric_bytes=0
    selected_count=0;shadow_mode_count=0;trace_arrays={}

    class Observed(local_class):
        def __init__(self,*args,**kwargs):
            nonlocal created
            super().__init__(*args,**kwargs)
            self.phase=created;created+=1

        def step(self):
            nonlocal ready,observer_seconds,max_buffer_bytes,max_extra_numeric_bytes,selected_count
            if self.phase==0 or rule=='none':return super().step()
            assert self.phase in [1,2] and self.method=='alm'
            tick=time.perf_counter()
            y=prediction(self.b,self.x)
            prev=previous.get(self.phase)
            if prev is None and self.phase==2:
                assert 1 in previous
                before=previous[1]
                prev=dict(b=before['b'][:1],h=before['h'][:,:1],y=before['y'][:1])
            if prev is not None:
                assert prev['y'].shape==y.shape
                errors=np.max(abs(y-self.v),axis=1)
                residual=np.stack([self.h[j]-original.cold.base.g((self.x if j==0 else self.h[j-1])+self.b[:,j,None]) for j in range(4)])
                eligible=(np.max(abs(y-prev['y']),axis=1)<=1e-8)&(errors>.001001)&(np.max(abs(residual),axis=(0,2))>1e-6)
                ids=np.flatnonzero(eligible)
                if len(ids):
                    buffers.append(dict(b=self.b[ids].copy(),h=self.h[:,ids].copy(),u=self.u[:,ids].copy(),best=self.best[ids].copy()))
                    locations.extend([[self.phase-1,self.step_count+1,int(i)] for i in ids])
                    selected_count+=len(ids)
            previous[self.phase]=dict(b=self.b.copy(),h=self.h.copy(),y=y.copy())
            buffer_bytes=sum(a.nbytes for item in buffers for a in item.values())
            prev_bytes=sum(a.nbytes for item in previous.values() for a in item.values())
            max_buffer_bytes=max(max_buffer_bytes,buffer_bytes)
            max_extra_numeric_bytes=max(max_extra_numeric_bytes,buffer_bytes+prev_bytes)
            observer_seconds+=time.perf_counter()-tick
            super().step()
            if self.phase==2 and self.step_count==32:ready=True

    def augmented_modes(xx,bank):
        nonlocal flushed,shadow_seconds,mode_seconds,max_extra_numeric_bytes,shadow_mode_count
        ans=original_modes(xx,bank)
        if not ready or flushed:return ans
        flushed=True
        if not buffers:
            if trace:
                trace_arrays.update(shadow_locations=np.empty((0,3),dtype=int),shadow_b=np.empty((65,0,4)),
                    shadow_initial_b=np.empty((0,4)),shadow_initial_h=np.empty((4,0,len(x))),
                    shadow_initial_u=np.empty((4,0,len(x))),shadow_initial_best=np.empty((0,4)))
            return ans
        tick=time.perf_counter()
        b=np.concatenate([s['b'] for s in buffers],axis=0)
        h=np.concatenate([s['h'] for s in buffers],axis=1)
        u=np.concatenate([s['u'] for s in buffers],axis=1)
        best=np.concatenate([s['best'] for s in buffers],axis=0)
        shadow=local_class(b,x,v,'alm')
        shadow.h=h.copy();shadow.u=u.copy();shadow.best=best.copy()
        shadow.errors,shadow.moves=original.cold.base.score(best,x,v,np.zeros(4))
        shadow_seconds+=time.perf_counter()-tick
        shadow_modes={}
        def collect_shadow():
            nonlocal mode_seconds
            start=time.perf_counter()
            for key,representative in original_modes(xx,shadow.b).items():shadow_modes.setdefault(key,representative.copy())
            mode_seconds+=time.perf_counter()-start
        collect_shadow();history=[b.copy()] if trace else None
        for _ in range(64):
            start=time.perf_counter();shadow.step();shadow_seconds+=time.perf_counter()-start
            collect_shadow()
            if trace:history.append(shadow.b.copy())
        for key,representative in shadow_modes.items():ans.setdefault(key,representative)
        shadow_mode_count=len(shadow_modes)
        named=sum(a.nbytes for a in [b,h,u,best])+shadow.numeric_state_bytes()
        prev_bytes=sum(a.nbytes for item in previous.values() for a in item.values())
        max_extra_numeric_bytes=max(max_extra_numeric_bytes,max_buffer_bytes+prev_bytes+named+
            sum(a.nbytes for a in shadow_modes.values()))
        if trace:
            trace_arrays.update(shadow_locations=np.array(locations,dtype=int).reshape(-1,3),shadow_b=np.array(history),
                shadow_initial_b=b,shadow_initial_h=h,shadow_initial_u=u,shadow_initial_best=best,
                shadow_final_h=shadow.h.copy(),shadow_final_u=shadow.u.copy())

        return ans

    begin=time.perf_counter()
    try:
        original.cold.Local,original.modes=Observed,augmented_modes
        arrays,metadata=original.fit(cfg,x,v,q,seed,trace=trace)
        assert created==3
        assert rule=='none' or (ready and flushed)
    finally:
        original.cold.Local,original.modes=local_class,original_modes
    if trace:arrays.update(trace_arrays)
    metadata.update(online_stasis_rule=rule,shadow_family='alm_keep',shadow_horizon=64,
        selected_states=selected_count,shadow_state_steps=64*selected_count,shadow_visited_modes=shadow_mode_count,
        observer_seconds=observer_seconds,shadow_solver_seconds=shadow_seconds,shadow_mode_collection_seconds=mode_seconds,
        selected_state_buffer_numeric_bytes=max_buffer_bytes,extra_numeric_state_subtotal=max_extra_numeric_bytes,
        extra_state_scope='buffers, previous support state, merged state, named solver arrays and shadow representatives; excludes temporaries, Python objects, diagnostic trace and original state',
        wrapped_complete_seconds=time.perf_counter()-begin,diagnostic_trace_enabled=trace,
        selected_using_global_bp=False,global_bp_used=False,query_targets_accessed=False,
        archive_or_reference_access_in_fit=False)
    return arrays,metadata

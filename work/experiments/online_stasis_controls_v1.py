"""306: real support-only online controls; no archive/teacher access in fit."""
import time
import numpy as np
import certificate_activity_attribution as original
import conditioned_mode_geometry as conditioned
import batched_bp_discovery as bp
from online_stasis_memory_v1 import prediction

FAMILIES = ['alm_keep','alm_reset','nodual','pc','adam','alm_random_sign','alm_residual',
            'adam_perturb_001','adam_perturb_004']


def fit(x,v,q,seed,family='alm_keep',trace=False):
    assert family in FAMILIES and original.cold.base.BOUND == .12
    cfg = next(dict(c) for c in original.CONFIGS if c['name']=='credit_control_probe33')
    local_class, original_modes = original.cold.Local, original.modes
    created=0; previous={}; buffers=[]; locations=[]; ready=False; flushed=False
    observer_seconds=shadow_seconds=mode_seconds=0.
    max_buffer_bytes=max_extra_numeric_bytes=0
    selected_count=shadow_mode_count=0; trace_arrays={}; solver_metadata={}

    class Observed(local_class):
        def __init__(self,*args,**kwargs):
            nonlocal created
            super().__init__(*args,**kwargs)
            self.phase=created; created+=1

        def step(self):
            nonlocal ready,observer_seconds,max_buffer_bytes,max_extra_numeric_bytes,selected_count
            if self.phase==0:return super().step()
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
        nonlocal flushed,shadow_seconds,mode_seconds,max_extra_numeric_bytes,shadow_mode_count,solver_metadata
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
        start_b=b.copy(); start_u=u.copy()
        if family in ['alm_reset','nodual','pc']:
            start_u=np.zeros_like(u)
        elif family=='alm_random_sign':
            signs=np.stack([np.random.default_rng(np.random.SeedSequence([303911,seed,p,t,o,0])).choice([-1.,1.],size=(4,len(x))) for p,t,o in locations],axis=1)
            start_u=u*signs
        elif family=='alm_residual':
            residual=np.stack([h[j]-original.cold.base.g((x if j==0 else h[j-1])+b[:,j,None]) for j in range(4)])
            unorm=np.sqrt(np.sum(u*u,axis=(0,2)));rnorm=np.sqrt(np.sum(residual*residual,axis=(0,2)))
            scale=np.divide(unorm,rnorm,out=np.zeros_like(unorm),where=(unorm>0)&(rnorm>0))
            start_u=residual*scale[None,:,None]
        elif family.startswith('adam_perturb'):
            signs=np.array([np.random.default_rng(np.random.SeedSequence([303911,seed,p,t,o,1])).choice([-1.,1.],size=4) for p,t,o in locations])
            amplitude=.01 if family=='adam_perturb_001' else .04
            start_b=np.clip(b+amplitude*signs,-.12,.12)
        shadow_modes={}; history=[] if trace else None
        def collect(points):
            nonlocal mode_seconds
            tick=time.perf_counter()
            for key, representative in original_modes(xx,points).items():
                shadow_modes.setdefault(key,representative.copy())
            if trace:history.append(points.copy())
            mode_seconds+=time.perf_counter()-tick
        shadow_seconds+=time.perf_counter()-tick
        if family.startswith('adam'):
            saved_evaluate=bp.evaluate
            def evaluate(points,xx,vv,with_jacobian=True):
                collect(points)
                return saved_evaluate(points,xx,vv,with_jacobian)
            tick=time.perf_counter(); prior_mode_seconds=mode_seconds
            try:
                bp.evaluate=evaluate
                _,solver_metadata=bp.refine(start_b,x,v,np.zeros(4),solver='adam',steps=64)
            finally:
                bp.evaluate=saved_evaluate
            shadow_seconds+=time.perf_counter()-tick-(mode_seconds-prior_mode_seconds)
            live_bytes=solver_metadata['major_arrays_bytes_subtotal']
        else:
            tick=time.perf_counter()
            method='alm' if family.startswith('alm_') else family
            shadow=local_class(start_b,x,v,method)
            shadow.h=h.copy();shadow.u=start_u.copy();shadow.best=best.copy()
            shadow.errors,shadow.moves=original.cold.base.score(best,x,v,np.zeros(4))
            shadow_seconds+=time.perf_counter()-tick
            collect(shadow.b)
            for _ in range(64):
                tick=time.perf_counter();shadow.step();shadow_seconds+=time.perf_counter()-tick
                collect(shadow.b)
            live_bytes=shadow.numeric_state_bytes()
            solver_metadata=dict(named_live_state_bytes=live_bytes)
            if trace:
                trace_arrays.update(shadow_final_h=shadow.h.copy(),shadow_final_u=shadow.u.copy())
        for key,representative in shadow_modes.items():ans.setdefault(key,representative)
        shadow_mode_count=len(shadow_modes)
        named=sum(a.nbytes for a in [b,h,u,best,start_b,start_u])+live_bytes
        prev_bytes=sum(a.nbytes for item in previous.values() for a in item.values())
        max_extra_numeric_bytes=max(max_extra_numeric_bytes,max_buffer_bytes+prev_bytes+named+
                                   sum(a.nbytes for a in shadow_modes.values()))
        if trace:
            assert len(history)==65
            trace_arrays.update(shadow_locations=np.array(locations,dtype=int).reshape(-1,3),shadow_b=np.array(history),
                shadow_initial_b=b,shadow_initial_h=h,shadow_initial_u=u,shadow_initial_best=best,
                shadow_effective_initial_b=start_b,shadow_effective_initial_u=start_u)
        return ans

    begin=time.perf_counter();repairs=[]
    try:
        original.cold.Local,original.modes=Observed,augmented_modes
        with conditioned.geometry_scope(repairs):
            arrays,metadata=original.fit(cfg,x,v,q,seed,trace=trace)
        assert created==3 and ready and flushed
    finally:
        original.cold.Local,original.modes=local_class,original_modes
    if trace:arrays.update(trace_arrays)
    metadata.update(online_stasis_rule='forward_stasis',shadow_family=family,shadow_horizon=64,
        selected_states=selected_count,shadow_state_steps=64*selected_count,shadow_visited_modes=shadow_mode_count,
        observer_seconds=observer_seconds,shadow_solver_seconds=shadow_seconds,shadow_mode_collection_seconds=mode_seconds,
        selected_state_buffer_numeric_bytes=max_buffer_bytes,extra_numeric_state_subtotal=max_extra_numeric_bytes,
        extra_state_scope='Named array subtotal; excludes direction temporaries, Python objects, diagnostic trace and original state; not peak',
        shadow_solver_metadata=solver_metadata,diagnostic_trace_enabled=trace,
        selected_using_global_bp=False,global_bp_used=family.startswith('adam') and selected_count>0,
        query_targets_accessed=False,archive_or_reference_access_in_fit=False,
        geometry_repair_count=len(repairs),geometry_repair_log=repairs,
        charged_complete_seconds=time.perf_counter()-begin)
    return arrays,metadata

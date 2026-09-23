"""Same-discovery posterior materialization with explicitly charged credit.

Local-only never creates BP observers. The BP-local union is a diagnostic
comparator, not a no-global-BP implementation. Geometry is the strong shared
LP/convex-hull sampler. No evaluator quantities enter any method below.
"""
import time
import numpy as np
import rejection_credit_capture as capture
import parallelotope_sampler as proposal
import neighbor_mode_memory as geometry
import region_posterior_memory as posterior

METHODS=('c20','local','bp_residual','bp_local_union','geometry')


def prepare(x,v,cfg,method):
    if method not in METHODS:raise ValueError(method)
    start=time.perf_counter()
    if method in ('bp_residual','bp_local_union'):result,roles=capture.capture(x,v,cfg,True)
    else:
        result=capture.model.prepare(x,v,cfg);roles={'local':result[3]}
    capture_seconds=time.perf_counter()-start
    start=time.perf_counter();regs=capture.pool(result);regs=regs[~capture.model.conflict.screen.contract(x,v,regs,20)]
    contraction_seconds=time.perf_counter()-start
    matrices=[];rhs=[];boxes=[];polys=[];poly_indices=[];trace=[];start=time.perf_counter()
    for i,reg in enumerate(regs):
        p,c,a,r=geometry.pattern_matrix(x,v,reg);matrices.append(a);rhs.append(r)
        if method=='geometry':
            poly,note=posterior.polytope(a,r);trace.append(note)
            if poly is not None:polys.append(poly);poly_indices.append(i)
        else:boxes.append(proposal.make_box(p,c,v))
    construction_seconds=time.perf_counter()-start
    mask=np.zeros(len(regs),bool);proofs={};start=time.perf_counter()
    if method!='geometry':
        volumes=np.array([b['volume'] for b in boxes])
        kinds={'c20':[],'local':['local'],'bp_residual':['bp','residual'],'bp_local_union':['bp','residual','local']}[method]
        for kind in kinds:
            rejected,meta=capture.screen(x,v,regs,volumes,roles[kind],32);mask|=rejected;proofs[kind]=meta
    screen_seconds=time.perf_counter()-start
    indices=np.flatnonzero(~mask)
    data=dict(regs=regs,boxes=[boxes[i] for i in indices] if boxes else [],matrices=[matrices[i] for i in indices],
              rhs=[rhs[i] for i in indices],indices=indices,polys=polys,poly_indices=np.array(poly_indices),method=method)
    meta=dict(capture_seconds=capture_seconds,contraction_seconds=contraction_seconds,construction_seconds=construction_seconds,
        screen_seconds=screen_seconds,rejected=int(mask.sum()),pool=len(regs),proofs=proofs,geometry_trace=trace,
        evaluated_pattern_keys=[r.tobytes().hex() for r in result[1]],surviving_pattern_keys=[regs[i].tobytes().hex() for i in indices],
        feedback_proofs=[c.feedback_bank.proofs for c in result[3]],
        proposal_arrays_subtotal=sum(b['transform'].nbytes+b['inverse'].nbytes+b['center'].nbytes+b['radius'].nbytes for b in boxes),
        inequality_arrays_subtotal=sum(a.nbytes+r.nbytes for a,r in zip(matrices,rhs)),
        geometry_arrays_subtotal=sum(a.nbytes for p in polys for a in p.values() if isinstance(a,np.ndarray)))
    return data,meta


def sample(data,count,rng,batch=32768,maximum_proposals=268435456):
    start=time.perf_counter()
    if data['method']=='geometry':
        if not data['polys']:raise RuntimeError('Geometry found no nonzero cell')
        weights=np.array([p['volume'] for p in data['polys']]);weights/=weights.sum()
        owners=rng.choice(len(weights),count,p=weights);points=np.empty((count,4))
        for i,poly in enumerate(data['polys']):
            indices=np.flatnonzero(owners==i)
            if len(indices):points[indices]=posterior.sample(poly,len(indices),rng)
        labels=data['poly_indices'][owners];attempts=count;batches=1;overflow=0
    else:
        points=[];labels=[];retained=0;attempts=0;batches=0;overflow=0
        while retained<count and attempts<maximum_proposals:
            n=min(batch,maximum_proposals-attempts)
            pp,ll,_=proposal.attempts(data['boxes'],data['matrices'],data['rhs'],n,rng)
            take=min(count-retained,len(pp));overflow+=len(pp)-take
            points.append(pp[:take]);labels.append(data['indices'][ll[:take]])
            retained+=take;attempts+=n;batches+=1
        points=np.concatenate(points) if points else np.empty((0,4));labels=np.concatenate(labels) if labels else np.empty(0,int)
    elapsed=time.perf_counter()-start
    if len(points)!=count:raise RuntimeError(f'Explicit proposal cap reached: {len(points)}/{count}, attempts={attempts}')
    return points,labels,dict(sampling_seconds=elapsed,physical_proposals=attempts,batches=batches,accepted_overflow_discarded=overflow)


def predict(points,q):return posterior.make_predict(points)(q)


def verify(x,v,cfg,expected_patterns,expected_masks):
    checks=0
    for method in METHODS:
        data,meta=prepare(x,v,cfg,method)
        assert [r.tobytes().hex() for r in data['regs']]==expected_patterns
        if method!='geometry':assert np.array_equal(data['indices'],np.flatnonzero(~expected_masks[method]))
        points,labels,note=sample(data,64,np.random.default_rng(481757))
        code,h=capture.model.light.forward_many(x,points)
        assert np.array_equal(code,data['regs'][labels]) and np.max(abs(h[:,-1]-v))<=.001+1e-8
        checks+=1
    jac=capture.model.base.forward_jacobian;bp_module=capture.model.old.old.old.old.old.core.batched;refine=bp_module.refine
    bp_capture=capture.capture
    def forbidden(*a,**k):raise AssertionError('Global BP or BP observer in local materialization')
    try:
        capture.model.base.forward_jacobian=forbidden;bp_module.refine=forbidden;capture.capture=forbidden
        data,_=prepare(x,v,cfg,'local');sample(data,64,np.random.default_rng(481757))
    finally:capture.model.base.forward_jacobian=jac;bp_module.refine=refine;capture.capture=bp_capture
    return dict(passed=True,common_pool_and_original_masks=checks,local_no_global_bp_and_no_bp_capture=True)

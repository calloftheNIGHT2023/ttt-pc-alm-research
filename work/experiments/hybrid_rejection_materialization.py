"""Fixed-attempt rejection with exact-distribution geometric completion.

The switch uses only success counts at a predetermined attempt boundary,
never query values, particle locations, or a wall-clock stopping condition.
"""
import itertools,time
from fractions import Fraction as F
import numpy as np
import rejection_memory_materialization as original

METHODS=original.METHODS
capture=original.capture
geometry=original.geometry
prepare=original.prepare
predict=original.predict
PROPOSALS_PER_PARTICLE=128


def sample(data,count,rng,batch=32768,maximum_proposals=268435456,proposal_budget=None):
    seeds=rng.integers(0,2**63,size=2,dtype=np.int64)
    rejection_rng=np.random.default_rng(int(seeds[0]));completion_rng=np.random.default_rng(int(seeds[1]))
    start=time.perf_counter()
    if data['method']=='geometry':
        points,labels,note=original.sample(data,count,completion_rng,batch,maximum_proposals)
        return points,labels,dict(note,fallback=False,geometry_fill_samples=count,rejection_sampling_seconds=0.,fallback_construction_seconds=0.,fallback_draw_seconds=note['sampling_seconds'])
    cap=min(maximum_proposals,PROPOSALS_PER_PARTICLE*count if proposal_budget is None else proposal_budget)
    if cap<0:raise ValueError('Negative proposal budget')
    points=[];labels=[];retained=0;attempts=0;batches=0;overflow=0
    while retained<count and attempts<cap:
        n=min(batch,cap-attempts)
        pp,ll,_=original.proposal.attempts(data['boxes'],data['matrices'],data['rhs'],n,rejection_rng)
        take=min(count-retained,len(pp));overflow+=len(pp)-take
        points.append(pp[:take]);labels.append(data['indices'][ll[:take]])
        retained+=take;attempts+=n;batches+=1
    reject_seconds=time.perf_counter()-start;fill=count-retained;construction=0.;draw=0.
    if fill:
        begin=time.perf_counter();polys=[];poly_indices=[]
        for i,a,r in zip(data['indices'],data['matrices'],data['rhs']):
            poly,note=original.posterior.polytope(a,r)
            if poly is not None:polys.append(poly);poly_indices.append(i)
        construction=time.perf_counter()-begin
        fallback=dict(data);fallback.update(method='geometry',polys=polys,poly_indices=np.array(poly_indices))
        pp,ll,note=original.sample(fallback,fill,completion_rng,batch,maximum_proposals)
        draw=note['sampling_seconds'];points.append(pp);labels.append(ll)
    points=np.concatenate(points);labels=np.concatenate(labels)
    assert len(points)==count and attempts<=cap
    return points,labels,dict(sampling_seconds=time.perf_counter()-start,physical_proposals=attempts,batches=batches,
        accepted_overflow_discarded=overflow,fallback=bool(fill),geometry_fill_samples=fill,
        rejection_sampling_seconds=reject_seconds,fallback_construction_seconds=construction,fallback_draw_seconds=draw,
        declared_proposal_cap=cap)


def verify_distribution():
    # Complete enumeration, not Monte Carlo: failure 1/2, accepted A 3/10,
    # accepted B 1/5. Conditional target is (3/5,2/5).
    prob={-1:F(1,2),0:F(3,10),1:F(1,5)};target={0:F(3,5),1:F(2,5)};checks=0
    for attempts in [0,1,3,5]:
        for count in [1,2,3]:
            joint={key:F(0) for key in itertools.product([0,1],repeat=count)}
            for raw in itertools.product([-1,0,1],repeat=attempts):
                weight=F(1)
                for value in raw:weight*=prob[value]
                retained=tuple(v for v in raw if v!=-1)[:count]
                for extra in itertools.product([0,1],repeat=count-len(retained)):
                    p=weight
                    for value in extra:p*=target[value]
                    joint[retained+extra]+=p
            for values,p in joint.items():
                expected=F(1)
                for value in values:expected*=target[value]
                assert p==expected;checks+=1
    return dict(passed=True,exact_joint_distribution_checks=checks,attempt_budgets=[0,1,3,5],particle_counts=[1,2,3],
                scope='exact finite enumeration of rejection-count-dependent completion; no point-dependent switch')


def verify(x,v,cfg,expected_patterns,expected_masks):
    result=original.verify(x,v,cfg,expected_patterns,expected_masks);result['hybrid_distribution']=verify_distribution()
    jac=capture.model.base.forward_jacobian;bp_module=capture.model.old.old.old.old.old.core.batched;refine=bp_module.refine;bp_capture=capture.capture
    def forbidden(*a,**k):raise AssertionError('Global BP in local hybrid including fallback')
    try:
        capture.model.base.forward_jacobian=forbidden;bp_module.refine=forbidden;capture.capture=forbidden
        data,_=prepare(x,v,cfg,'local')
        for cap in [0,65536]:
            points,labels,note=sample(data,64,np.random.default_rng(481781),proposal_budget=cap)
            codes,h=capture.model.light.forward_many(x,points)
            assert np.array_equal(codes,data['regs'][labels]) and np.max(abs(h[:,-1]-v))<=.001+1e-8
            if cap==0:assert note['fallback'] and note['geometry_fill_samples']==64 and note['physical_proposals']==0
    finally:capture.model.base.forward_jacobian=jac;bp_module.refine=refine;capture.capture=bp_capture
    return dict(result,forced_fallback_and_rejection_valid=True,local_fallback_no_bp=True)

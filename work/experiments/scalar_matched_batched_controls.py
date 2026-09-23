"""Batched strong controls plugged into the unchanged study-34 readout.

Sequential scoped adapter: restore the frozen pipeline after every fit. Random
starts match the old confirmation exactly; contextual starts are a separate,
explicit F256 control available equally to ALM and BP.
"""
import numpy as np
import posterior_confirmation_pipeline as pipeline
import batched_bp_discovery as batched
import contextual_candidate_bank as contextual
base=pipeline.base
frozen_discover=pipeline.discover


def discover(x,v,anchor,cfg):
    if cfg.get('initialization','random')=='random' and cfg['generator'] not in ['adam','gauss_newton']:
        return frozen_discover(x,v,anchor,cfg)
    with pipeline.discovery_box(cfg.get('discovery_bound',.15)):
        if cfg.get('initialization','random')=='contextual':
            starts,pm=contextual.proposals(x,v,anchor,cfg['features'],cfg['restarts'])
        else:
            starts=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(cfg['restarts']-1,len(anchor)))])
            pm=dict(initialization='same random generator 912 and anchor as study34',proposal_restarts=len(starts))
        if cfg['generator']=='alm':
            best,meta=contextual.refine_local(starts,x,v,anchor,sweeps=cfg.get('sweeps',120),dual_rate=cfg.get('dual_rate',.5))
        elif cfg['generator']=='direct':best=starts;meta=dict(no_iterative_refinement=True)
        else:best,meta=batched.refine(starts,x,v,anchor,solver=cfg['generator'],steps=cfg['steps'],lr=cfg.get('lr',.003))
        # The random original keeps each restart's support-selected best only.
        # Contextual controls all retain original proposal patterns as well.
        combined=np.vstack([starts,best]) if cfg.get('initialization')=='contextual' else best
        bank=contextual.deduplicate(combined,x,v,anchor)
    return bank,dict(**pm,**meta,unique_regions=len(bank),retained_bank_bytes=bank.nbytes)


def fit(x,v,anchor,cfg):
    previous=pipeline.discover
    try:
        pipeline.discover=discover
        return pipeline.fit(x,v,anchor,cfg)
    finally:pipeline.discover=previous


def verify():
    rng=np.random.default_rng(774619);x=rng.uniform(0,1,8);truth=rng.uniform(-.12,.12,4)
    v=base.forward(x,truth)+rng.uniform(-base.EPS,base.EPS,len(x));anchor=np.zeros(4);q=np.linspace(0,1,111)
    equal=[]
    for generator in ['alm','lbfgs']:
        cfg=dict(generator=generator,restarts=8,sweeps=12,discovery_bound=.15,posterior_samples=512)
        old,b,_=pipeline.fit(x,v,anchor,cfg);new,c,_=fit(x,v,anchor,cfg)
        assert np.array_equal(b,c) and np.array_equal(old(q),new(q))
        equal.append(generator)
    original_jacobian=base.forward_jacobian
    def forbidden(*args,**kwargs):raise AssertionError('global Jacobian entered local discovery')
    base.forward_jacobian=forbidden
    try:fit(x,v,anchor,dict(generator='alm',initialization='contextual',features=256,restarts=8,sweeps=12,discovery_bound=.12,posterior_samples=512))
    finally:base.forward_jacobian=original_jacobian
    assert pipeline.discover is frozen_discover
    return dict(passed=True,old_pipeline_exact_equivalence=equal,batched=batched.verify(),contextual_local_no_global_jacobian=True)

"""Expose the existing sampled memory without changing its frozen computation.

Sequential scoped factory adapter. State and predictor share the same immutable
array; a serialization copy is an artifact, not a required second online state.
"""
import numpy as np
import scalar_matched_batched_controls as core
import region_posterior_memory as posterior
import archived_region_memory as archived
base=core.base


def fit_captured(x,v,anchor,cfg,fit_function=core.fit):
    factory=posterior.make_predict;saved=[]
    def capture(bank):saved.append(bank);return factory(bank)
    try:
        posterior.make_predict=capture
        predict,point,meta=fit_function(x,v,anchor,cfg)
    finally:posterior.make_predict=factory
    assert len(saved)==1
    return predict,point,saved[0],meta


def select_pool(x,v,anchor,pool,restarts=64):
    regs=archived.signatures(x,pool)
    h=np.broadcast_to(x,(len(pool),len(x)))
    for j in range(pool.shape[1]):h=base.g(h+pool[:,j,None])
    mse=np.mean((h-v)**2,axis=1);order=np.argsort(mse,kind='stable')
    flat=regs.reshape(len(pool),-1);tokens=flat.view(np.dtype((np.void,flat.shape[1]))).reshape(-1)
    _,first=np.unique(tokens[order],return_index=True);ids=order[np.sort(first)][:restarts-1]
    starts=np.vstack([anchor,pool[ids]])
    return starts,dict(pool_candidates=len(pool),distinct_pool_patterns=len(first),selected_starts=len(starts),
                       best_pool_mse=float(mse.min()),best_pool_max_error=float(np.min(np.max(np.abs(h-v),axis=1))),
                       pool_parameter_bytes=pool.nbytes,pool_signature_bytes=regs.nbytes)


def make_pool(depth,kind,previous=None):
    if kind=='prior256':return np.random.default_rng(731).uniform(-.12,.12,(256,depth))
    if kind=='prior768':return np.random.default_rng(731).uniform(-.12,.12,(768,depth))
    if kind=='posterior_mix':
        assert previous is not None
        return np.vstack([make_pool(depth,'prior256'),previous])
    raise ValueError(kind)


def verify():
    rng=np.random.default_rng(184661);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    probe=np.linspace(0,1,117);cases=[]
    for gen in ['alm','adam']:
        left=np.zeros(4);right=np.zeros(4)
        for n in [4,8,16,24]:
            cfg=dict(generator=gen,initialization='contextual',features=256,restarts=8,sweeps=12,steps=12,discovery_bound=.12,posterior_samples=512)
            original,left,_=core.fit(x[:n],v[:n],left,cfg)
            captured,right,bank,_=fit_captured(x[:n],v[:n],right,cfg)
            assert np.array_equal(left,right) and np.array_equal(original(probe),captured(probe))
            assert np.array_equal(captured(probe),posterior.make_predict(bank)(probe));cases.append([gen,n])
    for n in [4,8,16,24]:
        old,_=core.contextual.proposals(x[:n],v[:n],np.zeros(4),256,64)
        selected,_=select_pool(x[:n],v[:n],np.zeros(4),make_pool(4,'prior256'))
        assert np.array_equal(old,selected)
    assert np.array_equal(make_pool(4,'prior256'),make_pool(4,'prior768')[:256])
    return dict(passed=True,full_original_output_and_anchor_equivalence=cases,
                predictor_from_exposed_state_exact=True,prior256_selection_matches_frozen_original=True,
                prior768_has_identical_first256=True,query_targets_absent_from_state_and_selection_interfaces=True)

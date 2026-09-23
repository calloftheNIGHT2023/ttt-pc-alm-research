"""Scoped first-block pool substitution; original solvers and recovery intact."""
import hashlib
import time
import numpy as np
import band_conditioned_initialization as initialization
import online_primal_upper_h2 as original
import light_h2_credit as light

neighbor=original.memory.neighbor
base=original.memory.base
interface=neighbor.previous.interface
POOLS=('prior256','prior1280','band_inverse')
LEARNERS=('alm','adam60','adam240','gn20','pc','nodual','direct64')
PRIMARY='band_inverse_alm_native_gate'

def configs():
    result=[];old={c['name']:c for c in original.configs()}
    for pool in POOLS:
        for learner in LEARNERS:
            parent={'adam240':'adam60','gn20':'adam60','direct64':'direct4096'}.get(learner,learner)
            c=dict(light.config(parent,'c5'),name=f'{pool}_{learner}_c5',learner=learner,band_pool=pool,restarts=64,initial_features=256)
            if learner=='adam240':c['steps']=240
            if learner=='gn20':c.update(generator='gauss_newton',steps=20)
            result.append(c)
        result.append(dict(old['alm_native_full_c5_interval_endpoints_primal_gate'],name=f'{pool}_alm_native_gate',band_pool=pool))
    for learner in ['adam60','pc','nodual']:
        result.append(dict(old[f'{learner}_native_full_c5_interval_endpoints_primal_gate'],name=f'band_inverse_{learner}_native_gate',band_pool='band_inverse'))
    result.extend(c for c in original.previous.configs() if c['family']=='regression' or c['name']=='direct4096_c5')
    result.extend(dict(name=n,family='meta') for n in ['prior256_fixed','meta_ridge64','meta_ridge128','meta_shallow64_5','meta_shallow64_20'])
    assert len(result)==len({c['name'] for c in result})==39
    return result

def conditioned_discover(saved,x,v,state,cfg):
    if state is not None or not cfg.get('band_pool'):return saved(x,v,state,cfg)
    kind=cfg['band_pool'];assert kind in POOLS and len(x)==4
    begin=time.perf_counter();records=None;generation=None;replacement=None
    if kind=='prior1280':replacement=np.random.default_rng(731).uniform(-.12,.12,(1280,4))
    elif kind=='band_inverse':
        points,records,generation=initialization.generate(x,v)
        assert len(points)<=1024
        replacement=np.vstack([np.random.default_rng(731).uniform(-.12,.12,(256,4)),points])
    pool_seconds=time.perf_counter()-begin;select=interface.select_pool;seen=[]
    def select_conditioned(xx,vv,anchor,pool,restarts=64):
        actual=pool if replacement is None else replacement
        starts,meta=select(xx,vv,anchor,actual,restarts)
        seen.append(dict(kind=kind,pool_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),starts_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),
            selected_starts=starts.tolist(),pool_candidates=len(actual),pool_generation_seconds=pool_seconds,generation=generation,inverse_records=records))
        return starts,meta
    try:
        interface.select_pool=select_conditioned
        bank,meta=saved(x,v,state,cfg)
    finally:interface.select_pool=select
    assert len(seen)==1
    return bank,dict(meta,band_initialization=seen[0],effective_pool=kind)

def fit(x,v,state,cfg,rng,count=2048):
    saved=neighbor.discover
    def discover(xx,vv,ss,cc):return conditioned_discover(saved,xx,vv,ss,cc)
    try:
        neighbor.discover=discover
        return original.fit(x,v,state,cfg,rng,count)
    finally:neighbor.discover=saved

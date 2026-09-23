"""Preserve visited activation cells, with identical screening and readout for ALM/BP.

This is a common memory-retention mechanism, not a claim of PC-specific novelty.
Only the first parameter representative per visited support pattern is stored.
"""
import time
import numpy as np
import scalar_matched_batched_controls as core
import local_region_screen as screen
base=core.base


def signatures(x,bank):
    h=np.broadcast_to(x,(len(bank),len(x)));regs=[]
    for j in range(bank.shape[1]):
        z=h+bank[:,j,None];regs.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));h=base.g(z)
    return np.ascontiguousarray(np.stack(regs,axis=1))


class Archive:
    def __init__(self,x):self.x=x;self.points={};self.visited_rows=0;self.seconds=0.
    def add(self,bank):
        start=time.perf_counter();self.visited_rows+=len(bank);regs=signatures(self.x,bank)
        flat=regs.reshape(len(bank),-1);tokens=flat.view(np.dtype((np.void,flat.shape[1]))).reshape(-1)
        _,first=np.unique(tokens,return_index=True)
        for i in np.sort(first):
            key=regs[i].tobytes()
            if key not in self.points:self.points[key]=bank[i].copy()
        self.seconds+=time.perf_counter()-start


def discover(x,v,anchor,cfg):
    archive=Archive(x);active=cfg.get('archive',False) and len(x)<=8
    old_score=base.score;old_evaluate=core.batched.evaluate
    def score(b,*args,**kwargs):archive.add(b);return old_score(b,*args,**kwargs)
    def evaluate(b,*args,**kwargs):archive.add(b);return old_evaluate(b,*args,**kwargs)
    try:
        if active:base.score=score;core.batched.evaluate=evaluate
        bank,meta=core.discover(x,v,anchor,cfg)
    finally:base.score=old_score;core.batched.evaluate=old_evaluate
    original_count=len(bank);original_keys={r.tobytes() for r in signatures(x,bank)}
    extras=[b for k,b in archive.points.items() if k not in original_keys]
    combined=np.vstack([bank,extras]) if extras else bank
    start=time.perf_counter();regs=signatures(x,combined);rejected=screen.contract(x,v,regs,rounds=5)
    # Preserve the old first-point fallback even when every cell is infeasible.
    # The original geometry handles it; all other rejected cells have no mass.
    retained=~rejected;retained[0]=True;filtered=combined[retained].copy();screen_seconds=time.perf_counter()-start
    return filtered,dict(**meta,archive_active=active,archive_visited_rows=archive.visited_rows,archive_unique_patterns=len(archive.points),
                         archive_extra_patterns=len(extras),original_bank_patterns=original_count,combined_patterns=len(combined),screen_rejected=int(rejected.sum()),
                         geometry_bank_patterns=len(filtered),archive_seconds=archive.seconds,screen_seconds=screen_seconds,
                         archive_numeric_key_bytes=sum(len(k)+b.nbytes for k,b in archive.points.items()),
                         archive_state_scope='keys plus numeric points only; dict/Python/native temporaries excluded; all archive state temporary within a write')


def fit(x,v,anchor,cfg):
    previous=core.pipeline.discover
    try:core.pipeline.discover=discover;return core.pipeline.fit(x,v,anchor,cfg)
    finally:core.pipeline.discover=previous


def verify():
    rng=np.random.default_rng(485776);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    q=np.linspace(0,1,111);cases=[]
    for generator in ['alm','adam']:
        oldanchor=np.zeros(4);anchor=np.zeros(4)
        for n in [4,8,16,24]:
            cfg=dict(generator=generator,restarts=8,sweeps=12,steps=12,discovery_bound=.12,archive=False,posterior_samples=512)
            old,oldanchor,_=core.fit(x[:n],v[:n],oldanchor,cfg);new,anchor,_=fit(x[:n],v[:n],anchor,cfg)
            assert np.array_equal(oldanchor,anchor) and np.array_equal(old(q),new(q));cases.append([generator,n])
    bank=rng.uniform(-.12,.12,(64,4));regs=signatures(x[:8],bank)
    assert all(np.array_equal(a,base.pattern(x[:8],b)) for a,b in zip(regs,bank))
    archive=Archive(x[:8]);archive.add(bank[:32]);archive.add(bank);assert len(archive.points)==len({r.tobytes() for r in regs})
    old=base.forward_jacobian;old_evaluate=core.batched.evaluate
    def forbidden(*args,**kwargs):raise AssertionError('global chain derivative entered local archive')
    try:
        base.forward_jacobian=forbidden;core.batched.evaluate=forbidden
        fit(x[:8],v[:8],np.zeros(4),dict(generator='alm',restarts=8,sweeps=12,discovery_bound=.12,archive=True,posterior_samples=512))
    finally:base.forward_jacobian=old;core.batched.evaluate=old_evaluate
    return dict(passed=True,screened_original_exact_equivalence=cases,signature_reference_cases=len(bank),local_no_global_gradient=True)

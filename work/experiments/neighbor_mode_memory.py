"""Finite local activation-neighborhood completion of discovered memory cells.

Shared by ALM, BP, PC and non-learning prior controls. Neighbor proposals use
only current observed constraints and discovered cells, never reference modes.
LP/QP/volume geometry is global and fully counted, not a purely local PC step.
"""
import hashlib,time
import numpy as np
from scipy.optimize import minimize
import stateful_posterior_memory as previous
base=previous.base;posterior=previous.posterior


def pattern_matrix(x,v,regs):
    d,n=regs.shape;p=np.zeros((n,d));c=x.copy();ar=[];rhs=[]
    low=np.array([-np.inf,0,.5,1.]);high=np.array([0,.5,1.,np.inf])
    for j in range(d):
        z=p.copy();z[:,j]+=1;r=regs[j]
        for i in range(n):
            if np.isfinite(high[r[i]]):ar.append(z[i]);rhs.append(high[r[i]]-c[i])
            if np.isfinite(low[r[i]]):ar.append(-z[i]);rhs.append(c[i]-low[r[i]])
        p=base.SLOPES[r,None]*z;c=base.SLOPES[r]*c+base.INTERCEPTS[r]
    for i in range(n):ar.extend([p[i],-p[i]]);rhs.extend([v[i]+base.EPS-c[i],-v[i]+base.EPS+c[i]])
    return p,c,np.array(ar),np.array(rhs)


def discover(x,v,state,cfg):
    later=state is not None
    effective=dict(cfg)
    if later:effective.update(generator='direct',restarts=128,pool='posterior_mix')
    active=cfg.get('archive',True) and not later
    archive=previous.interface.archived.Archive(x);old_score=base.score;old_evaluate=previous.core.batched.evaluate
    def score(b,*args,**kwargs):archive.add(b);return old_score(b,*args,**kwargs)
    def evaluate(b,*args,**kwargs):archive.add(b);return old_evaluate(b,*args,**kwargs)
    try:
        if active:base.score=score;previous.core.batched.evaluate=evaluate
        features=cfg.get('initial_features',256)
        if not later and features!=256:
            pool=np.random.default_rng(731).uniform(-.12,.12,(features,4))
            starts,pm=previous.interface.select_pool(x,v,np.zeros(4),pool,cfg['restarts'])
            with previous.core.pipeline.discovery_box(.12):
                if cfg['generator']=='alm':best,rm=previous.core.contextual.refine_local(starts,x,v,np.zeros(4),sweeps=cfg.get('sweeps',16),dual_rate=cfg.get('dual_rate',.5))
                elif cfg['generator']=='pc':best,rm=previous.refine_pc(starts,x,v,np.zeros(4),sweeps=cfg.get('sweeps',80))
                elif cfg['generator']=='direct':best=starts;rm=dict(no_iterative_refinement=True)
                else:best,rm=previous.core.batched.refine(starts,x,v,np.zeros(4),solver=cfg['generator'],steps=cfg['steps'],lr=.003)
                bank=previous.core.contextual.deduplicate(np.vstack([starts,best]),x,v,np.zeros(4))
            meta=dict(**pm,**rm,effective_pool=f'prior{features}',effective_generator=cfg['generator'],previous_posterior_samples=0,unique_regions=len(bank),retained_bank_bytes=bank.nbytes)
        else:bank,meta=previous.discover(x,v,state,effective)
    finally:base.score=old_score;previous.core.batched.evaluate=old_evaluate
    original={r.tobytes() for r in previous.interface.archived.signatures(x,bank)}
    extra=[b for k,b in archive.points.items() if k not in original]
    if extra:bank=np.vstack([bank,extra])
    return bank,dict(**meta,archive_active=active,archive_unique_patterns=len(archive.points),
        archive_numeric_key_bytes=sum(len(k)+b.nbytes for k,b in archive.points.items()),archive_bookkeeping_seconds=archive.seconds)


def fit(x,v,state,cfg):
    begin=time.perf_counter();bank,meta=discover(x,v,state,cfg);discovery=time.perf_counter()-begin
    anchor=np.zeros(4) if state is None else state.anchor
    begin=time.perf_counter();regs=previous.interface.archived.signatures(x,bank);skip=previous.screen.contract(x,v,regs,5)
    tested=set();polys={};trace=[];point=None;initial_calls=0
    for b,reg,reject in zip(bank,regs,skip):
        key=reg.tobytes()
        if key in tested:continue
        tested.add(key)
        if reject:continue
        _,_,g,rhs=pattern_matrix(x,v,reg);poly,note=posterior.polytope(g,rhs);initial_calls+=1;trace.append(note)
        if poly is None:continue
        polys[key]=poly
        if point is None:
            initial=poly['center']+poly['scale']*poly['interior']
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method='SLSQP',
                constraints={'type':'ineq','fun':lambda b:poly['rhs']-poly['a']@b,'jac':lambda b:-poly['a']},
                bounds=[(-.12,.12)]*len(anchor),options={'maxiter':300,'ftol':1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    if point is None:point=np.clip(bank[0],-.12,.12)
    initial_geometry=time.perf_counter()-begin;initial_keys=set(polys);frontier=list(polys)
    rounds=cfg.get('completion_rounds',0) if state is None else 0;budget=cfg.get('proposal_budget',1024)
    begin=time.perf_counter();proposed=0;screened=0;new_calls=0;round_stats=[];truncated=False;max_batch_bytes=0
    for iteration in range(rounds):
        proposals=[];parents=sorted(frontier,key=lambda k:(-polys[k]['volume'],k))
        for parent in parents:
            flat=np.frombuffer(parent,dtype=np.uint8)
            for position in range(len(flat)):
                for delta in [-1,1]:
                    code=int(flat[position])+delta
                    if not 0<=code<=3:continue
                    new=flat.copy();new[position]=code;key=new.tobytes()
                    if key in tested:continue
                    if proposed>=budget:truncated=True;break
                    tested.add(key);proposed+=1;proposals.append(new.reshape(regs.shape[1:]))
                if truncated:break
            if truncated:break
        if not proposals:break
        proposal_array=np.array(proposals);max_batch_bytes=max(max_batch_bytes,proposal_array.nbytes)
        rejected=previous.screen.contract(x,v,proposal_array,5);screened+=int(rejected.sum());frontier=[];calls=0
        for reg,reject in zip(proposal_array,rejected):
            if reject:continue
            _,_,g,rhs=pattern_matrix(x,v,reg);poly,note=posterior.polytope(g,rhs);new_calls+=1;calls+=1;trace.append(note)
            if poly is None:continue
            key=reg.tobytes();polys[key]=poly;frontier.append(key)
        round_stats.append(dict(round=iteration+1,proposals=len(proposals),screened=int(rejected.sum()),geometry_calls=calls,new_positive_regions=len(frontier)))
        if truncated or not frontier:break
    completion=time.perf_counter()-begin;begin=time.perf_counter();keys=sorted(polys)
    volume=np.array([polys[k]['volume'] for k in keys])
    if keys:
        rng=np.random.default_rng(6173);counts=rng.multinomial(cfg.get('posterior_samples',512),volume/volume.sum())
        samples=np.concatenate([posterior.sample(polys[k],int(n),rng) for k,n in zip(keys,counts) if n])
        assert max(np.max(np.abs(base.forward(x,b)-v)) for b in samples)<=base.EPS+1e-7
    else:samples=point[None].copy()
    sampling=time.perf_counter()-begin;point.setflags(write=False);samples.setflags(write=False)
    return posterior.make_predict(samples),previous.State(point,samples),dict(**meta,
        discovery_seconds=discovery,initial_geometry_seconds=initial_geometry,completion_seconds=completion,sampling_seconds=sampling,
        initial_screened_patterns=int(skip.sum()),initial_geometry_calls=initial_calls,completion_geometry_calls=new_calls,
        initial_positive_regions=len(initial_keys),positive_volume_regions=len(keys),completion_new_positive_regions=len(set(keys)-initial_keys),
        completion_round_stats=round_stats,completion_proposals=proposed,completion_screened=screened,proposal_budget_truncated=truncated,
        maximum_neighbor_pattern_array_bytes=max_batch_bytes,tested_pattern_key_numeric_bytes=sum(map(len,tested)),
        discovered_prior_mass=float(volume.sum()/.24**len(anchor)),positive_mode_keys=[k.hex() for k in keys],
        positive_cell_set_sha256=hashlib.sha256(b''.join(keys)).hexdigest(),anchor_output=point.tolist(),
        persistent_state_bytes=point.nbytes+samples.nbytes,common_observed_context_bytes=x.nbytes+v.nbytes,query_read_samples=len(samples),
        geometry_numeric_arrays_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a,np.ndarray)),
        geometry_trace=trace,posterior_claim='conditional on discovered region union; finite adjacent-branch completion, no global completeness claim')


def verify():
    rng=np.random.default_rng(184913);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    checks=0
    for n in [4,8,24]:
        for b in rng.uniform(-.12,.12,(12,4)):
            expected=base.branch_polytope(x[:n],v[:n],b);actual=pattern_matrix(x[:n],v[:n],base.pattern(x[:n],b))
            assert all(np.array_equal(a,b) for a,b in zip(expected,actual));checks+=1
    equal=[];q=np.linspace(0,1,117)
    for gen in ['alm','adam']:
        for enabled in [False,True]:
            cfg=dict(generator=gen,archive=enabled,pool='prior256',restarts=8,sweeps=12,steps=12,completion_rounds=0)
            bank,_=previous.interface.archived.discover(x[:4],v[:4],np.zeros(4),dict(generator=gen,archive=enabled,initialization='contextual',features=256,restarts=8,sweeps=12,steps=12,discovery_bound=.12))
            expected,state,_=previous.materialize(x[:4],v[:4],np.zeros(4),bank)
            actual,newstate,_=fit(x[:4],v[:4],None,cfg)
            assert np.array_equal(state.samples,newstate.samples) and np.array_equal(state.anchor,newstate.anchor) and np.array_equal(expected(q),actual(q));equal.append([gen,enabled])
    old_jac=base.forward_jacobian;old_refine=previous.core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('global BP entered local candidate')
    try:
        base.forward_jacobian=forbidden;previous.core.batched.refine=forbidden
        _,s,meta=fit(x[:4],v[:4],None,dict(generator='alm',archive=True,pool='prior256',restarts=8,sweeps=12,completion_rounds=2,proposal_budget=1024))
    finally:base.forward_jacobian=old_jac;previous.core.batched.refine=old_refine
    assert not meta['proposal_budget_truncated']
    large=[]
    for gen in ['alm','adam','direct']:
        cfg=dict(generator=gen,archive=False,pool='prior256',initial_features=512,restarts=8,sweeps=12,steps=12,completion_rounds=0)
        bank,_=previous.core.discover(x[:4],v[:4],np.zeros(4),dict(generator=gen,initialization='contextual',features=512,restarts=8,sweeps=12,steps=12,discovery_bound=.12))
        _,expected,_=previous.materialize(x[:4],v[:4],np.zeros(4),bank)
        _,actual,_=fit(x[:4],v[:4],None,cfg)
        assert np.array_equal(expected.samples,actual.samples) and np.array_equal(expected.anchor,actual.anchor);large.append(gen)
    return dict(passed=True,original_branch_matrix_bitwise_cases=checks,zero_completion_old_archive_exact=equal,
        local_candidate_no_global_bp=True,larger_prior_interface_exact=large,synthetic_neighbor_proposals=meta['completion_proposals'],synthetic_positive_regions=meta['positive_volume_regions'])


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

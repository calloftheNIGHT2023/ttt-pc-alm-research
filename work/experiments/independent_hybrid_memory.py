"""Own-trajectory, own-state online controls for bounded materialization.

No target generator, query answer, old state artifact, or evaluator enters fit.
All posterior arms use the first emitted particle as the next anchor. H2
geometry is retained, not recomputed. This is a new state convention, not a
claim of bitwise equivalence to the historical closest-point QP convention.
"""
import hashlib
import time
from types import SimpleNamespace
import numpy as np
import hybrid_rejection_materialization as hybrid
import rejection_credit_capture as capture
import neighbor_mode_memory as neighbor
import parallelotope_sampler as proposal
import region_posterior_memory as posterior

base=neighbor.base
State=neighbor.previous.State


def configs():
    answer=[]
    families=[('alm','local',16),('adam60','bp',60),('pc','residual',80),('nodual','residual',16)]
    for name,credit,iterations in families:
        gen='adam' if name=='adam60' else name
        samplers=['hybrid','c20','geometry'] if name in ['alm','adam60'] else ['hybrid','geometry']
        for sampler in samplers:
            answer.append(dict(name=f'{name}_{sampler}',family='typed',generator=gen,credits=[credit],
                sweeps=iterations,steps=iterations,restarts=64,retention_mode='bank',feedback_mode='repair',
                route_mode='tied_forward',sampler=sampler,archive=True,pool='posterior_mix'))
    answer.append(dict(answer[0],name='alm_residual_hybrid',credits=['residual']))
    for name,gen,features in [('alm_h2','alm',256),('adam60_h2','adam',256),('direct4096_h2','direct',4096)]:
        answer.append(dict(name=name,family='h2',generator=gen,sweeps=16,steps=60,
            initial_features=features,restarts=128 if gen=='direct' else 64,archive=True,
            pool='posterior_mix',completion_rounds=2,proposal_budget=1024,sampler='geometry'))
    for name in ['linear_ls','residual_linear_ls','residual_linear_rls','prior4096_ridge','rbf_loocv']:
        answer.append(dict(name=name,family='regression'))
    return answer


def state_digest(state):
    if state is None:return None
    return hashlib.sha256(state.anchor.tobytes()+state.samples.tobytes()).hexdigest()


def h2_prepare(x,v,cfg):
    """Same H2 discovery/parent ordering as the frozen implementation, no QP.

    The historical QP anchor did not feed first-write H2 proposals. Only its
    unused computation and the final historical sample step are omitted.
    """
    begin=time.perf_counter();bank,discovery=neighbor.discover(x,v,None,cfg)
    regs=neighbor.previous.interface.archived.signatures(x,bank)
    skip=neighbor.previous.screen.contract(x,v,regs,5)
    tested=set();polys={};calls=0;trace=[]
    for reg,reject in zip(regs,skip):
        key=reg.tobytes()
        if key in tested:continue
        tested.add(key)
        if reject:continue
        _,_,a,r=neighbor.pattern_matrix(x,v,reg);poly,note=posterior.polytope(a,r)
        calls+=1;trace.append(note)
        if poly is not None:polys[key]=poly
    initial=set(polys);frontier=list(polys);proposed=0;truncated=False;rounds=[]
    for iteration in range(cfg.get('completion_rounds',2)):
        candidates=[]
        for parent in sorted(frontier,key=lambda k:(-polys[k]['volume'],k)):
            flat=np.frombuffer(parent,np.uint8)
            for pos in range(len(flat)):
                for delta in [-1,1]:
                    value=int(flat[pos])+delta
                    if not 0<=value<=3:continue
                    rr=flat.copy();rr[pos]=value;key=rr.tobytes()
                    if key in tested:continue
                    if proposed>=cfg.get('proposal_budget',1024):truncated=True;break
                    tested.add(key);proposed+=1;candidates.append(rr.reshape(4,len(x)))
                if truncated:break
            if truncated:break
        if not candidates:break
        rr=np.array(candidates);rejected=neighbor.previous.screen.contract(x,v,rr,5);frontier=[]
        for reg,reject in zip(rr,rejected):
            if reject:continue
            _,_,a,r=neighbor.pattern_matrix(x,v,reg);poly,note=posterior.polytope(a,r)
            calls+=1;trace.append(note)
            if poly is not None:polys[reg.tobytes()]=poly;frontier.append(reg.tobytes())
        rounds.append(dict(round=iteration+1,proposals=len(rr),positive=len(frontier)))
        if truncated or not frontier:break
    keys=sorted(polys);regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in keys],np.uint8).reshape(-1,4,len(x))
    indices=np.arange(len(regs))
    data=dict(regs=regs,indices=indices,boxes=[],matrices=[],rhs=[],method='geometry',
        polys=[polys[k] for k in keys],poly_indices=indices)
    return data,dict(discovery=discovery,discovery_seconds=time.perf_counter()-begin,
        pool=len(tested),surviving=len(regs),post_credit_rejected=0,post_credit_seconds=0.,
        geometry_calls=calls,positive_mode_keys=[k.hex() for k in keys],
        initial_positive=len(initial),completion_new_positive=len(set(keys)-initial),
        completion_proposals=proposed,completion_rounds=rounds,proposal_budget_truncated=truncated,
        geometry_trace=trace,geometry_reused=True,collector_kinds=[])


def prepare(x,v,state,cfg):
    if state is None and cfg['family']=='h2':return h2_prepare(x,v,cfg)
    begin=time.perf_counter();collectors=[]
    if state is None:
        result=capture.model.prepare(x,v,cfg)
        regs=capture.pool(result);collectors=result[3];discovery=result[2]
        assert [c.kind for c in collectors]==cfg['credits']
        assert discovery['effective_generator']==cfg['generator']
    else:
        bank,discovery=neighbor.discover(x,v,state,cfg)
        regs=neighbor.previous.interface.archived.signatures(x,bank)
        # Sampling is over disjoint cells, never over duplicate visits.
        keys=sorted({r.tobytes() for r in regs})
        regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in keys],np.uint8)
        assert discovery['effective_generator']=='direct'
        assert discovery['previous_posterior_samples']==len(state.samples)
    discovery_seconds=time.perf_counter()-begin
    begin=time.perf_counter();regs=regs[~capture.model.conflict.screen.contract(x,v,regs,20)]
    contraction_seconds=time.perf_counter()-begin
    begin=time.perf_counter();boxes=[];matrices=[];rhs=[];polys=[];poly_indices=[];trace=[]
    for i,reg in enumerate(regs):
        p,c,a,r=neighbor.pattern_matrix(x,v,reg);matrices.append(a);rhs.append(r)
        if cfg['sampler']=='geometry':
            poly,note=posterior.polytope(a,r);trace.append(note)
            if poly is not None:polys.append(poly);poly_indices.append(i)
        else:boxes.append(proposal.make_box(p,c,v))
    construction_seconds=time.perf_counter()-begin
    begin=time.perf_counter();reject=np.zeros(len(regs),bool);proofs=[]
    if cfg['sampler']=='hybrid' and collectors:
        groups=[collectors]
        # BP retained entries already contain residuals. The residual-only
        # bank is a view, preserving the stronger two-bank comparator without
        # another observer or another BP pass over the trajectory.
        if cfg['credits']==['bp']:
            groups.append([SimpleNamespace(kind='residual',retained={k:v for k,v in c.retained.items() if k[1]=='residual'}) for c in collectors])
        volumes=np.array([b['volume'] for b in boxes])
        for group in groups:
            mask,note=capture.screen(x,v,regs,volumes,group,32);reject|=mask;proofs.append(note)
    credit_seconds=time.perf_counter()-begin;indices=np.flatnonzero(~reject)
    data=dict(regs=regs,indices=indices,boxes=[boxes[i] for i in indices] if boxes else [],
        matrices=[matrices[i] for i in indices],rhs=[rhs[i] for i in indices],
        polys=polys,poly_indices=np.array(poly_indices),method='geometry' if cfg['sampler']=='geometry' else 'c20')
    return data,dict(discovery=discovery,discovery_seconds=discovery_seconds,
        contraction_seconds=contraction_seconds,construction_seconds=construction_seconds,
        post_credit_seconds=credit_seconds,post_credit_rejected=int(reject.sum()),proofs=proofs,
        pool=len(regs),surviving=len(indices),collector_kinds=[c.kind for c in collectors],
        geometry_calls=len(trace),geometry_trace=trace,geometry_reused=False,
        positive_mode_keys=[regs[i].tobytes().hex() for i in poly_indices],
        surviving_pattern_keys=[regs[i].tobytes().hex() for i in indices])


def fit(x,v,state,cfg,rng,count=2048,proposal_budget=None):
    previous=state_digest(state)
    data,meta=prepare(x,v,state,cfg)
    points,labels,sampling=hybrid.sample(data,count,rng,proposal_budget=proposal_budget)
    anchor=points[0].copy();anchor.setflags(write=False);points.setflags(write=False)
    new=State(anchor,points)
    return posterior.make_predict(points),new,dict(meta,sampling=sampling,
        previous_state_digest=previous,new_state_digest=state_digest(new),
        persistent_state_bytes=anchor.nbytes+points.nbytes,
        old_and_new_particle_bytes=(0 if state is None else state.anchor.nbytes+state.samples.nbytes)+anchor.nbytes+points.nbytes,
        own_state_only=True,anchor_rule='first_emitted_particle',
        actual_mode_keys=sorted({data['regs'][int(i)].tobytes().hex() for i in labels}))


def regression(x,v,name,state=None):
    """Same observations; task-shaped fixed nonlinear features from the prior.

    The affine LS heads include intercepts. RLS is exactly the streaming form
    of a fixed ridge objective, checked against a batch solve in preflight.
    """
    if name=='prior4096_ridge':
        pred,meta=posterior.prior_moments(x,v,4,4096)
        return pred,None,meta
    if name=='rbf_loocv':
        # Tune only on observed support values; no query feature or answer.
        best=None;mean=float(v.mean());target=v-mean
        for length in [.01,.02,.04,.08,.16,.32]:
            gram=np.exp(-.5*((x[:,None]-x[None,:])/length)**2)
            for ridge in [1e-6,1e-4,.01,.1,1.]:
                inverse=np.linalg.solve(gram+ridge*np.eye(len(x)),np.eye(len(x)))
                alpha=inverse@target;loo=float(np.mean((alpha/np.diag(inverse))**2))
                # The fixed mean is fitted to all support values. The
                # criterion is a support-only PRESS heuristic, not an
                # unbiased leave-one-out score for a refitted intercept.
                if best is None or loo<best[0]:best=(loo,length,ridge,alpha.copy())
        _,length,ridge,alpha=best;saved_x=x.copy()
        def predict(q):return mean+np.exp(-.5*((q[:,None]-saved_x[None,:])/length)**2)@alpha
        return predict,None,dict(persistent_state_bytes=saved_x.nbytes+alpha.nbytes+24,
            selected_length=length,selected_ridge=ridge,support_press=best[0],hyperparameter_solves=30)
    phi=np.column_stack([np.ones(len(x)),x])
    residual=name!='linear_ls';target=v-base.forward(x,np.zeros(4)) if residual else v
    if name=='residual_linear_rls':
        if state is None:p=np.eye(2)/1e-4;w=np.zeros(2);seen=0
        else:p=state['p'].copy();w=state['w'].copy();seen=state['seen']
        for feature,value in zip(phi[seen:],target[seen:]):
            pf=p@feature;gain=pf/(1+feature@pf)
            w+=gain*(value-feature@w);p-=np.outer(pf,pf)/(1+feature@pf)
        new=dict(p=p,w=w,seen=len(x));size=p.nbytes+w.nbytes+8
    else:w=np.linalg.lstsq(phi,target,rcond=None)[0];new=None;size=w.nbytes
    def predict(q):
        answer=w[0]+q*w[1]
        return answer+base.forward(q,np.zeros(4)) if residual else answer
    return predict,new,dict(persistent_state_bytes=size,regression_name=name,
        ridge_lambda=1e-4 if name=='residual_linear_rls' else 0.)


def verify():
    rng=np.random.default_rng(5900001);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,truth)+np.random.default_rng(24900001).uniform(-.001,.001,24)
    q=np.linspace(0,1,61);checks=[];h2_checks=0;rls_checks=0
    for cfg in configs():
        if cfg['family']=='regression':
            state=None
            for n in [4,8,16,24]:
                pred,state,meta=regression(xx[:n],vv[:n],cfg['name'],state)
                assert np.all(np.isfinite(pred(q)))
                if cfg['name']=='residual_linear_rls':
                    phi=np.column_stack([np.ones(n),xx[:n]]);y=vv[:n]-base.forward(xx[:n],np.zeros(4))
                    expected=np.linalg.solve(phi.T@phi+1e-4*np.eye(2),phi.T@y)
                    assert np.max(abs(expected-state['w']))<1e-9;rls_checks+=1
            continue
        state=None
        for n in [4,8,16,24]:
            digest=state_digest(state)
            pred,state,meta=fit(xx[:n],vv[:n],state,cfg,np.random.default_rng(481829+n),count=64)
            assert meta['previous_state_digest']==digest
            assert np.array_equal(state.anchor,state.samples[0])
            assert np.all(np.isfinite(pred(q)))
            codes,h=capture.model.light.forward_many(xx[:n],state.samples)
            assert np.max(abs(h[:,-1]-vv[:n]))<=.001+1e-8
            assert sorted({r.tobytes().hex() for r in codes})==meta['actual_mode_keys']
            checks.append([cfg['name'],n])
        if cfg['family']=='h2':
            _,_,old=neighbor.fit(xx[:4],vv[:4],None,dict(cfg,posterior_samples=64))
            _,new=h2_prepare(xx[:4],vv[:4],cfg)
            assert old['positive_mode_keys']==new['positive_mode_keys'];h2_checks+=1
    jac=base.forward_jacobian;module=capture.model.old.old.old.old.old.core.batched;refine=module.refine;observer=capture.capture
    def forbidden(*a,**k):raise AssertionError('Global BP or extra observer in local candidate')
    try:
        base.forward_jacobian=forbidden;module.refine=forbidden;capture.capture=forbidden
        state=None
        for n in [4,8,16,24]:
            _,state,meta=fit(xx[:n],vv[:n],state,configs()[0],np.random.default_rng(481829+n),64,proposal_budget=0)
            assert meta['sampling']['fallback']
    finally:base.forward_jacobian=jac;module.refine=refine;capture.capture=observer
    return dict(passed=True,posterior_four_stage_checks=checks,h2_positive_set_equivalence=h2_checks,
        rls_batch_equivalence=rls_checks,local_all_stages_forced_fallback_no_global_bp=True,
        hybrid_distribution=hybrid.verify_distribution())


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)

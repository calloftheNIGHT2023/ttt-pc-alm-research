"""409 component checks; gated by both 405 completion and 407 exact math."""
from contextlib import contextmanager
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import common_band_optimizers_v1 as opt
import common_band_objective_math_v1 as exact
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/common_band_optimizers/preflight_v1'
DESIGN='outputs/ttt-pc-alm-research/409_common_band_optimizer_protocol_v1.md'


def scalar_activity(a,target,bias,old,trust):
    a,target,bias,old,trust=map(F,(a,target,bias,old,trust))
    knots=sorted({F(0),F(1)}|{k-bias for k in [F(0),F(1,2),F(1)] if 0<k-bias<1})
    candidates=set(knots)
    for left,right in zip(knots,knots[1:]):
        midpoint=(left+right)/2;slope=exact.slope(midpoint+bias)
        offset=exact.tent(midpoint+bias)-slope*midpoint
        root=(a+slope*(target-offset)+trust*old)/(1+slope**2+trust)
        candidates.add(min(right,max(left,root)))
    def energy(h):return (h-a)**2+(exact.tent(h+bias)-target)**2+trust*(h-old)**2
    return min(candidates,key=energy),min(map(energy,candidates)),energy


def scalar_bias(previous,target,old,trust,bound):
    previous,target=list(map(F,previous)),list(map(F,target));old,trust,bound=map(F,(old,trust,bound))
    n=len(previous);knots=sorted({-bound,bound}|{k-p for p in previous for k in [F(0),F(1,2),F(1)] if -bound<k-p<bound})
    candidates=set(knots)
    for left,right in zip(knots,knots[1:]):
        middle=(left+right)/2;slopes=[exact.slope(p+middle) for p in previous]
        offsets=[exact.tent(p+middle)-s*middle for p,s in zip(previous,slopes)]
        aa=sum((s*s for s in slopes),F(0))/n+trust
        bb=sum((s*(t-c) for s,t,c in zip(slopes,target,offsets)),F(0))/n+trust*old
        candidates.add(min(right,max(left,bb/aa)))
    def energy(b):return sum(((exact.tent(p+b)-t)**2 for p,t in zip(previous,target)),F(0))/n+trust*(b-old)**2
    return min(candidates,key=energy),min(map(energy,candidates)),energy


@contextmanager
def forbid_global_credit():
    original=opt.evaluate;old_chain=opt.base.forward_jacobian
    def guarded(*args,**kwargs):
        assert not kwargs.get('jacobian',False), 'Local method requested entire-network Jacobian'
        return original(*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Local method called old entire-network Jacobian')
    opt.evaluate=guarded;opt.base.forward_jacobian=forbidden
    try:yield
    finally:opt.evaluate=original;opt.base.forward_jacobian=old_chain


def main():
    begin=time.perf_counter();rng=np.random.default_rng(409731)
    files=[Path(__file__),Path(opt.__file__),Path(exact.__file__),ROOT/DESIGN,
        ROOT/'work/experiments/streaming_branch_projection.py',ROOT/'work/experiments/local_branch_memory.py']
    hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in files}
    io.save(OUT/'protocol.json',dict(source_sha256=hashes,random_seed=409731,
        families=list(opt.FAMILIES),query_targets_accessed=False,
        exact_math_summary_sha256=io.sha(ROOT/'results/common_band_objective/math_preflight_v1/summary.json'),
        matched_budget_or_task_risk_experiment=False))
    counts=dict(output_proximals=0,activity_blocks=0,bias_blocks=0,local_derivative_values=0,
        bp_derivative_values=0,block_sweep_descent=0,first_primal_sweep_pairs=0,
        zero_dual_rate_pairs=0,family_runs=0,individual_restart_replays=0,
        query_permutations=0,local_no_global_credit_runs=0)
    maxima=dict(output_prox=0.,activity_energy_gap=0.,bias_energy_gap=0.,local_derivative=0.,bp_derivative=0.)
    for _ in range(64):
        lo,hi=np.sort(rng.random(2));a=float(rng.uniform(-1,2));previous=float(rng.random())
        rho=float(rng.uniform(.1,2));trust=.01;tau=float(rng.uniform(.2,2))
        expected=exact.output_prox(a,previous,lo,hi,rho=rho,trust=trust,tau=tau)
        actual=opt.output_prox(a,previous,lo,hi,rho,trust,tau)
        error=abs(float(expected)-actual);assert error<2e-14
        maxima['output_prox']=max(maxima['output_prox'],error);counts['output_proximals']+=1
    a=rng.uniform(-.5,1.5,(6,5));target=rng.uniform(-.5,1.5,(6,5));old=rng.random((6,5));bias=rng.uniform(-.12,.12,6)
    actual=opt.activity_block(a,target,bias,old,.01)
    for r in range(6):
        for i in range(5):
            point,value,energy=scalar_activity(a[r,i],target[r,i],bias[r],old[r,i],.01)
            gap=float(energy(F(float(actual[r,i])))-value);assert -1e-13<=gap<1e-12
            maxima['activity_energy_gap']=max(maxima['activity_energy_gap'],abs(gap));counts['activity_blocks']+=1
    previous=rng.random((6,5));target=rng.uniform(-.5,1.5,(6,5));old=rng.uniform(-.12,.12,6)
    previous_bound=opt.base.BOUND;actual=opt.bias_block(previous,target,old,.01,.12)
    assert opt.base.BOUND==previous_bound
    for r in range(6):
        point,value,energy=scalar_bias(previous[r],target[r],old[r],.01,.12)
        gap=float(energy(F(float(actual[r])))-value);assert -1e-13<=gap<1e-11
        maxima['bias_energy_gap']=max(maxima['bias_energy_gap'],abs(gap));counts['bias_blocks']+=1
    x=rng.uniform(.01,.99,5);v=rng.uniform(.01,.99,5);start=rng.uniform(-.11,.11,(4,4));q=np.linspace(0,1,17)
    h=rng.uniform(.02,.98,(4,4,5));u=rng.uniform(-.3,.3,h.shape)
    gb,gh=opt.local_partials(start,h,u,x,v,rho=.7,tau=1.3)
    for r in range(len(start)):
        eb,eh=exact.local_partials(list(map(F,x)),list(map(F,v)),list(map(F,start[r])),
            [list(map(F,row)) for row in h[:,r]],[list(map(F,row)) for row in u[:,r]],F(.001),rho=F(.7),tau=F(1.3))
        error=max(np.max(abs(np.array(eb,float)-gb[r])),np.max(abs(np.array(eh,float)-gh[:,r])))
        assert error<1e-12;maxima['local_derivative']=max(maxima['local_derivative'],float(error))
        counts['local_derivative_values']+=len(eb)+np.size(eh)
    _,residual,jac,_=opt.evaluate(start,x,v,tau=1.3,jacobian=True)
    gradient=np.einsum('rni,rn->ri',jac,residual)/(len(x)*1.3)
    delta=F(1,2**48)
    for r in range(len(start)):
        for j in range(4):
            bp=list(map(F,start[r]));bm=bp.copy();bp[j]+=delta;bm[j]-=delta
            expected=(exact.objective(x,v,bp,F(.001),tau=F(1.3))-exact.objective(x,v,bm,F(.001),tau=F(1.3)))/(2*delta)
            error=abs(float(expected)-gradient[r,j]);assert error<1e-10
            maxima['bp_derivative']=max(maxima['bp_derivative'],error);counts['bp_derivative_values']+=1
    for kind in ['grad','block']:
        pc=opt.Local(start,x,v,'pc_'+kind);alm=opt.Local(start,x,v,'alm_'+kind)
        assert np.count_nonzero(pc.u)==np.count_nonzero(alm.u)==0
        pc.step();alm.step()
        np.testing.assert_array_equal(pc.b,alm.b);np.testing.assert_array_equal(pc.h,alm.h)
        assert not np.any(pc.u) and np.any(alm.u);counts['first_primal_sweep_pairs']+=1
        pc=opt.Local(start,x,v,'pc_'+kind);zero=opt.Local(start,x,v,'alm_'+kind,dual_rate=0.)
        for _ in range(5):pc.step();zero.step()
        for key in ['b','h','u']:np.testing.assert_array_equal(getattr(pc,key),getattr(zero,key))
        counts['zero_dual_rate_pairs']+=1
    for family in ['pc_block','alm_block']:
        state=opt.Local(start,x,v,family,rho=.7,tau=1.3)
        for _ in range(5):
            before=opt.augmented(state.b,state.h,state.u,x,v,rho=.7,tau=1.3);old_dual=state.u.copy()
            state.step();after=opt.augmented(state.b,state.h,old_dual,x,v,rho=.7,tau=1.3)
            assert np.max(after-before)<=1e-11;counts['block_sweep_descent']+=len(start)
    signatures=(start.tobytes(),x.tobytes(),v.tobytes(),q.tobytes())
    outputs={}
    for family in opt.FAMILIES:
        if family.startswith('bp_'):arrays,metadata=opt.fit(start,x,v,q,family=family,steps=5,trace=True)
        else:
            with forbid_global_credit():arrays,metadata=opt.fit(start,x,v,q,family=family,steps=5,trace=True)
            assert metadata['whole_chain_jacobian_rows']==0 and not metadata['dual_initialized_from_bp']
            counts['local_no_global_credit_runs']+=1
        assert signatures==(start.tobytes(),x.tobytes(),v.tobytes(),q.tobytes())
        assert np.all(np.diff(arrays['trace_objective'],axis=0)<=0)
        assert np.all(abs(arrays['best_bank'])<=.12) and np.isfinite(arrays['prediction']).all()
        independent=[]
        for row in start:
            aa,_=opt.fit(row[None],x,v,q,family=family,steps=5)
            independent.append(aa['best_bank'][0]);counts['individual_restart_replays']+=1
        np.testing.assert_allclose(arrays['best_bank'],independent,atol=1e-11,rtol=1e-11)
        reordered,_=opt.fit(start,x,v,q[::-1],family=family,steps=5)
        np.testing.assert_array_equal(reordered['best_bank'],arrays['best_bank'])
        np.testing.assert_array_equal(reordered['prediction'],arrays['prediction'][::-1])
        index=min(range(len(start)),key=lambda i:(arrays['best_objective'][i],arrays['best_squared_norm'][i],i))
        assert index==metadata['selected_index']
        for i,point in enumerate(arrays['best_bank']):
            expected=float(exact.objective(x,v,list(map(F,point)),F(.001)))
            assert abs(expected-arrays['best_objective'][i])<1e-12
        if family.startswith('bp_'):assert metadata['whole_chain_jacobian_rows']==len(start)*5
        assert metadata['named_returned_array_bytes']==sum(a.nbytes for a in arrays.values())
        outputs.update({family+'_'+key:value for key,value in arrays.items()})
        counts['family_runs']+=1;counts['query_permutations']+=1
        assert opt.base.BOUND==previous_bound
    np.savez_compressed(OUT/'arrays.npz',**outputs)
    io.verify_hashes(ROOT,hashes)
    summary=dict(passed=True,counts=counts,maxima=maxima,seconds=time.perf_counter()-begin,
        source_sha256=hashes,query_targets_accessed=False,task_risk_compared=False,
        matched_resource_benchmark=False,
        outputs_sha256={f:io.sha(OUT/f) for f in ['protocol.json','arrays.npz']})
    io.save(OUT/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    assert io.read(ROOT/'results/runtime_matched_prefix/report_v1/qa_numeric.json')['passed']
    exact_gate=io.read(ROOT/'results/common_band_objective/math_preflight_v1/summary.json')
    assert exact_gate['passed'];io.verify_hashes(ROOT,exact_gate['source_sha256'])
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

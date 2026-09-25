"""Exact algebra checks for the residual-drift lemma, not a new ML experiment.

Waits for the completed307 pipeline as a guard against competing with timed fit.
All vectors are abstract rational algebra cases, not claimed optimizer states.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:encoded(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encoded(v) for v in value]
    return value


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(encoded(value),f,ensure_ascii=False,indent=2)


def dot(a,b,w):return sum((x*y*z for x,y,z in zip(a,b,w)),F(0))
def inf(a):return max(map(abs,a))
def one(a,w):return sum((abs(x)*y for x,y in zip(a,w)),F(0))


def fold(z):return max(F(0),min(2*z,2-2*z))


def forward(x,b):
    value=x
    for bias in b:value=fold(value+bias)
    return value


def stasis_counterexample():
    x=[F(10+i,100) for i in range(4)]
    teacher=[F(0),F(0),F(0),-F(1,20)]
    v=[forward(z,teacher) for z in x];states=[]
    for t in [-F(1,20),F(1,20)]:
        b=[t,-2*t,F(0),F(0)]
        h=[[2*z-2*t for z in x],[F(1,2)]*4,[F(1)]*4,v.copy()]
        y=[forward(z,b) for z in x];residual=[]
        for j in range(4):
            previous=x if j==0 else h[j-1]
            residual.append([h[j][i]-fold(previous[i]+b[j]) for i in range(4)])
        assert max(map(abs,b))<=F(12,100)
        assert all(0<=z<=1 for layer in h for z in layer)
        assert h[-1]==v and max(abs(a-z) for a,z in zip(y,v))==F(1,10)
        states.append(dict(t=t,b=b,h=h,prediction=y,residual=residual))
    assert states[0]['prediction']==states[1]['prediction']==[F(2,5),F(6,25),F(2,25),F(2,25)]
    assert v==[F(1,2),F(17,50),F(9,50),F(9,50)]
    drift=max(abs(a-b) for l,r in zip(states[0]['residual'],states[1]['residual']) for a,b in zip(l,r))
    assert drift==F(4,5)
    return dict(passed=True,x=x,v=v,teacher=teacher,states=states,prediction_change=F(0),residual_drift=drift,
        claim='Legal four-fold states satisfying forward-stasis criteria do not imply small residual drift',
        not_claimed_reachable_by_frozen_solver=True)


def algebra_cases():
    rng=random.Random(307947);cases=[];counts=Counter()
    for number in range(4096):
        dim=3;t=1+number%9
        w=[rng.choice([F(1),F(1,4)]) for _ in range(dim)]
        r=[F(rng.randint(-4,4),8) for _ in range(dim)]
        d0=[F(rng.randint(-4,4),8) for _ in range(dim)]
        u0=[F(rng.randint(-3,3),8) for _ in range(dim)]
        eta=rng.choice([F(1,4),F(1,2),F(1)])
        zero_drift=number%5==0
        eps=F(0) if zero_drift else rng.choice([F(1,64),F(1,16)])
        residuals=[[z+eps*F(rng.randint(-2,2),2) for z in r] for _ in range(t)]
        drift=[F(0) if zero_drift else F(rng.randint(-2,2),64) for _ in range(dim)]
        dt=[a+b for a,b in zip(d0,drift)];beta=one(drift,w)
        a0=F(rng.randint(-8,8),16)
        a_change=F(0) if zero_drift else F(rng.randint(-4,4),64)
        at=a0+a_change;alpha=max(F(0),a_change)
        ut=[u0[i]+eta*sum((rs[i] for rs in residuals),F(0)) for i in range(dim)]
        delta0=a0+dot(u0,d0,w);kappa=-dot(r,d0,w)
        effective=kappa-eps*one(d0,w)-beta*(inf(r)+eps)
        c=delta0+alpha+inf(u0)*beta
        gap=at+dot(ut,dt,w)
        residual_error=eta*sum((dot([rs[i]-r[i] for i in range(dim)],d0,w) for rs in residuals),F(0))
        reconstructed=delta0-eta*t*kappa+residual_error+(at-a0)+dot(ut,drift,w)
        upper=c-eta*t*effective
        assert max(inf([rs[i]-r[i] for i in range(dim)]) for rs in residuals)<=eps
        assert at-a0<=alpha and one([dt[i]-d0[i] for i in range(dim)],w)<=beta
        assert gap==reconstructed and gap<=upper
        counts['exact_identities']+=1;counts['exact_upper_bounds']+=1
        if zero_drift:
            assert gap==delta0-eta*t*kappa==upper
            counts['fixed_state_reductions']+=1
        threshold=None
        if effective>0:
            threshold=max(0,math.floor(c/(eta*effective))+1)
            assert c-eta*threshold*effective<0
            if threshold>0:assert c-eta*(threshold-1)*effective>=0
            counts['positive_effective_drift']+=1
            if t>=threshold:
                assert gap<0 and upper<0
                counts['certified_negative_gaps']+=1
        cases.append(dict(number=number,weights=w,reference_residual=r,initial_multiplier=u0,
            initial_residual_difference=d0,current_residual_difference=dt,update_residuals=residuals,
            eta=eta,t=t,A0=a0,At=at,epsilon=eps,beta=beta,alpha=alpha,delta0=delta0,
            kappa=kappa,effective_kappa=effective,C=c,gap=gap,reconstructed_gap=reconstructed,
            upper_bound=upper,threshold=threshold,abstract_algebra_case_not_optimizer_trajectory=True))
    assert counts['exact_identities']==counts['exact_upper_bounds']==4096
    assert counts['fixed_state_reductions']>0 and counts['certified_negative_gaps']>0
    return cases,dict(counts)


def alternating_residual():
    # A positive initial alignment is insufficient when update residuals change sign.
    eta=F(1,2);delta0=F(3,4);residuals=[F(1) if i%2==0 else -F(1) for i in range(10)]
    multipliers=[eta*sum(residuals[:t],F(0)) for t in range(11)]
    gaps=[delta0-u for u in multipliers]
    assert min(gaps)==F(1,4) and all(g>0 for g in gaps)
    return dict(passed=True,eta=eta,delta0=delta0,reference_residual=F(1),d0=-F(1),
        residuals=residuals,multipliers=multipliers,gaps=gaps,epsilon=F(2),effective_kappa=-F(1),
        illustrates_missing_coherence_assumption=True,not_an_optimizer_trajectory=True)


def run(root,out):
    design=root/'outputs/ttt-pc-alm-research/307_residual_memory_escape_conditions.md'
    save(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),design_sha256=sha(design),
        cases=4096,seed=307947,exact_fraction_arithmetic=True,reads_query_answers=False,
        experiment='Algebra unit checks, no adaptation or new task evaluation',core_research_goal_complete=False))
    cases,counts=algebra_cases();counterexample=stasis_counterexample();alternating=alternating_residual()
    save(out/'cases.json',cases);save(out/'stasis_counterexample.json',counterexample)
    save(out/'alternating_residual.json',alternating)
    result=dict(passed=True,counts=counts,forward_stasis_counterexample_passed=counterexample['passed'],
        alternating_residual_counterexample_passed=alternating['passed'],actual307_drift_bounds_not_tested=True,
        no_new_task_quality_claim=True,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','cases.json','stasis_counterexample.json','alternating_residual.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();base=root/'results/online_stasis_fresh_pilot'
    # No additional research Python should be started during timed prediction.
    # This read-only guard is also checked here for an accidental early invocation.
    for folder in ['pilot_predictions_v1','pilot_evaluation_v1']:
        summary=json.loads((base/folder/'summary.json').read_text(encoding='utf-8'))
        assert summary['passed'] and summary['tasks']==256
    out=base/'drift_lemma_tests_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise

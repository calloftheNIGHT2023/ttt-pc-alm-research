"""407 independent rational proximal, local-gradient and stationary checks."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import random
import time
import traceback
import common_band_objective_math_v1 as core

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/common_band_objective/math_preflight_v1'
DESIGN='outputs/ttt-pc-alm-research/407_common_band_objective_math_v1.md'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def enumerate_quadratics(a,previous,lo,hi,rho,trust,tau):
    """Independent interval-by-interval minimization, not the prox formula."""
    knots=sorted({F(0),F(1),lo,hi});candidates=set(knots)
    for left,right in zip(knots,knots[1:]):
        midpoint=(left+right)/2
        if midpoint<lo:weight,target=1/tau,lo
        elif midpoint>hi:weight,target=1/tau,hi
        else:weight,target=F(0),F(0)
        stationary=(rho*a+rho*trust*previous+weight*target)/(rho+rho*trust+weight)
        candidates.add(min(right,max(left,stationary)))
    return min(candidates,key=lambda h:core.output_energy(h,a,previous,lo,hi,rho=rho,trust=trust,tau=tau))


def main():
    begin=time.perf_counter();rng=random.Random(407731)
    sources=[Path(__file__),Path(core.__file__),ROOT/DESIGN]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    save(OUT/'protocol.json',dict(source_sha256=hashes,random_seed=407731,
        research_query_targets_accessed=False,constructed_duals_for_diagnostics_only=True,
        candidate_optimizer_implemented=False))
    counts=dict(proximal_exact_cases=0,proximal_kkt_checks=0,local_bias_derivatives=0,
                local_activity_derivatives=0,stationary_credit_cases=0,
                end_to_end_bias_derivatives=0,nonzero_end_to_end_bias_derivatives=0,invalid_inputs_rejected=0)
    for _ in range(256):
        lo,hi=sorted([F(rng.randrange(17),16),F(rng.randrange(17),16)])
        a=F(rng.randrange(-20,41),20);previous=F(rng.randrange(21),20)
        rho=F(rng.randrange(1,10),7);trust=F(rng.randrange(5),10);tau=F(rng.randrange(1,7),3)
        actual=core.output_prox(a,previous,lo,hi,rho=rho,trust=trust,tau=tau)
        expected=enumerate_quadratics(a,previous,lo,hi,rho,trust,tau)
        assert actual==expected
        task_derivative=(actual-lo)/tau if actual<lo else ((actual-hi)/tau if actual>hi else F(0))
        derivative=task_derivative+rho*(actual-a)+rho*trust*(actual-previous)
        assert derivative>=0 if actual==0 else (derivative<=0 if actual==1 else derivative==0)
        counts['proximal_exact_cases']+=1;counts['proximal_kkt_checks']+=1
    eps=F(1,1000);delta=F(1,2**48)
    for depth in [1,2,4]:
        for _ in range(20):
            n=3;x=[F(rng.randrange(1,997),997) for _ in range(n)]
            v=[F(rng.randrange(1,997),997) for _ in range(n)]
            bias=[F(rng.choice([j for j in range(-25,26) if j]),223) for _ in range(depth)]
            h=[[F(rng.randrange(1,997),997) for _ in range(n)] for _ in range(depth)]
            u=[[F(rng.randrange(-20,21),991) for _ in range(n)] for _ in range(depth)]
            rho=F(rng.randrange(1,6),7);tau=F(rng.randrange(1,6),3)
            gb,gh=core.local_partials(x,v,bias,h,u,eps,rho=rho,tau=tau)
            for j in range(depth):
                plus,minus=bias.copy(),bias.copy();plus[j]+=delta;minus[j]-=delta
                numeric=(core.augmented(x,v,plus,h,u,eps,rho=rho,tau=tau)-core.augmented(x,v,minus,h,u,eps,rho=rho,tau=tau))/(2*delta)
                assert gb[j]==numeric;counts['local_bias_derivatives']+=1
                for i in range(n):
                    hp,hm=[row.copy() for row in h],[row.copy() for row in h]
                    hp[j][i]+=delta;hm[j][i]-=delta
                    numeric=(core.augmented(x,v,bias,hp,u,eps,rho=rho,tau=tau)-core.augmented(x,v,bias,hm,u,eps,rho=rho,tau=tau))/(2*delta)
                    assert gh[j][i]==numeric;counts['local_activity_derivatives']+=1
            # Reverse credit is built in the TEST ONLY, never an initialization.
            trace=core.forward_trace(x,bias);lam=[[F(0)]*n for _ in range(depth)]
            lam[-1]=[-core.loss_derivative(y,*core.band(t,eps),tau) for y,t in zip(trace[-1],v)]
            for j in reversed(range(depth-1)):
                lam[j]=[core.slope(trace[j][i]+bias[j+1])*lam[j+1][i] for i in range(n)]
            scaled=[[value/rho for value in row] for row in lam]
            cb,ch=core.local_partials(x,v,bias,trace,scaled,eps,rho=rho,tau=tau)
            assert all(value==0 for row in ch for value in row)
            assert core.augmented(x,v,bias,trace,scaled,eps,rho=rho,tau=tau)==core.objective(x,v,bias,eps,tau=tau)
            for j in range(depth):
                plus,minus=bias.copy(),bias.copy();plus[j]+=delta;minus[j]-=delta
                numeric=(core.objective(x,v,plus,eps,tau=tau)-core.objective(x,v,minus,eps,tau=tau))/(2*delta)
                assert cb[j]==numeric;counts['end_to_end_bias_derivatives']+=1
                counts['nonzero_end_to_end_bias_derivatives']+=int(numeric!=0)
            counts['stationary_credit_cases']+=1
    assert counts['nonzero_end_to_end_bias_derivatives']>0
    lo,hi=F(7,10),F(4,5);a=before=F(1,10)
    common=core.output_prox(a,before,lo,hi)
    hard=core.clip((a+F(1,100)*before)/(1+F(1,100)),lo,hi)
    gap=core.output_energy(hard,a,before,lo,hi)-core.output_energy(common,a,before,lo,hi)
    assert common==F(267,670) and hard==lo and gap>0
    tiny=core.output_prox(a,before,lo,hi,tau=F(1,2**30))
    assert abs(tiny-hard)<=F(101,100)*F(1,2**30)*abs(a-hard)
    invalid=[lambda:core.output_prox(0,0,F(3,4),F(1,4)),
             lambda:core.output_prox(0,0,0,1,rho=0),
             lambda:core.output_prox(0,0,0,1,tau=0),
             lambda:core.output_prox(0,0,0,1,trust=-1),
             lambda:core.band(2,F(1,1000)),lambda:core.slope(F(1,2))]
    for fn in invalid:
        try:fn()
        except AssertionError:counts['invalid_inputs_rejected']+=1
        else:raise AssertionError('Invalid mathematical case accepted')
    save(OUT/'evidence.json',dict(output_prox_exact=str(common),old_hard_projection=str(hard),
        common_objective_energy_improvement=str(gap),finite_tau_differs_from_hard_band=True,
        all_derivative_checks_exact=True,credit_diagnostic_used_in_candidate=False))
    assert all(sha(ROOT/name)==digest for name,digest in hashes.items())
    summary=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,source_sha256=hashes,
        outputs_sha256={f:sha(OUT/f) for f in ['protocol.json','evidence.json']},
        research_query_targets_accessed=False,candidate_optimizer_implemented=False,
        finite_step_query_advantage_established=False)
    save(OUT/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    gate=ROOT/'results/runtime_matched_prefix/report_v1/qa_numeric.json'
    assert gate.is_file() and json.loads(gate.read_text(encoding='utf-8'))['passed'], 'Wait for original 405 chain.'
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

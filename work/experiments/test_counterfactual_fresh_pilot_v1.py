"""294 exact-risk algebra and no-new-query primitive tests."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import counterfactual_fresh_pilot_v1 as pilot
import probe_confirmation_statistics as statistics
from diagnose_gradient_flat_split_states_v1 import sha


def run(root,out):
    hashes=pilot.gate(root);counts=dict(exact_identities=0,sufficient_nonincrease=0,strict_improvements=0,complete_mass_cases=0)
    # Exact rational posterior with three disjoint constant-output regions.
    values=[F(0),F(1,4),F(1,2),F(3,4),F(1)]
    for ai in range(1,10):
        for bi in range(1,11-ai):
            a,b,c=F(ai,10),F(bi,10),F(10-ai-bi,10)
            for ma in values:
                for mb in values:
                    for mc in values:
                        m=a*ma+b*mb+c*mc;tau=b/(a+b);d=mb-ma;e=mc-ma
                        new=ma+tau*d
                        change=(new-m)**2-(ma-m)**2
                        risk=lambda g:a*(g-ma)**2+b*(g-mb)**2+c*(g-mc)**2
                        assert change==risk(new)-risk(ma)==tau*((tau-2*b)*d*d-2*c*d*e)
                        counts['exact_identities']+=1
                        if a+b>=F(1,2) and d*e>=0:
                            assert change<=0;counts['sufficient_nonincrease']+=1
                            if (d!=0 and a+b>F(1,2)) or c*d*e>0:
                                assert change<0;counts['strict_improvements']+=1
                        if not c:
                            assert change==-b*b*d*d;counts['complete_mass_cases']+=1
    assert len(pilot.PILOT_SEEDS)==128 and pilot.PILOT_SEEDS[0]==294000000 and pilot.PILOT_SEEDS[-1]==294000127
    assert not set(pilot.OLD_SEEDS)&set(pilot.PILOT_SEEDS)
    # Only OLD task queries appear in these arithmetic tests.
    gap=0.
    for seed in pilot.OLD_SEEDS:
        q=np.linspace(0,1,257);b=np.random.default_rng(seed).uniform(-.12,.12,4)
        gap=max(gap,float(np.max(abs(pilot.scalar_truth(seed,q)-pilot.forward(q,b[None])[0]))))
    assert gap<2e-12
    selftest=statistics.selftest()
    pilot.exclusive(out/'protocol.json',dict(source_sha256=hashes,design_sha256=sha(root/pilot.DESIGN),
        fresh_query_targets_accessed=False,old_query_seeds=pilot.OLD_SEEDS))
    result=dict(passed=True,counts=counts,old_scalar_teacher_max_gap=gap,bootstrap_selftest=selftest,
        fresh_query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={'protocol.json':sha(out/'protocol.json')})
    pilot.exclusive(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);root=p.parse_args().project.resolve()
    out=root/'results/counterfactual_fresh_pilot/tests_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        pilot.exclusive(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()

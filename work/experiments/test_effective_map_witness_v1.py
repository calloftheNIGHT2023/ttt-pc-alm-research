"""315 exact affine-map witness across a redundant input-region change."""
import argparse
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import random
import numpy as np
from complete_credit_rational_reference_v1 import direct_step,tent
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(root,out):
    domain=[(F(-3,25),F(-2,25)),(F(0),F(1)),(F(-1,50),F(1,50))]
    states=list(product(*domain));rng=random.Random(315071)
    states += [tuple(lo+(hi-lo)*F(rng.randrange(10001),10000) for lo,hi in domain) for _ in range(1024)]
    regions=set()
    for b,h,u in states:
        state=dict(b=[b],h=[[h]],direction=[[u]],x=[F(1,10)],v=[F(2,5)],trust=F(1,100),bound=F(3,25),eps=F(1,1000))
        rr=direct_step(state,1);expected_b=(F(199,500)+2*u+b/100)/F(401,100)
        assert rr['h']==[[F(399,1000)]] and rr['b']==[expected_b]
        actual_u=u+F(1,2)*(rr['h'][0][0]-tent(F(1,10)+rr['b'][0]))
        assert actual_u==u+F(199,2000)-expected_b
        regions.add(sum(F(1,10)+b>=k for k in [F(0),F(1,2),F(1)]))
    assert regions=={0,1}
    assert F(7,101)<F(399,1000)
    assert -F(1,10)<F(3568,40100)<F(4372,40100)<F(3,25)
    assert F(21,1000)**2+F(1,100)*F(22,100)**2<F(379,1000)**2
    with discovery_box(.12):
        runs=[step(np.array([[b]]),np.array([[[.5]]]),np.array([[[0.]]]),np.array([.1]),np.array([.4])) for b in [-.11,-.09]]
    assert not np.array_equal(runs[0]['policies']['activity_0'],runs[1]['policies']['activity_0'])
    gaps=[]
    for b,r in zip([-.11,-.09],runs):
        ideal=np.array([float((F(199,500)+F(b)/100)/F(401,100)),.399])
        gaps.append(float(np.max(abs(np.array([r['b'][0,0],r['h'][0,0,0]])-ideal))))
    assert max(gaps)<1e-14
    final=dict(passed=True,exact_states=len(states),input_regions=sorted(regions),same_exact_affine_formula=True,
        actual_float_policy_differs=True,max_float_decimal_model_gap=max(gaps),
        decimal_rational_witness_distinct_from_binary64_constant_model=True,
        task_observation_realizable=True,real_task_archive_accessed=False,
        full_effective_map_engine_implemented=False,independent_task_gain_established=False,
        source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'solver_policy_trace_v1.py','complete_credit_rational_reference_v1.py']},
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/315_effective_affine_map_design.md'))
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)

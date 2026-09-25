"""312 synthetic domains, exact bounds, affine synergy, and independent updates."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import paired_primal_dual_predictor_v1 as predictor
from test_complete_credit_amplitude_events_v2 import fixtures
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(out):
    rng=np.random.default_rng(312902);checks=0
    for _ in range(512):
        b=rng.uniform(-.12,.12,4);delta=rng.uniform(-.3,.3,4)
        a,exact=predictor.admissible_alpha(b,delta)
        for q,d in zip(b,delta):assert -F(.12)<=F(float(q))+exact*F(float(d))<=F(.12)
        if exact<1:assert any(abs(F(float(q))+exact*F(float(d)))==F(.12) for q,d in zip(b,delta))
        assert F(a)<=exact;checks+=1
    assert predictor.admissible_alpha(np.array([.12]),np.array([.01]))==(0.,F(0))
    assert predictor.admissible_alpha(np.array([.12]),np.array([0.]))==(1.,F(1))
    # Conditional single affine guard example; no assertion of ML generalization.
    gap,p,q=F(3,2),F(1),F(1)
    assert p<gap and q<gap and p+q>gap
    assert p+F(0)<=gap  # no dual contribution is a failure counterexample
    rows=[]
    for name,fixture in fixtures():
        current={k:np.array(fixture[v],dtype=float) for k,v in [('b','b'),('h','h'),('u','direction')]}
        previous={k:a.copy() for k,a in current.items()}
        previous['b']*=.7;previous['u']*=.5
        previous['h']=np.clip(previous['h']+.03,0,1)
        x,v=np.array(fixture['x'],dtype=float),np.array(fixture['v'],dtype=float)
        previous['h'][-1]=np.clip(previous['h'][-1],np.maximum(0,v-.001),np.minimum(1,v+.001))
        result,meta=predictor.evaluate(current,previous,x,v,[312903,0,3,0])
        assert all(not ref['value_discrepancy'] for r in result for ref in r['references'])
        assert all(np.all(abs(np.array(s['b']))<=.12) and np.all((np.array(s['h'])>=0)&(np.array(s['h'])<=1)) for r in result for s in r['states'])
        rows.append(dict(fixture=name,configurations=len(result),local_steps=sum(len(r['references']) for r in result),
            mode_differences=sum(ref['mode_discrepancy'] for r in result for ref in r['references']),
            max_gap=max(max(ref['gaps'].values()) for r in result for ref in r['references'])))
    root=Path(__file__).resolve().parents[2]
    save(out/'summary.json',dict(passed=True,box_cases=checks+2,fixtures=rows,
        conditional_synergy_example=True,no_dual_contribution_counterexample=True,real_task_data_accessed=False,
        source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'paired_primal_dual_predictor_v1.py']}))
    print(dict(passed=True,box_cases=checks+2,fixtures=rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(args.out)
